/*
 * c/model.h — full Qwen3.6-35B-A3B model state and forward pass (M5):
 * embedding -> N decoder layers (c/layer.h, typed by the config's
 * layer_types) -> final RMSNorm -> lm_head -> bf16 logits widened to
 * FP32. C11.
 *
 * Normative reference: docs/M4-CONTRACT.md §1/§6 (emb[id] bf16, no
 * scaling; final RMSNorm is the (1+w) variant; lm_head untied, no bias,
 * bf16 GEMM fp32 accum, logits bf16 values in an fp32 container — the
 * head GEMV below rounds to bf16 then widens, exactly those values).
 *
 * Loader: config.json (the REAL qwen3_5_moe schema — nested text_config,
 * reference/config.json; a bare text config is also accepted, same rule
 * as tools/oracle.py config_from_json) + model.safetensors.index.json in
 * the model dir (the M1 converted container: tools/convert.py copies the
 * source config.json through and writes the standard index alongside
 * apus.index.json). Config key -> engine field mapping (all dims from
 * text_config; defaults per reference/configuration_qwen3_5_moe.py):
 *
 *   hidden_size                    -> hidden (2048)
 *   num_hidden_layers              -> n_layers (40)
 *   vocab_size                     -> vocab (248320, padded)
 *   num_attention_heads            -> attn_nh (16)
 *   num_key_value_heads            -> attn_nkv (2)
 *   head_dim                       -> attn_d (256; default hidden/heads)
 *   linear_num_key_heads           -> gdn_hk (16)
 *   linear_num_value_heads         -> gdn_hv (32; repeat = hv/hk = 2)
 *   linear_key_head_dim            -> gdn_dk (128)
 *   linear_value_head_dim          -> gdn_dv (128)
 *   linear_conv_kernel_dim         -> conv k; MUST be 4 (c/gdn.h pins
 *                                     the k=4 fused causal conv)
 *   moe_intermediate_size          -> moe_inter (512)
 *   shared_expert_intermediate_size-> shared_inter (512)
 *   num_experts                    -> experts (256)
 *   num_experts_per_tok            -> top_k (8)
 *   layer_types                    -> per-layer kind: "linear_attention"
 *                                     = GDN, "full_attention" = gated
 *                                     GQA. Absent: derived from
 *                                     full_attention_interval (default
 *                                     4): full iff (L+1)%interval == 0.
 *   rope_parameters.rope_theta     -> rope_theta (1e7)
 *   rope_parameters.partial_rotary_factor (top-level twin also honored)
 *                                  -> attn_rot = factor*attn_d (64)
 *   eos_token_id                   -> eos_ids[]/n_eos: int OR array of
 *                                     ints (HF allows both; the real
 *                                     config has the single 248044, the
 *                                     generation_config list is [248046,
 *                                     248044] — <|im_end|> reaches the
 *                                     set via the tokenizer lookup in
 *                                     c/apus-qwen.c, M7). eos_id keeps
 *                                     eos_ids[0] (-1 when none)
 *   mtp_num_hidden_layers          -> n_mtp (1 real; M8 loads it — the
 *                                     default load NEVER touches the
 *                                     mtp shard group)
 *   rms_norm_eps                   -> validated: the kernels pin 1e-6
 *                                     (c/attn.h APUS_ATTN_RMS_EPS)
 *
 * Ignored (decorative or training-only, verified in Phase A / the
 * contract): attn_output_gate (the output gate is unconditional),
 * mamba_ssm_dtype, router_aux_loss_coef, output_router_logits,
 * attention_bias/dropout, hidden_act (silu is the only path),
 * max_position_embeddings (max_seq is a load parameter), mrope_section/
 * mrope_interleaved (text-only => mrope is the identity, contract §4),
 * tie_word_embeddings (false; lm_head is always loaded), vision_config,
 * dtype/torch_dtype (the container is pure BF16 by construction).
 *
 * Tensors are resolved through c/st.h ApusStSet (lazy shard open,
 * per-tensor pread materialization) under their REAL M1 container names:
 *
 *   model.language_model.embed_tokens.weight        [V, H]
 *   model.language_model.norm.weight                [H]     ((1+w))
 *   lm_head.weight                                  [V, H]
 *   model.language_model.layers.{L}.input_layernorm.weight /
 *     .post_attention_layernorm.weight              [H]     ((1+w))
 *   ...layers.{L}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,
 *     in_proj_a,out_proj}.weight, .conv1d.weight [C,1,4],
 *     .norm.weight [dv] (DIRECT), .{A_log,dt_bias} [hv]
 *   ...layers.{L}.self_attn.{q,k,v,o}_proj.weight,
 *     .{q,k}_norm.weight [d] ((1+w))
 *   ...layers.{L}.mlp.gate.weight                   [E, H]
 *   ...layers.{L}.mlp.experts.{E}.{gate_up_proj,down_proj}.weight
 *                                     (the M1 2-member slab slices
 *                                     [2I, H] + [H, I])
 *   ...layers.{L}.mlp.shared_expert.{gate,up,down}_proj.weight,
 *     .shared_expert_gate.weight                    [1, H]
 *
 * All weight pointers are zero-copy views into the materialized payloads,
 * stable until apus_model_free, with two ownership exceptions: routed
 * experts in eager mode (apus_model_load) are copied into owned
 * contiguous arrays exp_gu [E,2I,H] / exp_d [E,H,I] (the M1 slab slice
 * IS the fused gate_up layout — a plain concat copy, no repack), and
 * A_log/dt_bias are widened to owned fp32 arrays when the container
 * stores them BF16 (the real checkpoint is pure BF16; the oracle/M1
 * fixtures store F32 — both are accepted, F32 stays zero-copy). The
 * (1+w) RMSNorm gain is applied at runtime per the contract — container
 * bytes untouched. Tiered mode (apus_model_load_ex, tiered=1) leaves
 * the routed experts unresolved and the forward dispatches MoE layers
 * through the c/cache.h expert store (M6).
 *
 * MTP: the loader tolerates the mtp shard group's ABSENCE (n_mtp is
 * recorded from config but nothing mtp.* is resolved here); apus_mtp_load
 * (c/mtp.h, M8) is the only mtp consumer.
 *
 * Forward: per-token through the whole stack (the M4c per-token body per
 * layer), so prefill and decode share numerics by construction and any
 * chunking is bitwise identical (gated in tests/m5). logits_all=0
 * writes the LAST position's logits at logits[0..V) (HF computes the
 * last position only); logits_all=1 writes [T,V].
 *
 * Usage: #define APUS_MODEL_IMPLEMENTATION in exactly one TU (needs the
 * layer.h + st.h + sample-free implementations — see c/apus-qwen.c).
 */
#ifndef APUS_MODEL_H
#define APUS_MODEL_H

#include <stddef.h>
#include <stdint.h>

/* The tiered forward references the c/cache.h store implementation;
 * pull it (and c/compat.h) in automatically when the model
 * implementation is compiled without an explicit APUS_CACHE_IMPLEMENTATION
 * TU, so single-TU users of APUS_MODEL_IMPLEMENTATION link unchanged. */
#if defined(APUS_MODEL_IMPLEMENTATION)
#ifndef APUS_COMPAT_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#endif
#ifndef APUS_CACHE_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION
#endif
#endif

#include "layer.h"
#include "st.h"
#include "cache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ApusLayerCfg lc;
    ApusLayerW   lw;
    /* owned fp32 widens when the container stores A_log/dt_bias as BF16
     * (the real pure-BF16 checkpoint); NULL when the F32 zero-copy view
     * is in use */
    float       *A_log_owned, *dt_bias_owned;
} ApusModelLayer;

typedef struct {
    int n_layers, hidden, vocab, max_seq;
    int eos_id;               /* == eos_ids[0] (-1 when none) */
#define APUS_MODEL_MAX_EOS 8
    int eos_ids[APUS_MODEL_MAX_EOS];  /* config eos_token_id int|array */
    int n_eos;
    int n_mtp;                /* mtp_num_hidden_layers (M8; 0/1) */
    ApusModelLayer *layers;   /* [n_layers], owned */
    ApusStSet *set;           /* owned */
    ApusStore *store;         /* M6: tiered expert store, NOT owned
                                 (attach via apus_model_attach_store) */
    /* top-level tensors: zero-copy views into the shard (BF16),
     * [vocab, hidden] / [hidden] */
    const uint16_t *embed;
    const uint16_t *head;
    const uint16_t *norm_w;
} ApusModel;

typedef struct {
    ApusLayerState *layers;   /* [n_layers], owned */
    int pos;                  /* tokens processed (== layers[0].pos) */
} ApusModelState;

/* Load config.json + the container. max_seq caps the full-attention KV
 * caches and the forward length. tiered != 0 leaves routed-expert tensors
 * unresolved (lw.exp_* = NULL) — attach a c/cache.h store via
 * apus_model_attach_store before forwarding. Returns 0 on success. */
int  apus_model_load_ex(ApusModel *m, const char *dir, int max_seq,
                        int tiered, char *err, size_t errcap);
int  apus_model_load(ApusModel *m, const char *dir, int max_seq,
                     char *err, size_t errcap);
void apus_model_free(ApusModel *m);

/* Attach the tiered expert store (c/cache.h). NOT owned by the model.
 * From here on apus_model_forward dispatches MoE layers through
 * apus_store_layer_forward — bitwise the eager path. */
void apus_model_attach_store(ApusModel *m, ApusStore *store);

void apus_model_state_init(ApusModelState *st, const ApusModel *m);
void apus_model_state_free(ApusModelState *st, const ApusModel *m);

/* ids [T] -> logits. logits_all=0: last position at logits[0..V);
 * logits_all=1: all positions at logits[t*V..]. h_trace (optional,
 * [n_layers*T*H] bf16) captures the hidden state after each layer. */
void apus_model_forward(const ApusModel *m, ApusModelState *st,
                        const int64_t *ids, size_t T, float *logits,
                        int logits_all, uint16_t *h_trace);

/* M8: forward with the final hidden (pre-final-norm, [T,H] bf16) out —
 * the MTP pair input. apus_model_forward == apus_model_forward_h with
 * h_out = NULL. */
void apus_model_forward_h(const ApusModel *m, ApusModelState *st,
                          const int64_t *ids, size_t T, float *logits,
                          int logits_all, uint16_t *h_out);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
#ifdef APUS_MODEL_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static void apus_model_err(char *err, size_t cap, const char *msg,
                           const char *arg) {
    if (err && cap) snprintf(err, cap, msg, arg);
}

static double apus_cfg_num(JVal *cfg, const char *key, double dflt) {
    JVal *v = json_obj_get(cfg, key);
    return v && json_type(v) == J_NUM ? json_num(v) : dflt;
}

static int apus_cfg_int(JVal *cfg, const char *key, int dflt) {
    return (int)apus_cfg_num(cfg, key, dflt);
}

/* Resolve a tensor with dtype + element-count validation. */
static const void *apus_model_get(ApusStSet *set, const char *name,
                                  ApusStDtype dt, int64_t nelem,
                                  char *err, size_t errcap) {
    const ApusStTensor *t = apus_st_set_get(set, name);
    if (!t) {
        apus_model_err(err, errcap, "model: missing tensor %s", name);
        return NULL;
    }
    if (t->dtype != dt || (nelem >= 0 &&
                           (int64_t)apus_st_nelem(t) != nelem)) {
        apus_model_err(err, errcap, "model: bad tensor %s", name);
        return NULL;
    }
    return t->data;
}

/* A_log / dt_bias: fp32 params [hv]. F32 containers keep the zero-copy
 * view; BF16 containers (the real pure-BF16 checkpoint) are widened
 * once into an owned fp32 array (lossless widening, no rounding). */
static const float *apus_model_get_f32param(ApusStSet *set,
                                            const char *name, int64_t n,
                                            float **owned,
                                            char *err, size_t errcap) {
    const ApusStTensor *t = apus_st_set_get(set, name);
    if (!t || (int64_t)apus_st_nelem(t) != n) {
        apus_model_err(err, errcap, "model: missing tensor %s", name);
        return NULL;
    }
    if (t->dtype == APUS_ST_F32) return t->data;
    if (t->dtype == APUS_ST_BF16) {
        float *f = malloc((size_t)n * sizeof(float));
        if (!f) {
            apus_model_err(err, errcap, "model: OOM on %s", name);
            return NULL;
        }
        const uint16_t *b = t->data;
        for (int64_t i = 0; i < n; i++) f[i] = apus_bf16_f32(b[i]);
        *owned = f;
        return f;
    }
    apus_model_err(err, errcap, "model: bad tensor %s", name);
    return NULL;
}

int apus_model_load(ApusModel *m, const char *dir, int max_seq,
                    char *err, size_t errcap) {
    return apus_model_load_ex(m, dir, max_seq, 0, err, errcap);
}

int apus_model_load_ex(ApusModel *m, const char *dir, int max_seq,
                       int tiered, char *err, size_t errcap) {
    memset(m, 0, sizeof *m);
    char path[1200], jerr[128];
    snprintf(path, sizeof path, "%s/config.json", dir);
    JVal *root = json_parse_file(path, jerr, sizeof jerr);
    if (!root) {
        apus_model_err(err, errcap, "model: config: %s", jerr);
        return -1;
    }
    /* the real qwen3_5_moe schema nests the text model under
     * text_config; a bare text config is accepted too (oracle rule) */
    JVal *cfg = json_obj_get(root, "text_config");
    if (!cfg || json_type(cfg) != J_OBJ) cfg = root;

    int n_layers = apus_cfg_int(cfg, "num_hidden_layers", 0);
    int H = apus_cfg_int(cfg, "hidden_size", 0);
    int V = apus_cfg_int(cfg, "vocab_size", 0);
    int heads = apus_cfg_int(cfg, "num_attention_heads", 0);
    int head_dim = apus_cfg_int(cfg, "head_dim",
                                heads > 0 ? H / heads : 0);
    int E = apus_cfg_int(cfg, "num_experts", 0);
    int gdn_hk = apus_cfg_int(cfg, "linear_num_key_heads", 0);
    int gdn_hv = apus_cfg_int(cfg, "linear_num_value_heads", 0);
    int gdn_dk = apus_cfg_int(cfg, "linear_key_head_dim", 0);
    int gdn_dv = apus_cfg_int(cfg, "linear_value_head_dim", 0);
    int conv_k = apus_cfg_int(cfg, "linear_conv_kernel_dim", 4);
    if (n_layers <= 0 || H <= 0 || V <= 0 || heads <= 0 ||
        head_dim <= 0 || E <= 0 || gdn_hk <= 0 || gdn_hv <= 0 ||
        gdn_dk <= 0 || gdn_dv <= 0) {
        apus_model_err(err, errcap, "model: bad config.json in %s", dir);
        json_free(root);
        return -1;
    }
    if (conv_k != 4) {
        /* c/gdn.h pins the fused causal conv to k=4 (contract §3) */
        apus_model_err(err, errcap,
                       "model: linear_conv_kernel_dim != 4 in %s", dir);
        json_free(root);
        return -1;
    }
    double eps = apus_cfg_num(cfg, "rms_norm_eps", 1e-6);
    if (fabs(eps - 1e-6) > 1e-12) {
        /* the kernels pin APUS_ATTN_RMS_EPS = 1e-6 (c/attn.h) */
        apus_model_err(err, errcap, "model: rms_norm_eps != 1e-6 in %s",
                       dir);
        json_free(root);
        return -1;
    }
    /* rope: rope_parameters object (real schema); the top-level
     * partial_rotary_factor twin is honored as a fallback */
    JVal *rp = json_obj_get(cfg, "rope_parameters");
    if (rp && json_type(rp) != J_OBJ) rp = NULL;
    double theta = rp ? apus_cfg_num(rp, "rope_theta", 10000000.0)
                      : 10000000.0;
    double prf = rp ? apus_cfg_num(rp, "partial_rotary_factor", -1.0)
                    : -1.0;
    if (prf < 0) prf = apus_cfg_num(cfg, "partial_rotary_factor", 0.25);

    /* layer_types: explicit array (real schema) or derived from
     * full_attention_interval (configuration_qwen3_5_moe.py L123-134) */
    JVal *lt = json_obj_get(cfg, "layer_types");
    int have_lt = lt && json_type(lt) == J_ARR &&
                  json_arr_len(lt) == (size_t)n_layers;
    int interval = apus_cfg_int(cfg, "full_attention_interval", 4);
    if (interval <= 0) interval = 4;

    m->n_layers = n_layers;
    m->hidden = H;
    m->vocab = V;
    m->max_seq = max_seq > 0 ? max_seq : 4096;
    /* eos_token_id: HF allows an int or an array of ints */
    m->n_eos = 0;
    {
        JVal *ej = json_obj_get(cfg, "eos_token_id");
        if (ej && json_type(ej) == J_ARR) {
            size_t ne = json_arr_len(ej);
            for (size_t i = 0; i < ne &&
                            m->n_eos < APUS_MODEL_MAX_EOS; i++) {
                JVal *v = json_arr_get(ej, i);
                if (v && json_type(v) == J_NUM)
                    m->eos_ids[m->n_eos++] = (int)json_num(v);
            }
        } else {
            int e = apus_cfg_int(cfg, "eos_token_id", -1);
            if (e >= 0) m->eos_ids[m->n_eos++] = e;
        }
    }
    m->eos_id = m->n_eos ? m->eos_ids[0] : -1;
    m->n_mtp = apus_cfg_int(cfg, "mtp_num_hidden_layers", 0);
    m->layers = calloc((size_t)n_layers, sizeof *m->layers);

    m->set = apus_st_set_open(dir, err, errcap);
    if (!m->set) {
        json_free(root);
        free(m->layers);
        return -1;
    }

    for (int L = 0; L < n_layers; L++) {
        ApusModelLayer *ml = &m->layers[L];
        ApusLayerCfg *c = &ml->lc;
        int full;
        if (have_lt) {
            JVal *s = json_arr_get(lt, (size_t)L);
            const char *t = s && json_type(s) == J_STR ? json_str(s) : "";
            if (!strcmp(t, "full_attention")) full = 1;
            else if (!strcmp(t, "linear_attention")) full = 0;
            else {
                apus_model_err(err, errcap,
                               "model: bad layer_types entry in %s", dir);
                goto fail;
            }
        } else {
            full = ((L + 1) % interval) == 0;
        }
        c->kind = full ? APUS_LAYER_FULL : APUS_LAYER_GDN;
        c->hidden = (size_t)H;
        c->gdn_hk = (size_t)gdn_hk;
        c->gdn_hv = (size_t)gdn_hv;
        c->gdn_dk = (size_t)gdn_dk;
        c->gdn_dv = (size_t)gdn_dv;
        c->attn_nh = (size_t)heads;
        c->attn_nkv = (size_t)apus_cfg_int(cfg, "num_key_value_heads",
                                           heads);
        c->attn_d = (size_t)head_dim;
        c->attn_rot = (size_t)(prf * head_dim + 0.5);
        c->rope_theta = theta;
        c->max_seq = (size_t)m->max_seq;
        c->experts = (size_t)E;
        c->moe_inter = (size_t)apus_cfg_int(cfg, "moe_intermediate_size",
                                            0);
        c->shared_inter = (size_t)apus_cfg_int(
            cfg, "shared_expert_intermediate_size", 0);
        c->top_k = (size_t)apus_cfg_int(cfg, "num_experts_per_tok", 8);

        ApusLayerW *w = &ml->lw;
        char nm[256];
#define GET16(field, fmt, ne) do { \
            snprintf(nm, sizeof nm, fmt, L); \
            w->field = apus_model_get(m->set, nm, APUS_ST_BF16, ne, \
                                      err, errcap); \
            if (!w->field) goto fail; \
        } while (0)
        int64_t key_dim = (int64_t)gdn_hk * gdn_dk;
        int64_t value_dim = (int64_t)gdn_hv * gdn_dv;
        int64_t conv_dim = 2 * key_dim + value_dim;
        GET16(ln1_w,
              "model.language_model.layers.%d.input_layernorm.weight", H);
        GET16(ln2_w,
              "model.language_model.layers.%d.post_attention_layernorm"
              ".weight", H);
        if (c->kind == APUS_LAYER_GDN) {
            GET16(w_qkv,
                  "model.language_model.layers.%d.linear_attn."
                  "in_proj_qkv.weight", conv_dim * H);
            GET16(w_z,
                  "model.language_model.layers.%d.linear_attn."
                  "in_proj_z.weight", value_dim * H);
            GET16(w_b,
                  "model.language_model.layers.%d.linear_attn."
                  "in_proj_b.weight", (int64_t)gdn_hv * H);
            GET16(w_a,
                  "model.language_model.layers.%d.linear_attn."
                  "in_proj_a.weight", (int64_t)gdn_hv * H);
            /* [conv_dim, 1, K] in the container — nelem conv_dim*K */
            GET16(conv_w,
                  "model.language_model.layers.%d.linear_attn."
                  "conv1d.weight", conv_dim * 4);
            GET16(onorm_w,
                  "model.language_model.layers.%d.linear_attn."
                  "norm.weight", (int64_t)gdn_dv);
            GET16(w_out,
                  "model.language_model.layers.%d.linear_attn."
                  "out_proj.weight", H * value_dim);
            snprintf(nm, sizeof nm,
                     "model.language_model.layers.%d.linear_attn.A_log",
                     L);
            w->A_log = apus_model_get_f32param(m->set, nm, gdn_hv,
                                               &ml->A_log_owned,
                                               err, errcap);
            if (!w->A_log) goto fail;
            snprintf(nm, sizeof nm,
                     "model.language_model.layers.%d.linear_attn.dt_bias",
                     L);
            w->dt_bias = apus_model_get_f32param(m->set, nm, gdn_hv,
                                                 &ml->dt_bias_owned,
                                                 err, errcap);
            if (!w->dt_bias) goto fail;
        } else {
            int64_t d = (int64_t)c->attn_d;
            GET16(wq,
                  "model.language_model.layers.%d.self_attn.q_proj.weight",
                  (int64_t)c->attn_nh * 2 * d * H);
            GET16(wk,
                  "model.language_model.layers.%d.self_attn.k_proj.weight",
                  (int64_t)c->attn_nkv * d * H);
            GET16(wv,
                  "model.language_model.layers.%d.self_attn.v_proj.weight",
                  (int64_t)c->attn_nkv * d * H);
            GET16(wo,
                  "model.language_model.layers.%d.self_attn.o_proj.weight",
                  H * (int64_t)c->attn_nh * d);
            GET16(qn_w,
                  "model.language_model.layers.%d.self_attn.q_norm.weight",
                  d);
            GET16(kn_w,
                  "model.language_model.layers.%d.self_attn.k_norm.weight",
                  d);
        }
        {
            GET16(rtr_w,
                  "model.language_model.layers.%d.mlp.gate.weight",
                  (int64_t)E * H);
            GET16(sh_gate,
                  "model.language_model.layers.%d.mlp."
                  "shared_expert_gate.weight", H);
            /* M6 tiered-store insertion point: per-expert resolution.
             * Tiered mode leaves exp_* NULL — the forward then goes
             * through apus_store_layer_forward (c/cache.h), which reads
             * the same slab bytes from the store. */
            if (tiered) {
                w->exp_gu = w->exp_d = NULL;
            } else {
            int64_t I = (int64_t)c->moe_inter;
            uint16_t *egu = malloc((size_t)E * 2 * I * H * 2);
            uint16_t *ed = malloc((size_t)E * H * I * 2);
            if (!egu || !ed) {
                free(egu); free(ed);
                apus_model_err(err, errcap, "model: OOM on experts %s",
                               nm);
                goto fail;
            }
            /* M1 slab slices: gate_up_proj.weight [2I, H] is ALREADY the
             * fused [gate|up] layout — a plain concat copy per expert,
             * no repack, bytes untouched. */
            for (int64_t e = 0; e < E; e++) {
                const uint16_t *t;
                snprintf(nm, sizeof nm,
                         "model.language_model.layers.%d.mlp.experts.%lld."
                         "gate_up_proj.weight", L, (long long)e);
                t = apus_model_get(m->set, nm, APUS_ST_BF16, 2 * I * H,
                                   err, errcap);
                if (!t) { free(egu); free(ed); goto fail; }
                memcpy(egu + e * 2 * I * H, t, (size_t)2 * I * H * 2);
                snprintf(nm, sizeof nm,
                         "model.language_model.layers.%d.mlp.experts.%lld."
                         "down_proj.weight", L, (long long)e);
                t = apus_model_get(m->set, nm, APUS_ST_BF16, H * I,
                                   err, errcap);
                if (!t) { free(egu); free(ed); goto fail; }
                memcpy(ed + e * H * I, t, (size_t)H * I * 2);
            }
            w->exp_gu = egu;
            w->exp_d = ed;
            }   /* tiered */
            int64_t Is = (int64_t)c->shared_inter;
            GET16(sh_g,
                  "model.language_model.layers.%d.mlp.shared_expert."
                  "gate_proj.weight", Is * H);
            GET16(sh_u,
                  "model.language_model.layers.%d.mlp.shared_expert."
                  "up_proj.weight", Is * H);
            GET16(sh_d,
                  "model.language_model.layers.%d.mlp.shared_expert."
                  "down_proj.weight", H * Is);
        }
    }
#undef GET16
    {
        const ApusStTensor *t;
        t = apus_st_set_get(m->set,
                            "model.language_model.embed_tokens.weight");
        if (!t || t->dtype != APUS_ST_BF16 ||
            (int64_t)apus_st_nelem(t) != (int64_t)V * H)
            goto fail_top;
        m->embed = t->data;
        t = apus_st_set_get(m->set, "lm_head.weight");
        if (!t || t->dtype != APUS_ST_BF16 ||
            (int64_t)apus_st_nelem(t) != (int64_t)V * H)
            goto fail_top;
        m->head = t->data;
        t = apus_st_set_get(m->set, "model.language_model.norm.weight");
        if (!t || t->dtype != APUS_ST_BF16 ||
            (int64_t)apus_st_nelem(t) != H)
            goto fail_top;
        m->norm_w = t->data;
    }
    json_free(root);
    return 0;

fail_top:
    apus_model_err(err, errcap, "model: bad top-level tensors in %s", dir);
fail:
    json_free(root);
    apus_model_free(m);
    return -1;
}

void apus_model_free(ApusModel *m) {
    if (!m) return;
    if (m->layers) {
        for (int L = 0; L < m->n_layers; L++) {
            ApusModelLayer *ml = &m->layers[L];
            /* expert arrays are owned copies (M6: lazy views instead) */
            free((void *)ml->lw.exp_gu);
            free((void *)ml->lw.exp_d);
            free(ml->A_log_owned);
            free(ml->dt_bias_owned);
        }
        free(m->layers);
    }
    if (m->set) apus_st_set_close(m->set);
    memset(m, 0, sizeof *m);
}

void apus_model_attach_store(ApusModel *m, ApusStore *store) {
    m->store = store;
}

void apus_model_state_init(ApusModelState *st, const ApusModel *m) {
    memset(st, 0, sizeof *st);
    st->layers = calloc((size_t)m->n_layers, sizeof *st->layers);
    for (int L = 0; L < m->n_layers; L++) {
        const ApusLayerCfg *c = &m->layers[L].lc;
        ApusLayerState *s = &st->layers[L];
        if (c->kind == APUS_LAYER_GDN) {
            size_t cd = 2 * c->gdn_hk * c->gdn_dk
                        + c->gdn_hv * c->gdn_dv;
            s->conv_state = calloc(cd * 3, 2);
            s->S = calloc(c->gdn_hv * c->gdn_dk * c->gdn_dv,
                          sizeof(float));
        } else {
            s->kcache = calloc(c->max_seq * c->attn_nkv * c->attn_d, 2);
            s->vcache = calloc(c->max_seq * c->attn_nkv * c->attn_d, 2);
        }
        apus_layer_state_zero(c, s);
    }
    st->pos = 0;
}

void apus_model_state_free(ApusModelState *st, const ApusModel *m) {
    if (!st->layers) return;
    for (int L = 0; L < m->n_layers; L++) {
        ApusLayerState *s = &st->layers[L];
        free(s->conv_state); free(s->S);
        free(s->kcache); free(s->vcache);
    }
    free(st->layers);
    st->layers = NULL;
}

static void apus_model_forward_impl(const ApusModel *m, ApusModelState *st,
                                    const int64_t *ids, size_t T,
                                    float *logits, int logits_all,
                                    uint16_t *h_trace, uint16_t *h_out) {
    size_t H = (size_t)m->hidden, V = (size_t)m->vocab;
    ApusScratchMark mk = apus_scratch_mark();
    uint16_t *xa = apus_scratch_alloc(T * H * 2);
    uint16_t *xb = apus_scratch_alloc(T * H * 2);
    for (size_t t = 0; t < T; t++)
        memcpy(xa + t * H, m->embed + (size_t)ids[t] * H,
               H * 2);                         /* §1: no scaling */
    if (m->store)
        apus_store_fwd_set_batch(m->store, (int)T, 0);
    for (int L = 0; L < m->n_layers; L++) {
        /* every layer is MoE in this model — the tiered store serves
         * all of them */
        if (m->store)
            apus_store_layer_forward(m->store, L, &m->layers[L].lc,
                                     &m->layers[L].lw,
                                     &st->layers[L], xa, xb, T);
        else
            apus_layer_forward_hot(&m->layers[L].lc, &m->layers[L].lw,
                                   &st->layers[L], xa, xb, T, NULL);
        uint16_t *tmp = xa;
        xa = xb;
        xb = tmp;
        if (h_trace)
            memcpy(h_trace + (size_t)L * T * H, xa, T * H * 2);
    }
    st->pos += (int)T;
    if (h_out)
        memcpy(h_out, xa, T * H * 2);   /* pre-final-norm hidden (M8) */
    /* lm_head per needed position (§6: bf16 out, widened) */
    uint16_t yn[H], lb[V];
    float *lf = apus_scratch_alloc(V * sizeof(float));
    for (size_t t = logits_all ? 0 : T - 1; t < T; t++) {
        apus_attn_rmsnorm(xa + t * H, m->norm_w, yn, H);
        apus_bf16_gemv_hot(m->head, yn, lb, V, H);
        float *dst = logits + (logits_all ? t * V : 0);
#ifdef __ARM_NEON
        if (lf) {
            apus_bf16_widen_neon(lb, lf, V);    /* exact widening */
            memcpy(dst, lf, V * sizeof(float));
        } else
#elif APUS_X86
        /* M12a-2: the AVX2 widen on the logits path (exact — the same
         * contract as the NEON branch; c/x86.h). */
        if (lf && apus_x86_have_avx2()) {
            apus_bf16_widen_x86(lb, lf, V);
            memcpy(dst, lf, V * sizeof(float));
        } else
#endif
        for (size_t v = 0; v < V; v++)
            dst[v] = apus_bf16_f32(lb[v]);
    }
    apus_scratch_reset(mk);
}

void apus_model_forward(const ApusModel *m, ApusModelState *st,
                        const int64_t *ids, size_t T, float *logits,
                        int logits_all, uint16_t *h_trace) {
    apus_model_forward_impl(m, st, ids, T, logits, logits_all, h_trace,
                            NULL);
}

void apus_model_forward_h(const ApusModel *m, ApusModelState *st,
                          const int64_t *ids, size_t T, float *logits,
                          int logits_all, uint16_t *h_out) {
    apus_model_forward_impl(m, st, ids, T, logits, logits_all, NULL,
                            h_out);
}

#endif /* APUS_MODEL_IMPLEMENTATION */
#endif /* APUS_MODEL_H */
