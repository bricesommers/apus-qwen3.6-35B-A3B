/*
 * c/apus-qwen.c — CLI driver for the apus-qwen engine. Single
 * translation unit: defines every APUS_*_IMPLEMENTATION macro.
 *
 *   bin/apus-qwen run --model DIR [--prompt "text" | --ids "1,2,3"]
 *       [--max-tokens N] [--seed S] [--temp T] [--top-p P] [--top-k K]
 *       [--greedy] [--max-seq N] [--tiered]
 *
 *   bin/apus-qwen serve --model DIR [--tiered]     (M7a: NDJSON stdio
 *       protocol, see tests/m7a/README.md; driven by tools/server.py)
 *
 * --prompt renders the Qwen ChatML template (c/encoding.h, the M2 port
 * of reference/chat_template.jinja) and tokenizes via
 * DIR/tokenizer.json (c/tok.h); --ids feeds raw ids (for the synthetic
 * M5 fixture model, whose 256-row vocab has no tokenizer). Sampling
 * defaults are the Qwen generation_config defaults: temp 1.0,
 * top_p 0.95, top_k 20; --greedy forces argmax. Prints generated ids
 * (decoded text with a tokenizer) and prefill/decode tok/s.
 *
 * EOS (M7): generation stops on ANY of the model's eos ids — the
 * config.json eos_token_id (int or array; the real config has 248044
 * <|endoftext|>) plus <|im_end|> (248046) resolved through the
 * tokenizer, mirroring the generation_config list [248046, 248044].
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_GDN_IMPLEMENTATION
#define APUS_ATTN_IMPLEMENTATION
#define APUS_MOE_IMPLEMENTATION
#define APUS_LAYER_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_CACHE_IMPLEMENTATION
#define APUS_PILOT_IMPLEMENTATION
#define APUS_MODEL_IMPLEMENTATION
#define APUS_SAMPLE_IMPLEMENTATION
#define APUS_MTP_IMPLEMENTATION
#define APUS_TOK_IMPLEMENTATION
#define APUS_ENCODING_IMPLEMENTATION
#include "json.h"
#include "st.h"
#include "bf16.h"
#include "gdn.h"
#include "attn.h"
#include "moe.h"
#include "layer.h"
#include "compat.h"
#include "cache.h"
#include "pilot.h"
#include "model.h"
#include "sample.h"
#include "mtp.h"
#include "tok.h"
#include "encoding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <io.h>     /* _setmode (M13) */
#include <fcntl.h>  /* _O_BINARY */
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(FILE *f) {
    fprintf(f,
        "apus-qwen run --model DIR [--prompt TEXT | --ids \"1,2,3\"]\n"
        "    [--max-tokens N] [--seed S] [--temp T] [--top-p P]\n"
        "    [--top-k K] [--greedy] [--max-seq N] [--tiered] [--metal]\n"
        "    [--spec [--spec-k K]]   (M8 MTP speculative decoding, "
        "default OFF, K=2)\n"
        "apus-qwen serve --model DIR [--tiered] [--max-seq N]\n"
        "    (NDJSON stdio protocol; driven by tools/server.py;\n"
        "    max_seq default 32768, env APUS_MAX_SEQ)\n");
}

/* ---- engine context shared by `run` and `serve` (M7a) ------------------ */

typedef struct {
    ApusModel  m;
    ApusStore *store;     /* tiered only; owned */
    ApusPilot *pilot;     /* tiered only; owned */
    Tok       *tok;       /* NULL when the model dir has no tokenizer */
    ApusMtpW  mtp_w;      /* spec only */
    int       spec;
    int       im_end_id;  /* <|im_end|> via the tokenizer, -1 when unknown
                             (M7: second EOS of generation_config's
                             [248046, 248044] list) */
} Engine;

/* M7: stop on ANY eos id — config eos_ids[] plus the tokenizer's
 * <|im_end|>. */
static int engine_is_eos(const Engine *e, int t) {
    for (int i = 0; i < e->m.n_eos; i++)
        if (t == e->m.eos_ids[i]) return 1;
    return t >= 0 && t == e->im_end_id;
}

static int engine_init(Engine *e, const char *model_dir, int max_seq,
                       int tiered, int spec, char *err, size_t errcap) {
    memset(e, 0, sizeof *e);
    e->im_end_id = -1;
    if (apus_model_load_ex(&e->m, model_dir, max_seq, tiered, err, errcap))
        return -1;
    e->spec = spec && e->m.n_mtp > 0;
    if (spec && !e->m.n_mtp) {
        snprintf(err, errcap, "%s", "model has no MTP layer "
                 "(mtp_num_hidden_layers == 0)");
        apus_model_free(&e->m);
        return -1;
    }
    if (tiered) {
        ApusStoreCfg sc = {0};
        sc.n_layers = e->m.n_layers + (e->spec ? e->m.n_mtp : 0);
        sc.n_main_layers = e->m.n_layers;   /* M8: mtp.layers.{K} ->
                                               store layer n_main + K */
        sc.n_experts = 0;
        if (e->m.n_layers > 0)   /* every layer is MoE in this model */
            sc.n_experts = (int)e->m.layers[0].lc.experts;
        e->store = apus_store_open(model_dir, &sc, err, errcap);
        if (!e->store) { apus_model_free(&e->m); return -1; }
        apus_model_attach_store(&e->m, e->store);
        if (apus_env_int("APUS_PILOT", 1)) {
            const ApusLayerCfg *mc = &e->m.layers[0].lc;
            ApusPilotCfg pc = {0};
            pc.store = e->store;
            pc.n_layers = e->m.n_layers;
            pc.n_experts = sc.n_experts;
            pc.top_k = (int)mc->top_k;
            pc.hidden = e->m.hidden;
            pc.enabled = 1;
            pc.pilot_k = apus_env_int("APUS_PILOT_K", 8);
            pc.ring_entries = (size_t)apus_env_int("APUS_PILOT_RING", 4096);
            pc.dump_path = getenv("APUS_PILOT_DUMP");
            e->pilot = apus_pilot_create(&pc);
            if (e->pilot) {
                for (int L = 0; L < e->m.n_layers; L++) {
                    const ApusLayerW *lw = &e->m.layers[L].lw;
                    ApusPilotRouter r = { lw->rtr_w, lw->ln2_w };
                    apus_pilot_attach_router(e->pilot, L, &r);
                }
                ApusStoreFwdHooks hooks;
                apus_pilot_store_hooks(e->pilot, &hooks);
                apus_store_fwd_hooks(e->store, &hooks);
                apus_pilot_start(e->pilot);
            }
        }
    }
    if (e->spec && apus_mtp_load(&e->m, &e->mtp_w, tiered, err, errcap)) {
        if (e->pilot) apus_pilot_destroy(e->pilot);
        if (e->store) apus_store_close(e->store);
        apus_model_free(&e->m);
        return -1;
    }
    /* tokenizer is optional (synthetic models feed ids) */
    char tpath[1200];
    snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model_dir);
    FILE *tf = fopen(tpath, "rb");
    if (tf) {
        fclose(tf);
        e->tok = tok_load(tpath);
        if (!e->tok) {
            snprintf(err, errcap, "cannot load %s", tpath);
            if (e->store) apus_store_close(e->store);
            if (e->pilot) apus_pilot_destroy(e->pilot);
            apus_model_free(&e->m);
            return -1;
        }
        e->im_end_id = (int)tok_token_to_id(e->tok, "<|im_end|>");
    }
    return 0;
}

static void engine_close(Engine *e) {
    if (e->spec) apus_mtp_free(&e->mtp_w);
    if (e->pilot) {
        ApusPilotStats ps;
        apus_pilot_stats(e->pilot, &ps);
        fprintf(stderr,
                "pilot: predictions %llu, enqueued %llu (full-drops %llu), "
                "issued %llu (stale %llu), recall %llu/%llu\n",
                (unsigned long long)ps.predictions,
                (unsigned long long)ps.hints_enqueued,
                (unsigned long long)ps.hints_dropped_full,
                (unsigned long long)ps.hints_issued,
                (unsigned long long)ps.hints_dropped_stale,
                (unsigned long long)ps.actual_hits,
                (unsigned long long)ps.actual_experts);
        apus_pilot_destroy(e->pilot);
    }
    if (e->store) {
        ApusStoreStats ss;
        apus_store_stats(e->store, &ss);
        apus_store_save_usage(e->store);
        fprintf(stderr,
                "store: hits %llu misses %llu preads %llu (hint %llu, "
                "demand %llu) evictions %llu rss_drops %llu waits %llu\n",
                (unsigned long long)ss.hits, (unsigned long long)ss.misses,
                (unsigned long long)ss.preads,
                (unsigned long long)ss.hint_loads,
                (unsigned long long)ss.demand_loads,
                (unsigned long long)ss.evictions,
                (unsigned long long)ss.rss_drops,
                (unsigned long long)ss.waits);
        apus_store_close(e->store);
    }
    apus_model_free(&e->m);
    if (e->tok) tok_free(e->tok);
}

/* ================= interactive `run` ================= */

/* M10: opt-in Metal backend (APUS_METAL=1 / --metal). Fail-soft: when the
 * backend isn't compiled in (plain CPU binary, Linux) or init fails, the
 * engine continues on the CPU kernels with a note on stderr. */
static void apus_cli_metal_init(void) {
    char merr[256];
    if (apus_metal_enable(merr, sizeof merr))
        fprintf(stderr, "metal: %s — continuing on CPU\n", merr);
    else
        fprintf(stderr, "metal: backend enabled (wrapped %llu B so far)\n",
                (unsigned long long)apus_metal_bytes_wrapped());
}

static int run_main(int argc, char **argv) {
    const char *model_dir = NULL, *prompt = NULL, *ids_arg = NULL;
    int max_tokens = 16, greedy = 0, max_seq = 4096;
    int tiered = 0;
    int spec = 0, spec_k = 2;
    int metal = 0;
    uint64_t seed = 0;
    float temp = 1.0f, top_p = 0.95f;   /* Qwen generation_config */
    int top_k = 20;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(a, "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(a, "--ids") && i + 1 < argc) ids_arg = argv[++i];
        else if (!strcmp(a, "--max-tokens") && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && i + 1 < argc)
            seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--temp") && i + 1 < argc)
            temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-p") && i + 1 < argc)
            top_p = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--greedy")) greedy = 1;
        else if (!strcmp(a, "--tiered")) tiered = 1;
        else if (!strcmp(a, "--metal")) metal = 1;
        else if (!strcmp(a, "--spec")) spec = 1;
        else if (!strcmp(a, "--spec-k") && i + 1 < argc)
            spec_k = atoi(argv[++i]);
        else if (!strcmp(a, "--max-seq") && i + 1 < argc)
            max_seq = atoi(argv[++i]);
        else {
            fprintf(stderr, "unknown/incomplete arg: %s\n", a);
            usage(stderr);
            return 2;
        }
    }
    if (apus_env_int("APUS_TIERED", 0)) tiered = 1;
    if (apus_env_int("APUS_SPEC", 0)) spec = 1;
    spec_k = apus_env_int("APUS_SPEC_K", spec_k);
    if (apus_env_int("APUS_METAL", 0)) metal = 1;
    if (metal) apus_cli_metal_init();
    if (!model_dir || (!prompt && !ids_arg)) {
        usage(stderr);
        return 2;
    }
    if (greedy) temp = 0.0f;

    Engine e;
    char err[256];
    double t0 = now_s();
    if (engine_init(&e, model_dir, max_seq, tiered, spec, err,
                    sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    fprintf(stderr, "loaded %s (%d layers, hidden %d, vocab %d%s) in %.2fs\n",
            model_dir, e.m.n_layers, e.m.hidden, e.m.vocab,
            tiered ? ", tiered" : "", now_s() - t0);

    /* input ids */
    int64_t ids[8192];
    size_t n_ids = 0;
    if (ids_arg) {
        const char *p = ids_arg;
        while (*p && n_ids < 8192) {
            ids[n_ids++] = strtoll(p, (char **)&p, 10);
            if (*p == ',') p++;
        }
    } else {
        if (!e.tok) {
            fprintf(stderr, "no tokenizer in model dir (use --ids)\n");
            engine_close(&e);
            return 1;
        }
        JVal *msgs = json_new_arr();
        JVal *msg = json_new_obj();
        json_obj_set(msg, "role", json_new_str("user"));
        json_obj_set(msg, "content", json_new_str(prompt));
        json_arr_push(msgs, msg);
        LingEncOpts opts = LING_ENC_OPTS_DEFAULT;
        size_t n = 0;
        uint32_t *tid = ling_encode_ids(e.tok, msgs, NULL, &opts, &n);
        json_free(msgs);
        if (!tid || n == 0 || n > 8192) {
            fprintf(stderr, "prompt encode failed: %s\n",
                    ling_last_error());
            engine_close(&e);
            return 1;
        }
        for (size_t i = 0; i < n; i++) ids[i] = tid[i];
        n_ids = n;
        free(tid);
    }
    if (n_ids == 0) {
        fprintf(stderr, "empty input\n");
        engine_close(&e);
        return 1;
    }

    ApusModelState st;
    apus_model_state_init(&st, &e.m);
    float *logits = malloc((size_t)e.m.vocab * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)e.m.vocab));
    ApusRng rng;
    apus_rng_seed(&rng, seed);

    t0 = now_s();
    int n_gen = 0;
    double t_dec0;
    if (spec) {
        /* M8: draft/verify decode; the emitted stream is bitwise the
         * non-speculative one for the same seed */
        int *tokens = malloc((size_t)(max_tokens ? max_tokens : 1)
                             * sizeof(int));
        ApusSpecStats ss;
        t0 = now_s();
        n_gen = apus_spec_run(&e.m, &e.mtp_w, &st, e.store, ids, n_ids,
                              max_tokens, temp, top_k, top_p, seed,
                              spec_k, NULL, NULL, tokens, &ss);
        double t_all = now_s() - t0;
        t_dec0 = t0;
        for (int g = 0; g < n_gen; g++) {
            if (e.tok) {
                uint32_t id = (uint32_t)tokens[g];
                size_t dl;
                char *s2 = tok_decode(e.tok, &id, 1, &dl);
                if (s2) { fwrite(s2, 1, dl, stdout); free(s2); }
            } else {
                printf("%d\n", tokens[g]);
            }
        }
        fflush(stdout);
        fprintf(stderr,
                "spec k=%d: %d tokens in %.2fs (%.2f tok/s), %llu steps "
                "(%.2f tok/batch), accepted %llu/%llu drafts (%.1f%%), "
                "%llu full, %llu re-feeds\n",
                spec_k, n_gen, t_all,
                n_gen / (t_all > 0 ? t_all : 1e-9),
                (unsigned long long)ss.steps,
                ss.steps ? (double)n_gen / (double)ss.steps : 0.0,
                (unsigned long long)ss.accepted,
                (unsigned long long)ss.drafts,
                ss.drafts ? 100.0 * (double)ss.accepted
                              / (double)ss.drafts : 0.0,
                (unsigned long long)ss.full_matches,
                (unsigned long long)ss.re_feeds);
        free(tokens);
        free(logits);
        free(scratch);
        apus_model_state_free(&st, &e.m);
        engine_close(&e);
        return 0;
    }
    apus_model_forward(&e.m, &st, ids, n_ids, logits, 0, NULL);
    double t_prefill = now_s() - t0;
    fprintf(stderr, "prefill %zu tokens in %.2fs (%.1f tok/s)\n",
            n_ids, t_prefill, n_ids / (t_prefill > 0 ? t_prefill : 1e-9));

    t_dec0 = now_s();
    for (int g = 0; g < max_tokens; g++) {
        int next = apus_sample(logits, (size_t)e.m.vocab, temp, top_k,
                               top_p, &rng, scratch);
        if (e.tok) {
            uint32_t id = (uint32_t)next;
            size_t dl;
            char *s = tok_decode(e.tok, &id, 1, &dl);
            if (s) { fwrite(s, 1, dl, stdout); free(s); }
        } else {
            printf("%d\n", next);
        }
        fflush(stdout);
        n_gen++;
        if (engine_is_eos(&e, next)) break;
        int64_t nid = next;
        apus_model_forward(&e.m, &st, &nid, 1, logits, 0, NULL);
    }
    double t_dec = now_s() - t_dec0;
    if (n_gen)
        fprintf(stderr, "decode %d tokens in %.2fs (%.1f tok/s)\n",
                n_gen, t_dec, n_gen / (t_dec > 0 ? t_dec : 1e-9));

    free(logits);
    free(scratch);
    apus_model_state_free(&st, &e.m);
    engine_close(&e);
    return 0;
}

/* ================= M7a serve mode =================
 *
 * NDJSON protocol on stdin/stdout: one JSON object per line in each
 * direction (the gateway tools/server.py spawns this process and owns
 * all networking; stdio keeps the engine libc-only — no sockets in C).
 * The process stays alive across requests; the model loads once. Every
 * request gets a FRESH ApusModelState (conversation state is the
 * gateway's job; multi-turn context is re-prefilled).
 *
 * Request (client -> engine):
 *   {"id": <any>, "cmd": "encode",
 *    "messages": [...], "tools": [...]|null, "thinking": true|false,
 *    "preserve_thinking": true|false}
 *   {"id": <any>, "cmd": "generate",
 *    "messages": [...] | "text": "raw prompt" | "ids": [1,2,3],
 *    "tools": [...]|null, "thinking": bool, "preserve_thinking": bool,
 *    "max_tokens": int, "temperature": float, "top_p": float,
 *    "top_k": int, "seed": uint, "stop": [str, ...],
 *    "presence_penalty": float}
 * "tools" (OpenAI format) is passed to ling_encode_messages as the
 * template's `tools` variable (the template renders the system block
 * with the <tools> section). "preserve_thinking" maps to the template's
 * preserve_thinking kwarg (thinking kept in ALL assistant history
 * turns; default off = only turns after the last real user query).
 * "text" is tokenized verbatim (no chat template, no BOS — the
 * /v1/completions path).
 *
 * Events (engine -> client):
 *   {"id","type":"encoded","text","ids"}      (encode; ids need tokenizer)
 *   {"id","type":"prompt","prompt_tokens"}    (generate, first)
 *   {"id","type":"token","token_id","text"}   (per generated token; EOS is
 *                                             never emitted; text needs a
 *                                             tokenizer)
 *   {"id","type":"done","finish_reason","prompt_tokens",
 *    "completion_tokens","text"}              finish_reason: "stop" (EOS)
 *                                             | "length" | "stop_string"
 *   {"id","type":"error","message"}           (request failed; loop lives)
 * Defaults: max_tokens 32, temperature 1.0, top_p 0.95, top_k 20,
 * seed 0, presence_penalty 0 (the Qwen generation_config sampling
 * defaults; temperature <= 0 = greedy). Stop strings are checked against
 * the assembled decoded text; a match truncates at the match start (a
 * partial last-token piece is emitted if it precedes the match).
 *
 * presence_penalty (M7; OpenAI/vLLM semantics): when != 0, the penalty
 * is subtracted from the logit of every token already GENERATED at
 * least once in this completion, before temperature/top-k/top-p (the
 * HF/vLLM processors-before-warpers order). 0.0 (default) = OFF — the
 * sampling pipeline is bitwise the pre-M7 one. The model card
 * recommends 1.5 for thinking mode; the default stays 0 pending the
 * user's gate decision.
 */

static char *serve_read_line(void) {
    size_t cap = 1 << 16, n = 0;
    char *buf = malloc(cap);
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (c == '\n') break;
        if (n + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
        buf[n++] = (char)c;
    }
    if (c == EOF && n == 0) { free(buf); return NULL; }
    buf[n] = 0;
    return buf;
}

static JVal *serve_resp(const JVal *id, const char *type) {
    JVal *o = json_new_obj();
    json_obj_set(o, "id", id ? json_clone(id) : json_new_null());
    json_obj_set(o, "type", json_new_str(type));
    return o;
}

static void serve_send(JVal *resp) {
    char *s = json_dumps(resp);
    fputs(s, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    free(s);
    json_free(resp);
}

static void serve_error(const JVal *id, const char *msg) {
    JVal *o = serve_resp(id, "error");
    json_obj_set(o, "message", json_new_str(msg ? msg : "error"));
    serve_send(o);
}

static void serve_token(const JVal *id, int token_id,
                        const char *text, size_t len) {
    JVal *o = serve_resp(id, "token");
    json_obj_set(o, "token_id", json_new_int(token_id));
    if (text) json_obj_set(o, "text", json_new_strn(text, len));
    serve_send(o);
}

/* Length of the longest prefix of s[0..n) that ends on a complete UTF-8
 * character boundary. A trailing partial multi-byte sequence (1-3 bytes)
 * is excluded so it can be completed by the next token's bytes. */
static size_t utf8_complete_prefix(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        size_t need = c < 0x80 ? 1 : c < 0xC0 ? 1 : c < 0xE0 ? 2
                      : c < 0xF0 ? 3 : 4;
        if (i + need > n) break;
        i += need;
    }
    return i;
}

/* Encode request messages (+ optional tools) -> rendered prompt string
 * (NULL on error, already reported). */
static char *serve_render(const JVal *req, const JVal *id) {
    JVal *jmsgs = json_obj_get((JVal *)req, "messages");
    if (!jmsgs || json_type(jmsgs) != J_ARR) {
        serve_error(id, "messages must be a JSON array");
        return NULL;
    }
    LingEncOpts opts = LING_ENC_OPTS_DEFAULT;
    JVal *th = json_obj_get((JVal *)req, "thinking");
    if (th && json_type(th) == J_BOOL && !json_bool(th))
        opts.thinking = 0;
    JVal *pt = json_obj_get((JVal *)req, "preserve_thinking");
    if (pt && json_type(pt) == J_BOOL && json_bool(pt))
        opts.preserve_thinking = 1;
    char *prompt = ling_encode_messages(
        jmsgs, json_obj_get((JVal *)req, "tools"), &opts);
    if (!prompt) serve_error(id, ling_last_error());
    return prompt;
}

static void serve_cmd_encode(Engine *e, const JVal *req, const JVal *id) {
    char *prompt = serve_render(req, id);
    if (!prompt) return;
    JVal *o = serve_resp(id, "encoded");
    json_obj_set(o, "text", json_new_str(prompt));
    if (e->tok) {
        size_t n = 0;
        uint32_t *ids = tok_encode_str(e->tok, prompt, 1, &n);
        JVal *arr = json_new_arr();
        for (size_t i = 0; i < n; i++)
            json_arr_push(arr, json_new_int((long long)ids[i]));
        json_obj_set(o, "ids", arr);
        free(ids);
    }
    free(prompt);
    serve_send(o);
}

static void serve_cmd_generate(Engine *e, const JVal *req, const JVal *id) {
    /* ---- prompt ids: ids | text | messages ---- */
    int64_t *ids = NULL;
    size_t n_ids = 0;
    JVal *jids = json_obj_get((JVal *)req, "ids");
    JVal *jtext = json_obj_get((JVal *)req, "text");
    JVal *jmsgs = json_obj_get((JVal *)req, "messages");
    if (jids && json_type(jids) == J_ARR) {
        n_ids = json_arr_len(jids);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) {
            JVal *v = json_arr_get(jids, i);
            if (!v || json_type(v) != J_NUM) {
                serve_error(id, "ids must be an array of numbers");
                free(ids);
                return;
            }
            ids[i] = (int64_t)json_num(v);
        }
    } else if (jtext && json_type(jtext) == J_STR) {
        if (!e->tok) { serve_error(id, "no tokenizer in model dir"); return; }
        uint32_t *u = tok_encode_str(e->tok, json_str(jtext), 1, &n_ids);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) ids[i] = u[i];
        free(u);
    } else if (jmsgs) {
        if (!e->tok) { serve_error(id, "no tokenizer in model dir"); return; }
        char *prompt = serve_render(req, id);
        if (!prompt) return;
        uint32_t *u = tok_encode_str(e->tok, prompt, 1, &n_ids);
        free(prompt);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) ids[i] = u[i];
        free(u);
    } else {
        serve_error(id, "generate needs messages, text, or ids");
        return;
    }
    if (n_ids == 0) { serve_error(id, "empty prompt"); free(ids); return; }

    /* ---- sampling params (Qwen generation_config defaults) ---- */
    int max_tokens = -1;   /* unset: fill the remaining context (OpenAI
                              semantics — generate until EOS/stop) */
    double temperature = 1.0, top_p = 0.95;
    int top_k = 20;
    uint64_t seed = 0;
    double presence_penalty = 0.0;   /* M7: 0 = OFF (bitwise pre-M7) */
    JVal *v;
    if ((v = json_obj_get((JVal *)req, "max_tokens")) && json_type(v) == J_NUM)
        max_tokens = (int)json_num(v);
    if (max_tokens < -1) max_tokens = 0;
    if ((v = json_obj_get((JVal *)req, "temperature")) && json_type(v) == J_NUM)
        temperature = json_num(v);
    if ((v = json_obj_get((JVal *)req, "top_p")) && json_type(v) == J_NUM)
        top_p = json_num(v);
    if ((v = json_obj_get((JVal *)req, "top_k")) && json_type(v) == J_NUM)
        top_k = (int)json_num(v);
    if ((v = json_obj_get((JVal *)req, "seed")) && json_type(v) == J_NUM)
        seed = (uint64_t)json_num(v);
    if ((v = json_obj_get((JVal *)req, "presence_penalty")) &&
        json_type(v) == J_NUM)
        presence_penalty = json_num(v);
    const char *stops[16];
    int n_stops = 0;
    JVal *jstop = json_obj_get((JVal *)req, "stop");
    if (jstop && json_type(jstop) == J_ARR) {
        size_t ns = json_arr_len(jstop);
        for (size_t i = 0; i < ns && n_stops < 16; i++) {
            JVal *s = json_arr_get(jstop, i);
            if (s && json_type(s) == J_STR && json_strlen(s))
                stops[n_stops++] = json_str(s);
        }
    }

    /* clamp to the state/cache capacity */
    if ((int64_t)n_ids >= e->m.max_seq) {
        serve_error(id, "prompt too long for max_seq");
        free(ids);
        return;
    }
    if (max_tokens < 0 || (int64_t)n_ids + max_tokens > e->m.max_seq)
        max_tokens = (int)(e->m.max_seq - (int64_t)n_ids);

    JVal *o = serve_resp(id, "prompt");
    json_obj_set(o, "prompt_tokens", json_new_int((long long)n_ids));
    serve_send(o);

    ApusModelState st;
    apus_model_state_init(&st, &e->m);
    int V = e->m.vocab;
    float *logits = malloc((size_t)V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)V));
    ApusRng rng;
    apus_rng_seed(&rng, seed);

    apus_model_forward(&e->m, &st, ids, n_ids, logits, 0, NULL);

    /* assembled decoded text (stop-string matching) */
    size_t tcap = 256, tn = 0;
    char *text = malloc(tcap);
    int completion = 0;
    const char *finish = "length";
    /* UTF-8 streaming guard: byte-level BPE can split a multi-byte
     * character across two tokens; hold a partial tail (<=3 bytes) back
     * until the next piece completes it, so token events are always
     * valid UTF-8 (real-model crash: em-dash/accented chars split). */
    char pend[8];
    size_t pn = 0;
    /* presence_penalty (M7): penalized = tokens generated at least once
     * (OpenAI/vLLM definition). penal_ids = compact unique list, seen =
     * per-token dedup marks; both unused (NULL) when the penalty is 0 so
     * the default path stays bitwise the pre-M7 one. The subtraction
     * hits the RAW logits before temperature/top-k/top-p (HF/vLLM
     * processors-before-warpers order); apus_model_forward rewrites the
     * whole logits vector every step, so per-step mutation is safe. */
    unsigned char *seen = NULL;
    int *penal_ids = NULL;
    size_t n_penal = 0;
    if (presence_penalty != 0.0) {
        seen = calloc((size_t)V, 1);
        penal_ids = malloc((size_t)(max_tokens ? max_tokens : 1)
                           * sizeof(int));
    }
    for (int step = 0; step < max_tokens; step++) {
        for (size_t i = 0; i < n_penal; i++)
            logits[penal_ids[i]] -= (float)presence_penalty;
        int t = apus_sample(logits, (size_t)V, (float)temperature, top_k,
                            (float)top_p, &rng, scratch);
        if (engine_is_eos(e, t)) { finish = "stop"; break; }
        completion++;
        if (seen && !seen[t]) {
            seen[t] = 1;
            penal_ids[n_penal++] = t;
        }
        size_t plen = 0;
        char *piece = e->tok
            ? tok_decode(e->tok, (const uint32_t[]){ (uint32_t)t }, 1, &plen)
            : NULL;
        size_t oldn = tn;
        if (piece) {
            if (tn + plen + 1 > tcap) {
                while (tn + plen + 1 > tcap) tcap *= 2;
                text = realloc(text, tcap);
            }
            memcpy(text + tn, piece, plen);
            tn += plen;
            text[tn] = 0;
        }
        /* stop strings: earliest match over the assembled text */
        long mpos = -1;
        if (n_stops && piece) {
            for (int k = 0; k < n_stops; k++) {
                char *hit = strstr(text, stops[k]);
                if (hit && (mpos < 0 || hit - text < mpos))
                    mpos = hit - text;
            }
        }
        if (mpos >= 0) {
            if ((size_t)mpos > oldn)
                serve_token(id, t, text + oldn, (size_t)mpos - oldn);
            tn = (size_t)mpos;
            finish = "stop_string";
            free(piece);
            break;
        }
        /* emit the longest complete-UTF-8 prefix of pend+piece */
        if (piece) {
            char *buf = malloc(pn + plen);
            memcpy(buf, pend, pn);
            memcpy(buf + pn, piece, plen);
            size_t comp = utf8_complete_prefix(buf, pn + plen);
            size_t hold = pn + plen - comp;
            if (comp) serve_token(id, t, buf, comp);
            pn = hold <= 3 ? hold : 0;  /* >3 means invalid input: drop */
            if (pn) memcpy(pend, buf + comp, pn);
            free(buf);
        } else if (pn) {
            /* tokenizer-less model: nothing decodable pending */
            pn = 0;
        }
        free(piece);
        int64_t next = t;
        apus_model_forward(&e->m, &st, &next, 1, logits, 0, NULL);
    }
    if (pn) {
        /* stream ended mid-sequence (shouldn't happen for real text):
         * close it out with the replacement character */
        serve_token(id, -1, "\xEF\xBF\xBD", 3);
        if (tn + 4 > tcap) { tcap += 4; text = realloc(text, tcap); }
        memcpy(text + tn, "\xEF\xBF\xBD", 3);
        tn += 3;
        text[tn] = 0;
    }

    JVal *d = serve_resp(id, "done");
    json_obj_set(d, "finish_reason", json_new_str(finish));
    json_obj_set(d, "prompt_tokens", json_new_int((long long)n_ids));
    json_obj_set(d, "completion_tokens", json_new_int(completion));
    json_obj_set(d, "text", json_new_strn(text ? text : "", tn));
    serve_send(d);

    free(text);
    free(seen);
    free(penal_ids);
    free(logits);
    free(scratch);
    free(ids);
    apus_model_state_free(&st, &e->m);
}

static int serve_main(int argc, char **argv) {
    const char *model_dir = NULL;
    int tiered = apus_env_int("APUS_TIERED", 0);
    int max_seq = apus_env_int("APUS_MAX_SEQ", 32768);
    int metal = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model_dir = argv[++i];
        else if (!strcmp(argv[i], "--tiered")) tiered = 1;
        else if (!strcmp(argv[i], "--metal")) metal = 1;
        else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc)
            max_seq = atoi(argv[++i]);
        else { usage(stderr); return 2; }
    }
    if (!model_dir) { usage(stderr); return 2; }
    if (apus_env_int("APUS_METAL", 0)) metal = 1;
    if (metal) apus_cli_metal_init();

    Engine e;
    char err[256];
    double t0 = now_s();
    if (engine_init(&e, model_dir, max_seq, tiered, 0, err, sizeof err)) {
        fprintf(stderr, "apus-qwen serve: %s\n", err);
        return 1;
    }
    fprintf(stderr,
            "apus-qwen serve: loaded %s (%d layers, hidden %d, vocab %d%s,"
            " max_seq %d) in %.2fs\n",
            model_dir, e.m.n_layers, e.m.hidden, e.m.vocab,
            tiered ? ", tiered" : "", e.m.max_seq, now_s() - t0);

    char jerr[128];
    for (;;) {
        char *line = serve_read_line();
        if (!line) break;                    /* EOF: clean shutdown */
        JVal *req = json_parse(line, strlen(line), jerr, sizeof jerr);
        free(line);
        if (!req) {
            serve_error(NULL, jerr);
            continue;
        }
        JVal *id = json_obj_get(req, "id");
        JVal *cmd = json_obj_get(req, "cmd");
        const char *c = cmd && json_type(cmd) == J_STR ? json_str(cmd) : "";
        if (!strcmp(c, "encode")) serve_cmd_encode(&e, req, id);
        else if (!strcmp(c, "generate")) serve_cmd_generate(&e, req, id);
        else serve_error(id, "unknown cmd");
        json_free(req);
    }
    engine_close(&e);
    return 0;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    /* M13 (Apus M15): MSVCRT defaults stdin/stdout to text mode
     * (\n -> \r\n), which would corrupt the NDJSON serve protocol;
     * force binary. */
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    if (!strcmp(argv[1], "run")) return run_main(argc, argv);
    if (!strcmp(argv[1], "serve")) return serve_main(argc, argv);
    usage(stderr);
    return 2;
}
