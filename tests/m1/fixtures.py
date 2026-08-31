"""Synthetic Qwen3.6-35B-A3B checkpoint fixtures for M1 tests.

Mimics the real checkpoint's naming scheme and dtypes at tiny scale
(verified against reference/model.safetensors.index.json):

  model.language_model.embed_tokens.weight / .norm.weight, lm_head.weight
  model.language_model.layers.{L}.self_attn.{q,k,v,o}_proj.weight +
    .{q,k}_norm.weight                              (full-attention layers)
  model.language_model.layers.{L}.linear_attn.{in_proj_qkv,in_proj_z,
    in_proj_b,in_proj_a,out_proj}.weight + .conv1d.weight + .{A_log,dt_bias}
    + .norm.weight                                  (linear-attention layers)
  model.language_model.layers.{L}.{input,post_attention}_layernorm.weight
  model.language_model.layers.{L}.mlp.gate.weight
  model.language_model.layers.{L}.mlp.experts.gate_up_proj  [E, 2M, H]  FUSED
  model.language_model.layers.{L}.mlp.experts.down_proj     [E, H, M]   FUSED
  model.language_model.layers.{L}.mlp.shared_expert.{gate,up,down}_proj.weight
  model.language_model.layers.{L}.mlp.shared_expert_gate.weight
  mtp.{fc,pre_fc_norm_embedding,pre_fc_norm_hidden,norm}.weight +
    mtp.layers.0.*                                  (MTP block, full attn+MoE)
  model.visual.*                                    (stripped at conversion)

Dims are chosen so the BF16 self-description invariants hold exactly like
the real model (per-expert slab == gate_up slice + down slice, 2 bytes per
element):

  hidden H = 64, expert intermediate M = 16, experts E = 8
  gate_up_proj [E, 2M, H] BF16 -> per-expert slice 2*M*H*2 = 4096 B
  down_proj    [E, H, M] BF16  -> per-expert slice H*M*2   = 2048 B
  -> per expert slab: 6144 B

The real model has MoE on ALL 40 layers (no dense layers), full attention
every 4th layer (3, 7, ..., 39), and a top-level MTP block. The fixture
reproduces this with N_LAYERS=4 (linear 0..2, full 3) plus mtp.layers.0.

Critically, the fixture also reproduces the real checkpoint's CROSS-SHARD
expert layers (17 of 41 real layers have their two fused tensors in
different input shards): layer 1 has down_proj in an earlier shard than
gate_up_proj, layer 2 the reverse, and the MTP layer is split too. This
exercises the converter's deferral machinery in both one-shot and driver
modes.

Payloads are random bytes (byte-identity tests don't need meaningful
values; test_3_bf16_values.py writes its own known-value tensors). The
fixture writes 5 input shards with a weight_map index, exactly like the
real layout.
"""

import json
import os

import numpy as np

import stutil

H = 64        # hidden size
M = 16        # moe intermediate size
N_EXPERTS = 8
N_LAYERS = 4  # num_hidden_layers: linear 0..2, full-attention 3
FULL_ATTN_LAYERS = {3}
VOCAB = 128   # tiny stand-in for 248,320
# GDN dims (linear attention): key hk*dk = 4*8, value hv*dv = 8*8
HK, DK = 4, 8
HV, DV = 8, 8
KEY_DIM = HK * DK          # 32
VALUE_DIM = HV * DV        # 64
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 128
CONV_K = 4
# full attention: q_proj packs per-head [q|gate] -> rows = 2*nh*d
NH, NKV, HEAD_D = 4, 2, 16
N_SHARDS = 5

PER_EXPERT_BYTES = 2 * M * H * 2 + H * M * 2
assert PER_EXPERT_BYTES == 6144

# Expert-carrying blocks: main layers 0..3 + the MTP block (layer index
# N_LAYERS in slab records).
N_EXPERT_LAYERS = N_LAYERS + 1


def _rand(rng, dtype, shape):
    return rng.randint(0, 256, size=stutil.tensor_nbytes(dtype, shape),
                       dtype=np.uint8).tobytes()


def _linear_attn_tensors(pre):
    """Gated DeltaNet block (layers 0..2 in the fixture; real: every layer
    except the periodic full-attention ones)."""
    return [
        (f"{pre}.linear_attn.in_proj_qkv.weight", "BF16", [CONV_DIM, H]),
        (f"{pre}.linear_attn.in_proj_z.weight", "BF16", [VALUE_DIM, H]),
        (f"{pre}.linear_attn.in_proj_b.weight", "BF16", [HV, H]),
        (f"{pre}.linear_attn.in_proj_a.weight", "BF16", [HV, H]),
        (f"{pre}.linear_attn.conv1d.weight", "BF16", [CONV_DIM, 1, CONV_K]),
        (f"{pre}.linear_attn.A_log", "F32", [HV]),
        (f"{pre}.linear_attn.dt_bias", "F32", [HV]),
        (f"{pre}.linear_attn.norm.weight", "BF16", [DV]),
        (f"{pre}.linear_attn.out_proj.weight", "BF16", [H, VALUE_DIM]),
    ]


def _full_attn_tensors(pre):
    """Full-attention (GQA + output gate) block: periodic layers + MTP."""
    return [
        (f"{pre}.self_attn.q_proj.weight", "BF16", [2 * NH * HEAD_D, H]),
        (f"{pre}.self_attn.k_proj.weight", "BF16", [NKV * HEAD_D, H]),
        (f"{pre}.self_attn.v_proj.weight", "BF16", [NKV * HEAD_D, H]),
        (f"{pre}.self_attn.o_proj.weight", "BF16", [H, NH * HEAD_D]),
        (f"{pre}.self_attn.q_norm.weight", "BF16", [HEAD_D]),
        (f"{pre}.self_attn.k_norm.weight", "BF16", [HEAD_D]),
    ]


def _moe_dense_tensors(pre):
    """Router + shared expert (the routed experts are the fused tensors,
    handled separately)."""
    t = [(f"{pre}.mlp.gate.weight", "BF16", [N_EXPERTS, H])]
    for w in ("gate_proj", "up_proj", "down_proj"):
        rows, cols = (M, H) if w != "down_proj" else (H, M)
        t.append((f"{pre}.mlp.shared_expert.{w}.weight", "BF16",
                  [rows, cols]))
    t.append((f"{pre}.mlp.shared_expert_gate.weight", "BF16", [1, H]))
    return t


def _fused_expert_tensors(pre):
    return [
        (f"{pre}.mlp.experts.gate_up_proj", "BF16", [N_EXPERTS, 2 * M, H]),
        (f"{pre}.mlp.experts.down_proj", "BF16", [N_EXPERTS, H, M]),
    ]


def _layer_dense_specs(layer):
    """All NON-fused tensors of one main-model layer."""
    pre = f"model.language_model.layers.{layer}"
    t = [(f"{pre}.input_layernorm.weight", "BF16", [H]),
         (f"{pre}.post_attention_layernorm.weight", "BF16", [H])]
    if layer in FULL_ATTN_LAYERS:
        t += _full_attn_tensors(pre)
    else:
        t += _linear_attn_tensors(pre)
    t += _moe_dense_tensors(pre)
    return t


def _mtp_dense_specs():
    """MTP block dense tensors: top-level extras + one full-attention
    layer's non-fused tensors."""
    t = [
        ("mtp.fc.weight", "BF16", [H, 2 * H]),
        ("mtp.pre_fc_norm_embedding.weight", "BF16", [H]),
        ("mtp.pre_fc_norm_hidden.weight", "BF16", [H]),
        ("mtp.norm.weight", "BF16", [H]),
    ]
    pre = "mtp.layers.0"
    t += [(f"{pre}.input_layernorm.weight", "BF16", [H]),
          (f"{pre}.post_attention_layernorm.weight", "BF16", [H])]
    t += _full_attn_tensors(pre)
    t += _moe_dense_tensors(pre)
    return t


def _global_specs():
    return [
        ("model.language_model.embed_tokens.weight", "BF16", [VOCAB, H]),
        ("model.language_model.norm.weight", "BF16", [H]),
        ("lm_head.weight", "BF16", [VOCAB, H]),
    ]


def _visual_specs():
    """Vision tower stand-ins: present in the source, stripped at
    conversion (333 tensors in the real checkpoint)."""
    return [
        ("model.visual.patch_embed.proj.weight", "BF16", [48, 48]),
        ("model.visual.blocks.0.attn.qkv.weight", "BF16", [48, 48]),
        ("model.visual.blocks.0.attn.qkv.bias", "BF16", [48]),
        ("model.visual.merger.norm.weight", "BF16", [48]),
    ]


def make_fixture_tree(root, seed=1234):
    """Write a fake checkpoint under root/. Returns list of shard file names.

    Layout mimics the real one — globals + vision tower in shard 1, layers
    spread over the remaining shards, MTP last — and deliberately splits
    three expert layers across shards (both directions), exactly like the
    real checkpoint does for 17 of its 41 MoE layers:

      layer 1:        down_proj in shard 2, gate_up_proj in shard 3
      layer 2:        gate_up_proj in shard 3, down_proj in shard 4
      mtp.layers.0:   down_proj in shard 4, gate_up_proj in shard 5
    """
    rng = np.random.RandomState(seed)
    groups = [[] for _ in range(N_SHARDS)]

    groups[0] += _global_specs() + _visual_specs()

    # layer 0: complete in shard 2
    pre0 = "model.language_model.layers.0"
    groups[1] += _layer_dense_specs(0) + _fused_expert_tensors(pre0)
    # layer 1: dense + down_proj in shard 2, gate_up_proj in shard 3
    pre1 = "model.language_model.layers.1"
    groups[1] += _layer_dense_specs(1)
    groups[1] += [(f"{pre1}.mlp.experts.down_proj", "BF16",
                   [N_EXPERTS, H, M])]
    groups[2] += [(f"{pre1}.mlp.experts.gate_up_proj", "BF16",
                   [N_EXPERTS, 2 * M, H])]
    # layer 2: dense + gate_up_proj in shard 3, down_proj in shard 4
    pre2 = "model.language_model.layers.2"
    groups[2] += _layer_dense_specs(2)
    groups[2] += [(f"{pre2}.mlp.experts.gate_up_proj", "BF16",
                   [N_EXPERTS, 2 * M, H])]
    groups[3] += [(f"{pre2}.mlp.experts.down_proj", "BF16",
                   [N_EXPERTS, H, M])]
    # layer 3 (full attention): complete in shard 4
    pre3 = "model.language_model.layers.3"
    groups[3] += _layer_dense_specs(3) + _fused_expert_tensors(pre3)
    # MTP: dense + down_proj in shard 4, top-level extras + gate_up_proj
    # in shard 5
    groups[3] += [(f"mtp.layers.0.mlp.experts.down_proj", "BF16",
                   [N_EXPERTS, H, M])]
    groups[4] += _mtp_dense_specs()
    groups[4] += [("mtp.layers.0.mlp.experts.gate_up_proj", "BF16",
                   [N_EXPERTS, 2 * M, H])]

    weight_map = {}
    names = []
    for i, specs in enumerate(groups):
        fname = f"model-{i + 1:05d}-of-{N_SHARDS:05d}.safetensors"
        tensors = [(name, dtype, shape, _rand(rng, dtype, shape))
                   for name, dtype, shape in specs]
        stutil.write_shard(os.path.join(root, fname), tensors)
        for name, _, _, _ in tensors:
            weight_map[name] = fname
        names.append(fname)

    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        total = sum(
            stutil.tensor_nbytes(d, s)
            for specs in groups for _, d, s in specs
        )
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": H,
                "moe_intermediate_size": M,
                "shared_expert_intermediate_size": M,
                "num_experts": N_EXPERTS,
                "num_experts_per_tok": 2,
                "num_hidden_layers": N_LAYERS,
                "num_attention_heads": NH,
                "num_key_value_heads": NKV,
                "head_dim": HEAD_D,
                "linear_num_key_heads": HK,
                "linear_key_head_dim": DK,
                "linear_num_value_heads": HV,
                "linear_value_head_dim": DV,
                "linear_conv_kernel_dim": CONV_K,
                "full_attention_interval": 4,
                "layer_types": [
                    "full_attention" if layer in FULL_ATTN_LAYERS
                    else "linear_attention"
                    for layer in range(N_LAYERS)
                ],
                "vocab_size": VOCAB,
                "mtp_num_hidden_layers": 1,
            },
            "vision_config": {"model_type": "qwen3_5_moe",
                              "hidden_size": 48},
        }, f)
    return names
