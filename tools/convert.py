#!/usr/bin/env python3
"""apus-qwen M1 — HF safetensors -> apus container converter.

Converts a directory of HuggingFace safetensors shards
(Qwen/Qwen3.6-35B-A3B, model_type qwen3_5_moe) into the apus weight
container (same format family as the Ling/DeepSeek apus containers; see
docs/ARCHITECTURE.md §6 of the base project):

  * Byte-identical copy of every tensor payload. Payloads are treated as
    raw bytes; this tool never requantizes, transcodes, or even interprets
    tensor data. (Qwen3.6-35B-A3B ships BF16 weights — there is no
    quantization to preserve, and the (1+w) RMSNorm convention is applied
    at RUNTIME by the engine, never here; the byte-identity contract is
    unchanged.)
  * TEXT-ONLY scope: the vision tower (model.visual.*, 333 tensors in the
    real checkpoint) is stripped — skipped, counted, reported in the
    manifest. Same precedent as the GLM adapter's vision strip.
  * Coalesced per-expert layout: the checkpoint stores routed experts as
    two FUSED tensors per layer — mlp.experts.gate_up_proj [E, 2M, H] and
    mlp.experts.down_proj [E, H, M]. The converter SLICES them per expert
    and coalesces: within each output shard, expert e's slab is
    gate_up_proj[e] followed by down_proj[e], contiguous and adjacent, so
    the engine fetches a whole expert with one pread. Content is
    byte-identical to the checkpoint bytes, order rearranged — the same
    rearrangement class as the base's 3-tensor coalescing. Slice tensors
    are recorded in the output under synthesized per-expert names
    {<layer prefix>.mlp.experts.<E>.gate_up_proj.weight / .down_proj.weight}
    so every output shard stays a structurally valid safetensors file.
  * Output shards are ordinary safetensors files, written MANUALLY (8-byte
    little-endian header length + JSON header + raw data). safetensors.numpy
    is not used for writing: we need exact control over tensor order and
    byte identity.
  * Shard groups: main weights in `apus-qwen-NNNNN.safetensors`, the MTP
    block (top-level mtp.* tensors) in `apus-qwen-mtp-NNNNN.safetensors`
    (so the engine can lazy-load the speculative set).
  * Manifest `apus.index.json`: format version, config hash, full tensor map
    (name -> shard, absolute file offset, nbytes, dtype, shape) and
    per-expert slab records (layer, expert -> shard, offset, total bytes).

Cross-shard expert layers
-------------------------
Unlike the Ling base (where an expert's 3 tensors always sat in one input
shard), the real Qwen checkpoint places a layer's two fused expert tensors
in DIFFERENT input shards for 17 of the 41 MoE layers (verified against
reference/model.safetensors.index.json). A layer's slabs are therefore
assembled only once BOTH fused tensors' home shards have been reached in
sorted input-shard order; until then the layer is deferred (deterministically
— the rule depends only on the sorted shard order, so a one-shot conversion
and the shard-at-a-time download driver produce byte-identical output).
The download driver keeps a source shard on disk until every tensor in it
has been consumed into the output, so the deferred partner is always
readable when assembly fires.

Resumability / crash safety
---------------------------
Conversion is driven input-shard by input-shard. A state file
(`apus.convert.state.json` in the output dir) is rewritten atomically after
every single tensor append, after every shard seal, and after every finished
input shard. Re-running after an interruption:

  1. Sealed output shards are verify-before-trust checked (size + header
     hash recorded in the state).
  2. The open output shard is validated against the state: bytes committed
     in the state must be present on disk; any tail beyond the last commit
     (a torn tensor write) is truncated and rewritten.
  3. Completed input shards are skipped entirely; a partially assembled
     expert layer resumes at the first unwritten expert slice.

The append sequence is a deterministic function of the input set, so an
interrupted+resumed run produces byte-identical output to an uninterrupted
run (covered by tests/m1/test_4_resume.py).

Output shard format detail: shards are created with a fixed-capacity header
region (HEADER_RESERVE bytes) holding a placeholder JSON document. This lets
us append tensor data streaming-style and "seal" the shard later by simply
rewriting the header region in place — no data copies, and the file is a
structurally valid safetensors file at every point in time. Sealing pads the
JSON with trailing spaces, which the safetensors format permits.

FORMAT_VERSION = 2: version 1 was the Ling adapter's container (per-expert
slabs of 3 whole source tensors, output tensor names == source names).
Version 2 changes the slab semantics — 2 slices per expert cut from the
fused gate_up_proj/down_proj tensors, recorded under converter-synthesized
names — so a consumer written against v1 assumptions (3 members, source-name
identity) must not silently read a v2 container. The manifest's top-level
keys are otherwise unchanged (format_version, config_hash, offset_base,
header_reserve, shard_groups, ntensors, tensor_map, expert_slabs), plus the
additive stripped_tensors audit count.

Usage:
    python tools/convert.py convert  SRC_DIR DST_DIR [--shard NAME ...]
                                     [--target-bytes N]
    python tools/convert.py verify   SRC_DIR DST_DIR [--shard NAME ...]
    python tools/convert.py finalize SRC_DIR DST_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile

FORMAT_VERSION = 2

# Fixed header capacity of every output shard. ~100 bytes of JSON per tensor
# entry => room for well over 100k tensors per shard; validated on seal.
HEADER_RESERVE = 16 * 1024 * 1024

# Target output shard size (~5 GB). A shard is sealed before a
# tensor/expert-slab append that would exceed this, so the actual size stays
# within target + one expert slab (6 MiB for the real model).
DEFAULT_TARGET_BYTES = 5 * 1024**3

COPY_CHUNK = 8 * 1024 * 1024

STATE_FILE = "apus.convert.state.json"
MANIFEST_FILE = "apus.index.json"

MAIN_PREFIX = "apus-qwen"
MTP_PREFIX = "apus-qwen-mtp"

# Text-only scope: the vision tower is stripped (skipped + counted).
VISUAL_PREFIX = "model.visual."

# Fallback when src config.json is absent: Qwen3.6-35B-A3B has 40 main
# layers; the MTP block (mtp.layers.{K}.*) is numbered num_hidden_layers+K
# in slab records, matching the base engine's layer-index convention.
DEFAULT_NUM_HIDDEN_LAYERS = 40

# Fused routed-expert tensors in the SOURCE checkpoint:
#   model.language_model.layers.{L}.mlp.experts.gate_up_proj  [E, 2M, H]
#   model.language_model.layers.{L}.mlp.experts.down_proj     [E, H, M]
#   mtp.layers.{K}.mlp.experts.{gate_up_proj,down_proj}       (MTP block)
FUSED_EXPERT_RE = re.compile(
    r"^(model\.language_model\.layers\.(\d+)|mtp\.layers\.(\d+))"
    r"\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
# Synthesized per-expert slice names in the OUTPUT container:
#   <layer prefix>.mlp.experts.{E}.{gate_up_proj.weight|down_proj.weight}
OUT_EXPERT_RE = re.compile(
    r"^(model\.language_model\.layers\.(\d+)|mtp\.layers\.(\d+))"
    r"\.mlp\.experts\.(\d+)\.(gate_up_proj|down_proj)\.weight$"
)
# Fixed intra-slab order: gate_up_proj slice, then down_proj slice.
SLAB_MEMBERS = ("gate_up_proj.weight", "down_proj.weight")
# Source-side member order (no .weight suffix in the checkpoint).
_SRC_MEMBERS = ("gate_up_proj", "down_proj")


def _layer_key(group, layer_idx):
    return (group, layer_idx)


def _layer_prefix(group, layer_idx):
    if group == "mtp":
        return f"mtp.layers.{layer_idx}"
    return f"model.language_model.layers.{layer_idx}"


def _fused_src_name(group, layer_idx, member):
    return f"{_layer_prefix(group, layer_idx)}.mlp.experts.{member}"


def _slice_name(group, layer_idx, expert, member):
    return (f"{_layer_prefix(group, layer_idx)}.mlp.experts.{expert}"
            f".{member}.weight")


def _match_group_layer(m):
    """(group, layer_idx) from a FUSED_EXPERT_RE/OUT_EXPERT_RE match;
    main layers -> 'main', mtp.layers.K -> 'mtp'."""
    if m.group(3) is not None:
        return "mtp", int(m.group(3))
    return "main", int(m.group(2))


# --------------------------------------------------------------------------
# safetensors header reading (source shards)
# --------------------------------------------------------------------------

class SrcTensor:
    """One tensor in a source shard: location + self-description, no data."""
    __slots__ = ("name", "dtype", "shape", "src_path", "file_offset", "nbytes")

    def __init__(self, name, dtype, shape, src_path, file_offset, nbytes):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.src_path = src_path
        self.file_offset = file_offset  # absolute offset in src_path
        self.nbytes = nbytes


def read_st_header(path):
    """Parse a safetensors header. Returns dict name -> SrcTensor.

    Only the header is read; tensor payloads are never touched here.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (too small)")
        (hlen,) = struct.unpack("<Q", raw)
        hjson = f.read(hlen)
        if len(hjson) != hlen:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(hjson)
    data_start = 8 + hlen
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        tensors[name] = SrcTensor(
            name, meta["dtype"], list(meta["shape"]), path,
            data_start + begin, end - begin,
        )
    return tensors


def load_weight_map(src_dir):
    """tensor name -> home input shard for the whole source tree.

    Uses model.safetensors.index.json when present (the real checkpoint
    layout), else scans every source shard header (all on disk in that
    mode).
    """
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"])
    wm = {}
    for n in sorted(os.listdir(src_dir)):
        if n.endswith(".safetensors") and not n.startswith("apus"):
            for name in read_st_header(os.path.join(src_dir, n)):
                wm[name] = n
    return wm


def list_input_shards(src_dir):
    """Input shard file names in deterministic processing order.

    Uses model.safetensors.index.json when present (the real checkpoint
    layout), else every *.safetensors file in the directory.
    """
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted(set(weight_map.values()))
    return sorted(
        n for n in os.listdir(src_dir)
        if n.endswith(".safetensors") and not n.startswith("apus")
    )


def config_hash(src_dir):
    """Stable hash identifying the model configuration being converted."""
    for name in ("config.json", "model.safetensors.index.json"):
        path = os.path.join(src_dir, name)
        if os.path.exists(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(COPY_CHUNK), b""):
                    h.update(chunk)
            return f"sha256:{h.hexdigest()}"
    return "sha256:none"


def num_hidden_layers(src_dir):
    """num_hidden_layers from src config.json — nested under text_config
    in the real qwen3_5_moe config, top-level in bare text configs;
    falls back to DEFAULT_NUM_HIDDEN_LAYERS when absent."""
    path = os.path.join(src_dir, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        text = cfg.get("text_config", cfg)
        return int(text["num_hidden_layers"])
    except Exception:
        return DEFAULT_NUM_HIDDEN_LAYERS


# --------------------------------------------------------------------------
# Tensor classification
# --------------------------------------------------------------------------

def classify_group(name):
    """Route a tensor name to its output shard group (or 'strip').

    The MTP block is the top-level mtp.* set (mtp.fc, mtp.norm,
    mtp.pre_fc_norm_*, mtp.layers.{K}.*); the vision tower is stripped.
    """
    if name.startswith(VISUAL_PREFIX):
        return "strip"
    if name.startswith("mtp."):
        return "mtp"
    return "main"


# --------------------------------------------------------------------------
# Output shard writer
# --------------------------------------------------------------------------

def _placeholder_header():
    doc = json.dumps({"__metadata__": {"apus_state": "open"}}).encode()
    return doc + b" " * (HEADER_RESERVE - len(doc))


def _sealed_header_bytes(entries):
    """Final JSON header for a shard, padded with spaces to HEADER_RESERVE."""
    header = {}
    off = 0
    for e in entries:
        header[e["name"]] = {
            "dtype": e["dtype"],
            "shape": e["shape"],
            "data_offsets": [off, off + e["nbytes"]],
        }
        off += e["nbytes"]
    doc = json.dumps(header, separators=(",", ":")).encode()
    if len(doc) > HEADER_RESERVE:
        raise ValueError(
            f"shard header needs {len(doc)} bytes > HEADER_RESERVE "
            f"({HEADER_RESERVE}); raise HEADER_RESERVE in tools/convert.py"
        )
    return doc + b" " * (HEADER_RESERVE - len(doc))


def _header_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(8 + HEADER_RESERVE))
    return h.hexdigest()


class GroupStream:
    """Manages the output shard sequence of one group (main/mtp).

    The "open" shard is created lazily on the first append, appended to
    tensor by tensor, and sealed (header rewritten in place) once the next
    append would exceed the target size. All mutations are reflected in the
    converter state and flushed to disk before the next mutation, so a kill
    at any point leaves a resumable on-disk state.
    """

    PREFIXES = {"main": MAIN_PREFIX, "mtp": MTP_PREFIX}

    def __init__(self, group, out_dir, state, save_state):
        self.group = group
        self.prefix = self.PREFIXES[group]
        self.out_dir = out_dir
        self.state = state          # state["groups"][group], mutated in place
        self.save_state = save_state

    # -- state helpers ----------------------------------------------------

    @property
    def gstate(self):
        return self.state["groups"][self.group]

    def _shard_name(self, idx):
        return f"{self.prefix}-{idx:05d}.safetensors"

    def _open_path(self):
        open_ = self.gstate["open"]
        if open_ is None:
            return None
        return os.path.join(self.out_dir, open_["file"])

    # -- write path --------------------------------------------------------

    def _create_open(self):
        idx = self.gstate["next_shard_idx"]
        name = self._shard_name(idx)
        path = os.path.join(self.out_dir, name)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", HEADER_RESERVE))
            f.write(_placeholder_header())
        self.gstate["open"] = {"file": name, "data_bytes": 0, "entries": []}
        self.save_state()

    def append_tensor(self, tensor, progress_cb=None):
        """Append one tensor's raw bytes to the open shard."""
        open_ = self.gstate["open"]
        if open_ is None:
            self._create_open()
            open_ = self.gstate["open"]
        path = os.path.join(self.out_dir, open_["file"])
        data_off = open_["data_bytes"]
        with open(path, "r+b") as out, open(tensor.src_path, "rb") as src:
            out.seek(8 + HEADER_RESERVE + data_off)
            src.seek(tensor.file_offset)
            remaining = tensor.nbytes
            while remaining:
                chunk = src.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    raise IOError(f"{tensor.src_path}: short read on {tensor.name}")
                out.write(chunk)
                remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        open_["entries"].append({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "offset": data_off,   # relative to data region start
            "nbytes": tensor.nbytes,
        })
        open_["data_bytes"] += tensor.nbytes
        self.save_state()
        if progress_cb:
            progress_cb("tensor", group=self.group, name=tensor.name,
                        shard=open_["file"])

    def ensure_capacity(self, nbytes, target_bytes, progress_cb=None):
        """Seal the open shard if appending nbytes would exceed the target.

        Called before a whole expert slab (or a single dense tensor), so a
        slab is never split across shards.
        """
        open_ = self.gstate["open"]
        if open_ and open_["data_bytes"] > 0 and \
                open_["data_bytes"] + nbytes > target_bytes:
            self.seal(progress_cb)

    def seal(self, progress_cb=None):
        """Finalize the open shard: rewrite its header region in place."""
        open_ = self.gstate["open"]
        if open_ is None:
            return
        path = os.path.join(self.out_dir, open_["file"])
        header = _sealed_header_bytes(open_["entries"])
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(struct.pack("<Q", HEADER_RESERVE))
            f.write(header)
            f.flush()
            os.fsync(f.fileno())
        size = os.path.getsize(path)
        self.gstate["sealed"][open_["file"]] = {
            "size": size,
            "header_sha256": _header_sha256(path),
            "ntensors": len(open_["entries"]),
        }
        sealed_name = open_["file"]
        self.gstate["open"] = None
        self.gstate["next_shard_idx"] += 1
        self.save_state()
        if progress_cb:
            progress_cb("seal", group=self.group, shard=sealed_name,
                        ntensors=self.gstate["sealed"][sealed_name]["ntensors"])

    # -- resume validation --------------------------------------------------

    def validate(self):
        """Verify-before-trust check of this group's on-disk output.

        Sealed shards must match recorded size + header hash. The open shard
        must contain at least the committed bytes; a torn tail (crash mid
        tensor write, after the previous state flush) is truncated. A shard
        whose seal was written but never recorded in the state is adopted as
        sealed. Anything else is corruption: fail loudly.
        """
        for name, rec in self.gstate["sealed"].items():
            path = os.path.join(self.out_dir, name)
            if not os.path.exists(path):
                raise ValueError(f"missing sealed output shard {name}")
            if os.path.getsize(path) != rec["size"]:
                raise ValueError(f"sealed shard {name}: size mismatch "
                                 f"(state {rec['size']}, disk "
                                 f"{os.path.getsize(path)})")
            if _header_sha256(path) != rec["header_sha256"]:
                raise ValueError(f"sealed shard {name}: header hash mismatch")

        open_ = self.gstate["open"]
        if open_ is None:
            return
        path = os.path.join(self.out_dir, open_["file"])
        if not os.path.exists(path):
            raise ValueError(f"missing open output shard {open_['file']}")
        data_start = 8 + HEADER_RESERVE
        expected = data_start + open_["data_bytes"]
        size = os.path.getsize(path)
        if size < expected:
            raise ValueError(
                f"open shard {open_['file']}: {size} bytes on disk but state "
                f"records {expected}; output is corrupt, delete the apus-* "
                f"shards and state file and reconvert"
            )
        if size > expected:
            # Torn write of the tensor that was being appended when we were
            # killed: the state flush for it never happened. Drop the tail.
            with open(path, "r+b") as f:
                f.truncate(expected)

        # Distinguish "placeholder header" from "sealed but unrecorded".
        with open(path, "rb") as f:
            f.read(8)
            raw = f.read(HEADER_RESERVE)
        header = json.loads(raw)
        if "__metadata__" not in header:
            expected_header = _sealed_header_bytes(open_["entries"])
            if raw != expected_header:
                raise ValueError(
                    f"open shard {open_['file']}: header neither placeholder "
                    f"nor the expected sealed header; refusing to trust it"
                )
            # Seal completed but the state flush did not: adopt it.
            self.gstate["sealed"][open_["file"]] = {
                "size": expected,
                "header_sha256": hashlib.sha256(
                    struct.pack("<Q", HEADER_RESERVE) + raw).hexdigest(),
                "ntensors": len(open_["entries"]),
            }
            self.gstate["open"] = None
            self.gstate["next_shard_idx"] += 1
            self.save_state()


# --------------------------------------------------------------------------
# Converter state
# --------------------------------------------------------------------------

def _empty_state(cfg_hash, target_bytes):
    return {
        "format_version": FORMAT_VERSION,
        "config_hash": cfg_hash,
        "target_shard_bytes": target_bytes,
        "header_reserve": HEADER_RESERVE,
        "inputs_done": [],
        "expert_layers": {},
        "groups": {
            g: {"next_shard_idx": 1, "open": None, "sealed": {}}
            for g in ("main", "mtp")
        },
        "complete": False,
    }


class Converter:
    def __init__(self, src_dir, dst_dir, target_bytes=DEFAULT_TARGET_BYTES):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.target_bytes = target_bytes
        self.state_path = os.path.join(dst_dir, STATE_FILE)
        self.cfg_hash = config_hash(src_dir)
        self.num_hidden_layers = num_hidden_layers(src_dir)
        self.weight_map = load_weight_map(src_dir)
        self.input_shards = sorted(set(self.weight_map.values()))
        self._shard_pos = {n: i for i, n in enumerate(self.input_shards)}
        self._hdr_cache = {}
        os.makedirs(dst_dir, exist_ok=True)
        self.state = self._load_or_init()
        self.streams = {
            g: GroupStream(g, dst_dir, self.state, self.save_state)
            for g in ("main", "mtp")
        }
        for stream in self.streams.values():
            stream.validate()
        # Names already written to the output (sealed shards + committed
        # open-shard entries). Conversion of an interrupted input shard
        # resumes exactly after these, never redoing them.
        self.written = set()
        for stream in self.streams.values():
            for shard in stream.gstate["sealed"]:
                self.written.update(read_st_header(
                    os.path.join(dst_dir, shard)))
            open_ = stream.gstate["open"]
            if open_ is not None:
                self.written.update(e["name"] for e in open_["entries"])

    # -- source tree helpers -------------------------------------------------

    def _src_header(self, shard_name):
        if shard_name not in self._hdr_cache:
            self._hdr_cache[shard_name] = read_st_header(
                os.path.join(self.src_dir, shard_name))
        return self._hdr_cache[shard_name]

    def _record_expert_layer(self, group, layer_idx, n_experts):
        """Remember a layer's expert count in the state (first sight wins;
        the assembly path validates both fused tensors agree). Recorded so
        that finalize can audit completeness without the source shards,
        which the download driver has already deleted by then."""
        key = f"{group}:{layer_idx}"
        known = self.state["expert_layers"].get(key)
        if known is None:
            self.state["expert_layers"][key] = n_experts
        elif known != n_experts:
            raise ValueError(
                f"{_layer_prefix(group, layer_idx)}: expert count changed "
                f"between fused tensors ({known} vs {n_experts})")

    def _expert_layer_status(self, group, layer_idx):
        """(n_written, total) slice-tensor counts of an expert layer, from
        the recorded expert count + the output; no source reads (sources
        may already be deleted)."""
        key = f"{group}:{layer_idx}"
        n_experts = self.state["expert_layers"].get(key)
        if n_experts is None:
            raise ValueError(
                f"{_layer_prefix(group, layer_idx)}: expert layer never "
                f"seen during conversion")
        n = 0
        for member in _SRC_MEMBERS:
            for e in range(n_experts):
                if _slice_name(group, layer_idx, e, member) in self.written:
                    n += 1
        return n, 2 * n_experts

    def shard_fully_consumed(self, shard_name):
        """True when every non-stripped tensor of a source shard has been
        written to the output (dense by name, fused via all expert slices
        of ITS OWN member — the partner tensor may live in a later shard
        that is not on disk yet). Drives the downloader's
        hold-until-consumed deletion."""
        for name, t in self._src_header(shard_name).items():
            if classify_group(name) == "strip":
                continue
            m = FUSED_EXPERT_RE.match(name)
            if m:
                group, layer_idx = _match_group_layer(m)
                member = m.group(4)
                for e in range(t.shape[0]):
                    if _slice_name(group, layer_idx, e, member) \
                            not in self.written:
                        return False
            elif name not in self.written:
                return False
        return True

    def _load_or_init(self):
        if not os.path.exists(self.state_path):
            existing = [
                n for n in os.listdir(self.dst_dir)
                if n.endswith(".safetensors") and n.startswith("apus")
            ]
            if existing:
                raise ValueError(
                    f"{self.dst_dir} contains apus shards but no state "
                    f"file; refusing to guess — remove them or restore "
                    f"{STATE_FILE}"
                )
            return _empty_state(self.cfg_hash, self.target_bytes)
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state["format_version"] != FORMAT_VERSION:
            raise ValueError("state format version mismatch")
        if state["config_hash"] != self.cfg_hash:
            raise ValueError(
                "config hash mismatch: the source directory changed since "
                "conversion started; refusing to mix outputs"
            )
        if state["target_shard_bytes"] != self.target_bytes:
            raise ValueError(
                f"target shard size changed ({state['target_shard_bytes']} "
                f"-> {self.target_bytes}); keep it constant across a run"
            )
        return state

    def save_state(self):
        """Atomic state flush: every crash window collapses to the last
        fully recorded step."""
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".state-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    # -- main flow ---------------------------------------------------------

    def convert(self, shard_names=None, progress_cb=None):
        """Convert input shards (all pending, or the named ones)."""
        pending = [
            n for n in list_input_shards(self.src_dir)
            if n not in self.state["inputs_done"]
        ]
        if shard_names is not None:
            wanted = set(shard_names)
            missing = wanted - set(list_input_shards(self.src_dir))
            if missing:
                raise ValueError(f"unknown input shard(s): {sorted(missing)}")
            pending = [n for n in pending if n in wanted]
        for name in pending:
            self._convert_input_shard(name, progress_cb)
            self.state["inputs_done"].append(name)
            self.save_state()
            if progress_cb:
                progress_cb("input_done", shard=name)

    def _convert_input_shard(self, shard_name, progress_cb):
        tensors = self._src_header(shard_name)
        pos = self._shard_pos[shard_name]
        # Stripped (vision tower) tensors: skipped, counted, reported.
        n_strip = sum(1 for n in tensors
                      if classify_group(n) == "strip")
        if n_strip and progress_cb:
            progress_cb("strip", shard=shard_name, ntensors=n_strip)

        # Expert layers touched by this shard. A layer's slabs are assembled
        # only once BOTH fused tensors' home shards have been reached in
        # sorted input order (<= this shard); otherwise the layer defers to
        # the pass of the later home shard. The rule is a pure function of
        # the sorted shard order, so one-shot and driver runs agree.
        layers = set()
        for name, t in tensors.items():
            m = FUSED_EXPERT_RE.match(name)
            if m:
                group, layer_idx = _match_group_layer(m)
                self._record_expert_layer(group, layer_idx, t.shape[0])
                layers.add(_layer_key(group, layer_idx))
        for group, layer_idx in sorted(layers):
            homes = {
                self.weight_map[_fused_src_name(group, layer_idx, member)]
                for member in _SRC_MEMBERS
            }
            if max(self._shard_pos[h] for h in homes) > pos:
                continue   # partner tensor lives in a later shard: defer
            self._assemble_expert_layer(group, layer_idx, progress_cb)

        # Dense tensors: copied through under their HF names, sorted.
        by_group = {"main": [], "mtp": []}
        for name in tensors:
            if FUSED_EXPERT_RE.match(name):
                continue
            g = classify_group(name)
            if g == "strip":
                continue
            by_group[g].append(name)
        for group, names in by_group.items():
            stream = self.streams[group]
            for name in sorted(names):
                if name in self.written:
                    continue
                t = tensors[name]
                stream.ensure_capacity(t.nbytes, self.target_bytes,
                                       progress_cb)
                stream.append_tensor(t, progress_cb)
                self.written.add(name)

    def _assemble_expert_layer(self, group, layer_idx, progress_cb):
        """Slice the layer's two fused expert tensors and coalesce per
        expert: slab = gate_up_proj[e] followed by down_proj[e], contiguous.

        The two source tensors may live in different input shards; both
        must be on disk (the caller's deferral rule plus the download
        driver's hold-until-consumed deletion guarantee it). Resumable:
        slice tensors already in the output are skipped, so a crash
        mid-layer resumes at the first unwritten slice, still contiguous
        at the open shard's tail.
        """
        srcs = {}
        for member in _SRC_MEMBERS:
            name = _fused_src_name(group, layer_idx, member)
            home = self.weight_map[name]
            path = os.path.join(self.src_dir, home)
            if not os.path.exists(path):
                raise ValueError(
                    f"{name}: home shard {home} not on disk; cannot "
                    f"assemble expert layer "
                    f"{_layer_prefix(group, layer_idx)}"
                )
            srcs[member] = self._src_header(home)[name]
        gu, d = srcs["gate_up_proj"], srcs["down_proj"]
        if gu.shape[0] != d.shape[0]:
            raise ValueError(
                f"{_layer_prefix(group, layer_idx)}: expert count mismatch "
                f"({gu.shape[0]} vs {d.shape[0]})")
        n_experts = gu.shape[0]
        self._record_expert_layer(group, layer_idx, n_experts)
        slices = {}
        for member, t in srcs.items():
            if t.nbytes % n_experts:
                raise ValueError(
                    f"{t.name}: {t.nbytes} bytes not divisible into "
                    f"{n_experts} expert slices")
            slices[member] = t.nbytes // n_experts
        stream = self.streams[group]
        for e in range(n_experts):
            slab = []
            for member in _SRC_MEMBERS:   # fixed intra-slab order
                out_name = _slice_name(group, layer_idx, e, member)
                if out_name in self.written:
                    continue
                t = srcs[member]
                slab.append(SrcTensor(
                    out_name, t.dtype, t.shape[1:], t.src_path,
                    t.file_offset + e * slices[member], slices[member]))
            if not slab:
                continue
            # Capacity check covers both members so a slab can never
            # straddle a shard boundary.
            stream.ensure_capacity(sum(s.nbytes for s in slab),
                                   self.target_bytes, progress_cb)
            for s in slab:
                stream.append_tensor(s, progress_cb)
                self.written.add(s.name)

    def finalize(self, progress_cb=None):
        """Seal any open shards and (re)write the manifest. Idempotent."""
        # A complete conversion must leave no deferred expert layer.
        if set(self.state["inputs_done"]) == set(self.input_shards):
            for key in self.state["expert_layers"]:
                group, layer_idx = key.split(":")
                n, total = self._expert_layer_status(group, int(layer_idx))
                if n != total:
                    raise ValueError(
                        f"conversion incomplete: expert layer "
                        f"{_layer_prefix(group, int(layer_idx))} has "
                        f"{n}/{total} slice tensors written (a home shard "
                        f"was never converted?)")
        for stream in self.streams.values():
            stream.seal(progress_cb)
        manifest = self.build_manifest()
        path = os.path.join(self.dst_dir, MANIFEST_FILE)
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".manifest-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Standard safetensors index for the C loader (st.h/cache.h read
        # model.safetensors.index.json, not the apus manifest).
        std_index = {
            "metadata": {"total_size": sum(t["nbytes"] for t in
                                           manifest["tensor_map"].values())},
            "weight_map": {name: t["shard"]
                           for name, t in manifest["tensor_map"].items()},
        }
        fd, tmp = tempfile.mkstemp(dir=self.dst_dir, prefix=".stindex-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(std_index, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(self.dst_dir,
                                     "model.safetensors.index.json"))
        self.state["complete"] = True
        self.save_state()
        return manifest

    # -- introspection ------------------------------------------------------

    def output_tensor_map(self):
        """name -> (shard_file, absolute_offset, nbytes, dtype, shape) for
        everything written so far, from sealed headers + open entries."""
        out = {}
        for group, stream in self.streams.items():
            gstate = stream.gstate
            for shard in gstate["sealed"]:
                path = os.path.join(self.dst_dir, shard)
                for name, t in read_st_header(path).items():
                    out[name] = (shard, t.file_offset, t.nbytes, t.dtype,
                                 t.shape)
            open_ = gstate["open"]
            if open_ is not None:
                base = 8 + HEADER_RESERVE
                for e in open_["entries"]:
                    out[e["name"]] = (open_["file"], base + e["offset"],
                                      e["nbytes"], e["dtype"], e["shape"])
        return out

    def build_manifest(self):
        tensor_map = {}
        slabs = []
        groups = {}
        for group, stream in self.streams.items():
            gstate = stream.gstate
            files = sorted(gstate["sealed"])
            groups[group] = files
            expert_parts = {}
            for shard in files:
                for name, t in read_st_header(
                        os.path.join(self.dst_dir, shard)).items():
                    tensor_map[name] = {
                        "shard": shard,
                        "offset": t.file_offset,   # absolute file offset
                        "nbytes": t.nbytes,
                        "dtype": t.dtype,
                        "shape": t.shape,
                    }
                    m = OUT_EXPERT_RE.match(name)
                    if m:
                        grp, lidx = _match_group_layer(m)
                        eid = int(m.group(4))
                        # Slab records number the MTP block's layers
                        # num_hidden_layers + K (base engine convention).
                        slab_layer = lidx if grp == "main" else \
                            self.num_hidden_layers + lidx
                        key = (slab_layer, eid)
                        expert_parts.setdefault(key, []).append(
                            (t.file_offset, t.nbytes, shard))
            for (layer, expert), parts in sorted(expert_parts.items()):
                if len(parts) != len(SLAB_MEMBERS):
                    raise ValueError(
                        f"expert L{layer}/{expert}: {len(parts)} tensors in "
                        f"output, expected {len(SLAB_MEMBERS)}"
                    )
                parts.sort()
                shards = {p[2] for p in parts}
                if len(shards) != 1:
                    raise ValueError(
                        f"expert L{layer}/{expert} straddles shards {shards}")
                start = parts[0][0]
                contiguous = all(
                    parts[k][0] + parts[k][1] == parts[k + 1][0]
                    for k in range(len(parts) - 1)
                )
                if not contiguous:
                    raise ValueError(
                        f"expert L{layer}/{expert} tensors not contiguous")
                slabs.append({
                    "layer": layer,
                    "expert": expert,
                    "shard": parts[0][2],
                    "offset": start,
                    "nbytes": sum(p[1] for p in parts),
                })
        return {
            "format_version": FORMAT_VERSION,
            "config_hash": self.cfg_hash,
            "offset_base": "file",
            "header_reserve": HEADER_RESERVE,
            "shard_groups": groups,
            "ntensors": len(tensor_map),
            "tensor_map": tensor_map,
            "expert_slabs": slabs,
            # Audit: vision-tower tensors stripped from the source
            # (text-only scope); their bytes are NOT in the container.
            "stripped_tensors": sum(
                1 for n in self.weight_map
                if classify_group(n) == "strip"),
        }


# --------------------------------------------------------------------------
# Verification: byte-compare source tensors against the output
# --------------------------------------------------------------------------

def verify_source(src_dir, dst_dir, shard_names=None, log=print,
                  strict=True):
    """Byte-compare every tensor of the given (or all converted) source
    shards against its copy in the output. Fused expert tensors are
    verified slice by slice: each expert's output bytes against the
    matching source region. Stripped (vision) tensors must be ABSENT from
    the output. Returns the number of source tensors verified; raises on
    the first mismatch.

    strict=True (default) raises when a source tensor is missing from the
    output. strict=False skips unwritten tensors — used by the download
    driver, where a layer whose partner tensor lives in a later shard is
    legitimately not written yet.
    """
    conv = Converter.__new__(Converter)   # lightweight: no validation writes
    conv.src_dir, conv.dst_dir = src_dir, dst_dir
    conv.state_path = os.path.join(dst_dir, STATE_FILE)
    with open(conv.state_path, "r", encoding="utf-8") as f:
        conv.state = json.load(f)
    conv.target_bytes = conv.state["target_shard_bytes"]
    conv.streams = {
        g: GroupStream(g, dst_dir, conv.state, lambda: None)
        for g in ("main", "mtp")
    }
    out_map = conv.output_tensor_map()

    def cmp_bytes(t, src_off, oshard, ooff, nbytes, label):
        with open(t.src_path, "rb") as fs, \
                open(os.path.join(dst_dir, oshard), "rb") as fo:
            fs.seek(src_off)
            fo.seek(ooff)
            remaining = nbytes
            while remaining:
                a = fs.read(min(COPY_CHUNK, remaining))
                b = fo.read(min(COPY_CHUNK, remaining))
                if a != b:
                    raise ValueError(
                        f"{label}: BYTE MISMATCH between source and output")
                remaining -= len(a)

    shards = list_input_shards(src_dir)
    if shard_names is not None:
        shards = [s for s in shards if s in set(shard_names)]
    done = set(conv.state["inputs_done"])
    nverified = 0
    for shard in shards:
        if shard not in done:
            continue
        src_tensors = read_st_header(os.path.join(src_dir, shard))
        nshard = 0
        for name, t in src_tensors.items():
            if classify_group(name) == "strip":
                if name in out_map:
                    raise ValueError(
                        f"{name}: stripped (vision) tensor present in "
                        f"output")
                continue
            m = FUSED_EXPERT_RE.match(name)
            if m:
                group, layer_idx = _match_group_layer(m)
                member = m.group(4)
                n_experts = t.shape[0]
                slice_nbytes = t.nbytes // n_experts
                npending = 0
                for e in range(n_experts):
                    out_name = _slice_name(group, layer_idx, e, member)
                    if out_name not in out_map:
                        if strict:
                            raise ValueError(
                                f"{out_name}: missing from output")
                        npending += 1
                        continue
                    oshard, ooff, onbytes, odtype, oshape = \
                        out_map[out_name]
                    if onbytes != slice_nbytes or odtype != t.dtype or \
                            oshape != t.shape[1:]:
                        raise ValueError(
                            f"{out_name}: metadata mismatch vs output")
                    cmp_bytes(t, t.file_offset + e * slice_nbytes,
                              oshard, ooff, slice_nbytes, out_name)
                if npending == 0:
                    nverified += 1
                    nshard += 1
                continue
            if name not in out_map:
                if strict:
                    raise ValueError(f"{name}: missing from output")
                continue
            oshard, ooff, onbytes, odtype, oshape = out_map[name]
            if onbytes != t.nbytes or odtype != t.dtype or oshape != t.shape:
                raise ValueError(f"{name}: metadata mismatch vs output")
            cmp_bytes(t, t.file_offset, oshard, ooff, t.nbytes, name)
            nverified += 1
            nshard += 1
        log(f"verify: {shard}: {nshard} tensors byte-identical")
    return nverified


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("convert", "verify", "finalize"):
        sp = sub.add_parser(cmd)
        sp.add_argument("src_dir")
        sp.add_argument("dst_dir")
        sp.add_argument("--shard", action="append", default=None,
                        help="limit to these input shard(s); repeatable")
        if cmd == "convert":
            sp.add_argument("--target-bytes", type=int,
                            default=DEFAULT_TARGET_BYTES)
    args = p.parse_args(argv)

    if args.cmd == "convert":
        conv = Converter(args.src_dir, args.dst_dir, args.target_bytes)
        conv.convert(shard_names=args.shard)
        manifest = conv.finalize()
        stripped = manifest["stripped_tensors"]
        if stripped:
            print(f"stripped {stripped} vision-tower tensors "
                  f"(text-only scope)")
    elif args.cmd == "finalize":
        conv = Converter(args.src_dir, args.dst_dir)
        conv.finalize()
    elif args.cmd == "verify":
        n = verify_source(args.src_dir, args.dst_dir, args.shard)
        print(f"verify: OK, {n} tensors byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
