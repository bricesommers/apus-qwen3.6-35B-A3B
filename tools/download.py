#!/usr/bin/env python3
"""apus-qwen M1 — disk-safe download + convert driver for Qwen3.6-35B-A3B.

Downloads the 26 HF checkpoint shards (~67 GiB total) ONE AT A TIME and
converts each into the apus container before fetching the next, so peak
disk usage stays at roughly (growing apus output) + (one or two ~2.6 GiB
source shards) instead of 67 GiB + 67 GiB.

Per shard the pipeline is:

    download (resumable)  ->  convert (resumable, tools/convert.py)
                          ->  byte-verify against output
                          ->  record "done" in the state file
                          ->  delete the source shard

Deletion is HOLD-UNTIL-CONSUMED: the Qwen checkpoint places a layer's two
fused expert tensors in different input shards for 17 of the 41 MoE
layers, and a layer's expert slabs can only be assembled once both are on
disk (see tools/convert.py's module docstring). A source shard is
therefore deleted only after EVERY tensor in it (dense or expert slice)
has been written to the output and byte-verified — a shard whose expert
layer is still waiting for its partner is held in the landing zone and
swept up by a later shard's pass. At most one or two source shards are
held at a time with the real checkpoint's layout.

The driver can be killed at any point and restarted cleanly:

  * A download interrupted mid-stream resumes from the partial bytes
    (HTTP Range resume; byte-range resume in --source-dir mode).
  * A conversion interrupted mid-shard is resumed by convert.py's own
    state machine (verify-before-trust + truncation of torn writes).
  * A source shard is deleted only AFTER its content is byte-verified in
    the output and the "done" mark is durably recorded. If the process
    dies between recording and deleting, startup garbage collection
    removes the leftover source file.

Network mode downloads over HTTPS with unlimited retries (HF_TOKEN is
honored but never logged). For tests and offline rehearsal,
--source-dir DIR treats a local directory as the "remote", copying shards
with the same resumable state machine.

Usage:
    python tools/download.py [--repo Qwen/Qwen3.6-35B-A3B] \
        [--work weights/work-qwen] [--out weights/apus-qwen] \
        [--token TOKEN] [--target-bytes N]
    python tools/download.py --source-dir DIR [--work DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert as apus_convert  # noqa: E402

DEFAULT_REPO = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_WORK_DIR = "weights/work-qwen"
DEFAULT_OUT_DIR = "weights/apus-qwen"

STATE_FILE = "apus.download.state.json"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
COPY_CHUNK = 8 * 1024 * 1024
# Socket timeout for connect AND reads: a silently stalled connection
# (e.g. network drop mid-stream) must raise so the retry loop kicks in,
# instead of hanging forever at 0% CPU.
HTTP_TIMEOUT = 60


class Driver:
    def __init__(self, work_dir, out_dir, repo=None, token=None,
                 source_dir=None, target_bytes=apus_convert.DEFAULT_TARGET_BYTES,
                 progress_cb=None):
        self.work_dir = work_dir
        self.src_dir = os.path.join(work_dir, "src")  # landing zone
        self.out_dir = out_dir
        self.repo = repo
        self.token = token
        self.source_dir = source_dir
        self.target_bytes = target_bytes
        self.progress_cb = progress_cb
        self._size_cache = {}
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        self.state_path = os.path.join(work_dir, STATE_FILE)
        self.state = self._load_state()

    # -- state --------------------------------------------------------------

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"format_version": 1, "files": {}}

    def save_state(self):
        fd, tmp = tempfile.mkstemp(dir=self.work_dir, prefix=".dl-state-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def fstate(self, name):
        return self.state["files"].setdefault(name, {"status": "pending"})

    # -- shard list ----------------------------------------------------------

    # Small support files the engine needs next to the converted container:
    # config + tokenizer + chat template (weights are useless without them).
    # Missing files are tolerated. config.json is also fetched above for
    # conversion, into work.
    SUPPORT_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt", "generation_config.json",
                     "chat_template.jinja")

    def shard_names(self):
        index_path = os.path.join(self.work_dir, INDEX_NAME)
        if not os.path.exists(index_path):
            self._fetch_small(INDEX_NAME)
        if not os.path.exists(os.path.join(self.work_dir, CONFIG_NAME)):
            try:
                self._fetch_small(CONFIG_NAME)
            except Exception:
                pass  # config is optional for conversion
        for name in self.SUPPORT_FILES:
            dst = os.path.join(self.out_dir, name)
            if not os.path.exists(dst):
                try:
                    self._fetch_small(name)
                    shutil.copyfile(os.path.join(self.work_dir, name), dst)
                except Exception:
                    pass  # tokenizer files are only needed at run time
        # convert.py expects config/index next to the shards it reads.
        for small in (INDEX_NAME, CONFIG_NAME):
            src = os.path.join(self.work_dir, small)
            dst = os.path.join(self.src_dir, small)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted(set(weight_map.values()))

    def _fetch_small(self, name):
        if self.source_dir:
            shutil.copyfile(os.path.join(self.source_dir, name),
                            os.path.join(self.work_dir, name))
        else:
            self._hf_download(name, os.path.join(self.work_dir, name))

    # -- download backends ----------------------------------------------------

    def _hf_download(self, name, dest):
        """Download via huggingface_hub (resumes partial downloads itself),
        then move into place."""
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=self.repo, filename=name, token=self.token,
        )
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copyfile(path, dest)
        return dest

    def _expected_size(self, name):
        """Authoritative byte size of a remote file (HF xet: x-linked-size).

        Retries transient network failures forever with capped backoff —
        an unattended 67 GiB download must survive long outages."""
        import time
        import urllib.request
        if name in self._size_cache:
            return self._size_cache[name]
        url = f"https://huggingface.co/{self.repo}/resolve/main/{name}"
        backoff = 1.0
        attempt = 0
        while True:
            attempt += 1
            req = urllib.request.Request(url, method="HEAD")
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    h = resp.headers
                    size = h.get("x-linked-size") or h.get("Content-Length")
                if size is None:
                    raise IOError(f"cannot determine remote size for {name}")
                size = int(size)
                self._size_cache[name] = size
                return size
            except Exception as e:
                print(f"[download] HEAD {name}: attempt {attempt} failed: "
                      f"{e}; retrying in {backoff:.0f}s",
                      file=sys.stderr, flush=True)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    def _download_shard(self, name):
        """Resumable download of one shard into the landing zone."""
        dest = os.path.join(self.src_dir, name)
        if self.source_dir:
            self._local_resume_copy(os.path.join(self.source_dir, name), dest)
        else:
            expected = self._expected_size(name)
            part = dest + ".part"
            # Heal a truncated "final" file from a pre-fix run: demote it to
            # the .part file and resume instead of trusting it.
            if os.path.exists(dest):
                have = os.path.getsize(dest)
                if have == expected:
                    return dest
                if os.path.exists(part):
                    os.remove(part)
                os.replace(dest, part)
            self._http_resume_copy(name, part, expected)
            if os.path.getsize(part) != expected:
                raise IOError(f"{name}: size mismatch after download")
            os.replace(part, dest)
        return dest

    def _local_resume_copy(self, src, dest):
        """Offline stand-in for HTTP resume: append-copy honoring an existing
        .part file, then rename. Exercises the same state machine."""
        part = dest + ".part"
        total = os.path.getsize(src)
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > total:
            os.remove(part)
            have = 0
        with open(src, "rb") as fin, open(part, "ab") as fout:
            fin.seek(have)
            while True:
                chunk = fin.read(COPY_CHUNK)
                if not chunk:
                    break
                fout.write(chunk)
                fout.flush()
                os.fsync(fout.fileno())
                if self.progress_cb:
                    self.progress_cb("download_chunk", file=name_of(src),
                                     have=fout.tell(), total=total)
        os.replace(part, dest)

    def _http_resume_copy(self, name, part, expected):
        """Size-verified HTTP download with Range resume and unlimited retries.

        Never treats an early EOF (dropped connection) as completion: loops
        until the .part file holds exactly `expected` bytes. Safe against
        long outages — retries forever with capped exponential backoff."""
        import time
        import urllib.request
        url = f"https://huggingface.co/{self.repo}/resolve/main/{name}"
        backoff = 1.0
        attempt = 0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have == expected:
                return
            if have > expected:
                os.remove(part)  # cannot be a prefix of the real file
                have = 0
            attempt += 1
            req = urllib.request.Request(url)
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    if have and resp.status != 206:
                        # Server ignored the range: start over.
                        have = 0
                    mode = "ab" if have else "wb"
                    with open(part, mode) as fout:
                        while True:
                            chunk = resp.read(COPY_CHUNK)
                            if not chunk:
                                break
                            fout.write(chunk)
                            have += len(chunk)
                        fout.flush()
                        os.fsync(fout.fileno())
                backoff = 1.0  # a clean EOF resets the backoff
            except Exception as e:
                print(f"[download] {name}: attempt {attempt} failed at "
                      f"{have}/{expected} bytes: {e}; retrying in "
                      f"{backoff:.0f}s", file=sys.stderr, flush=True)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    # -- pipeline ---------------------------------------------------------------

    def gc(self):
        """Remove source shards whose conversion was recorded done but whose
        deletion was lost to a crash."""
        for name, st in self.state["files"].items():
            path = os.path.join(self.src_dir, name)
            if st.get("status") == "done" and os.path.exists(path):
                os.remove(path)

    def _sweep_consumed(self, names, conv):
        """Verify + delete every landing-zone shard whose tensors have all
        been consumed into the output (including held shards whose expert
        layer was just assembled by the current pass)."""
        for held in names:
            hst = self.fstate(held)
            if hst["status"] not in ("downloaded", "converted"):
                continue
            path = os.path.join(self.src_dir, held)
            if not os.path.exists(path):
                continue
            if not conv.shard_fully_consumed(held):
                continue  # expert layer still waiting for a later shard
            apus_convert.verify_source(self.src_dir, self.out_dir, [held],
                                       log=lambda _msg: None)
            hst["status"] = "done"
            self.save_state()
            os.remove(path)
            if self.progress_cb:
                self.progress_cb("shard_done", file=held)

    def run(self):
        self.gc()
        names = self.shard_names()
        for name in names:
            st = self.fstate(name)
            if st["status"] == "done":
                continue
            dest = os.path.join(self.src_dir, name)
            if st["status"] == "downloaded" and not self.source_dir:
                # Revalidate against the authoritative remote size: a shard
                # truncated by a dropped connection must be re-fetched, not
                # handed to the converter.
                try:
                    good = (os.path.exists(dest) and
                            os.path.getsize(dest) == self._expected_size(name))
                except Exception:
                    good = os.path.exists(dest)  # offline: defer to converter
                if not good:
                    st["status"] = "pending"
                    self.save_state()
            if st["status"] == "pending":
                self._download_shard(name)
                st["status"] = "downloaded"
                st["size"] = os.path.getsize(dest)
                self.save_state()
                if self.progress_cb:
                    self.progress_cb("downloaded", file=name)
            conv = None
            if st["status"] == "downloaded":
                # Convert (idempotent/resumable). Also assembles any expert
                # layer whose partner fused tensor lives in a held shard.
                conv = apus_convert.Converter(self.src_dir, self.out_dir,
                                              self.target_bytes)
                conv.convert(shard_names=[name], progress_cb=self.progress_cb)
                st["status"] = "converted"
                self.save_state()
            # Held shards (status "converted", still on disk) may have
            # become fully consumed by this pass; verify + delete them.
            if conv is None:
                conv = apus_convert.Converter(self.src_dir, self.out_dir,
                                              self.target_bytes)
            self._sweep_consumed(names, conv)
        # All shards converted: seal the last open shards, write manifest.
        conv = apus_convert.Converter(self.src_dir, self.out_dir,
                                      self.target_bytes)
        conv.finalize()
        if self.progress_cb:
            self.progress_cb("complete")


def name_of(path):
    return os.path.basename(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--repo", default=DEFAULT_REPO,
                     help=f"HF repo id (default {DEFAULT_REPO})")
    src.add_argument("--source-dir", help="offline mode: treat DIR as the remote")
    p.add_argument("--work", default=DEFAULT_WORK_DIR,
                   help="work dir: state file + landing zone "
                        f"(default {DEFAULT_WORK_DIR})")
    p.add_argument("--out", default=DEFAULT_OUT_DIR,
                   help=f"apus container output dir (default {DEFAULT_OUT_DIR})")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--target-bytes", type=int,
                   default=apus_convert.DEFAULT_TARGET_BYTES)
    args = p.parse_args(argv)

    driver = Driver(args.work, args.out, repo=args.repo, token=args.token,
                    source_dir=args.source_dir,
                    target_bytes=args.target_bytes,
                    progress_cb=lambda ev, **kw: print(f"[{ev}] {kw}"))
    driver.run()
    print("download+convert complete:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
