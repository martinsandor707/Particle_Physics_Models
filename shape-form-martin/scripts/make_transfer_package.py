#!/usr/bin/env python3
"""H200 transfer package (PLAN.md section 15.1 / 16): the git-ignored files the moved steps need, as rsync/tar file
lists plus checksums, and a verify mode for the machine that receives them.

    uv run python -B shape-form-martin/scripts/make_transfer_package.py list               # laptop (source machine)
    uv run python -B shape-form-martin/scripts/make_transfer_package.py verify [--json]    # H200 before Gate 2c
    uv run python -B shape-form-martin/scripts/make_transfer_package.py verify --after-gate2c     # H200 after Gate 2c
    uv run python -B shape-form-martin/scripts/make_transfer_package.py verify --source-machine   # on the laptop

Standard library only (never imports torch or numpy), so it runs before the environment exists (plain `python3`
>= 3.11 works as well).

Groups (MANIFEST.json holds every field; the tables below are the source of truth, each entry justified by a reader
in the code - there is no blind directory walk for the package; glob patterns are only used together with an
expected count):
    a    tracked in the pushed branch; arrives with `git pull`. Some are laptop RECORDS that the code trusts
         (registry rows, done-markers, guard spec, ...): they are hazards on another GPU; verify lists them.
    b1   inputs: git-ignored, needed as-is on the H200 at the SAME repo-relative path.
    b2   laptop records: git-ignored per-event arrays (and monitors) computed on this laptop's Monte-Carlo sample
         points. Never copied to their live path (the code would pair H200 scores with them silently); copied into
         the quarantine root shape-form-martin/cache/laptop_records/, repo-relative path preserved, for comparison.
    c    regenerated on the H200 (command or "no command exists yet", laptop time, reproducibility).

Transfer phases (user decisions of 25 Nov 2026). EIGHT files are DEFERRED: they travel only after Gate 2c has passed,
so that their physical absence on the H200 shows that nothing there read test events before the final evaluation:
  - the seven TEST-DERIVED files (P4): the test cache, the earlier study's per-event test arrays test_v3_nll.npz, the
    test event order test_event_number.npy, and the four laptop test arrays {mlp,formula}_test_nll{,_ref1056}.npy;
  - the raw Parquet hit table (flag contains_test_events): not derived from test events, but it holds the raw hits
    of all events, the test events included (the split is made only at cache build).
Four lists:
    files_b_inputs.txt                 b1, now            -> <repo>/
    files_b_laptop_records.txt         b2, now            -> <repo>/shape-form-martin/cache/laptop_records/
    files_b_test_derived_inputs.txt    b1, after Gate 2c  -> <repo>/   (3 test-derived inputs + the raw Parquet table)
    files_b_test_derived_records.txt   b2, after Gate 2c  -> <repo>/shape-form-martin/cache/laptop_records/
             rsync -a --files-from=shape-form-martin/transfer/<list> <repo>/ <user>@<h200>:<destination root>

Test cache. shape-form-martin/cache/pion_s1_test.pt is COPIED after Gate 2c (not rebuilt) and treated here as an
OPAQUE BYTE STREAM: the only access is a chunked binary read inside sha256_file() (SHA-256 of the file's bytes) plus
os.stat() for the size. It is never unpickled, never loaded with torch, never passed to sfm.data.load_sets. A byte
hash is not a test access (nothing is logged in results/test_access.log). The file must stay unopened as data until
the final evaluation after Gate 2c.

CSV. No *.csv is ever listed, stat'ed for the package or hashed: sha256_file() and the table expansion refuse any
path ending in .csv, and the absence search skips *.csv names before any stat (the 15 GB raw CSV is not part of the
package; the Parquet copy is, after Gate 2c).

Outputs of `list` (in shape-form-martin/transfer/): the four lists, SHA256SUMS (sha256sum -c format, one delimited
section per list with its destination root, phase, contents and check command in the header) and MANIFEST.json.
`verify` exits non-zero on any failure. Default (before Gate 2c): every "now" file present and correct, and FAIL if
any deferred file exists on this machine: a test-derived file at its live path or under the quarantine root (any
content) or with the same name and size anywhere in the working tree (confirmed by SHA-256); the raw Parquet table at
its live path (any content) or with the same name and size elsewhere (confirmed by one SHA-256 of 2.6 GB; a Parquet
file of another size is information only). `--after-gate2c`: additionally all eight present and correct.
A stream-dependent laptop record found at its LIVE path always fails (the exact `mv` that would quarantine it is
printed, never executed).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import posixpath
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

SCHEMA = 2
MARTIN = "shape-form-martin"
PION = "shape-form-proton/results/particles/pion"
TRANSFER_REL = f"{MARTIN}/transfer"
QUARANTINE_REL = f"{MARTIN}/cache/laptop_records"          # inside the git-ignored cache/ (shape-form-martin/.gitignore)
TEST_CACHE_REL = f"{MARTIN}/cache/pion_s1_test.pt"          # opaque byte stream (module docstring)
META_REL = f"{MARTIN}/cache/pion_s1_meta.json"
CHECKSUMS_REL = f"{MARTIN}/results/m0_parity/checksums.json"
REGISTRY_REL = f"{MARTIN}/results/registry.jsonl"
CHUNK = 1 << 22

# The four file lists: (group, deferred) -> list file, transfer phase, destination root (repo-relative; "" = repo).
# deferred = test_derived or contains_test_events (copied only after Gate 2c).
LISTS = (
    {"key": "inputs", "file": "files_b_inputs.txt", "group": "b1", "deferred": False, "phase": "now",
     "root": "", "holds": "group b1 inputs needed now (no test-event content)"},
    {"key": "laptop_records", "file": "files_b_laptop_records.txt", "group": "b2", "deferred": False,
     "phase": "now", "root": QUARANTINE_REL, "holds": "group b2 laptop records on validation (comparison only)"},
    {"key": "test_derived_inputs", "file": "files_b_test_derived_inputs.txt", "group": "b1", "deferred": True,
     "phase": "after_gate2c", "root": "",
     "holds": "the three test-derived inputs (test cache, test_v3_nll.npz, test_event_number.npy) plus the raw "
              "Parquet table (contains_test_events; not test-derived)"},
    {"key": "test_derived_records", "file": "files_b_test_derived_records.txt", "group": "b2", "deferred": True,
     "phase": "after_gate2c", "root": QUARANTINE_REL,
     "holds": "the four test-derived laptop records ({mlp,formula}_test_nll{,_ref1056}.npy)"},
)
PHASES = ("now", "after_gate2c")
PHASE_LABEL = {"now": "now", "after_gate2c": "after Gate 2c"}
OUTPUT_NAMES = tuple(l["file"] for l in LISTS) + ("SHA256SUMS", "MANIFEST.json")
_TEST_WORD = re.compile(r"(^|[^a-z])test([^a-z]|$)")      # 'test' as a word in a file name (not 'latest')

TEST_CACHE_POLICY = (
    "shape-form-martin/cache/pion_s1_test.pt is test-derived: it is NOT copied now but only after Gate 2c (list "
    "files_b_test_derived_inputs.txt), byte for byte (not rebuilt), and verified by its SHA-256 and size only: the "
    "transfer script reads it as an opaque byte stream (chunked binary read in sha256_file) and never unpickles, "
    "torch-loads or load_sets() it. A byte hash is not a test access and is not logged. The file must stay unopened "
    "as data until the single final evaluation after Gate 2c.")
TEST_DERIVED_POLICY = (
    "Seven files carry per-event or per-hit information about the test events: the test cache "
    "shape-form-martin/cache/pion_s1_test.pt, the earlier study's stored per-event test arrays "
    "shape-form-proton/results/particles/pion/test_v3_nll.npz, the test event order "
    "shape-form-martin/results/m0_parity/test_event_number.npy, and the four laptop test arrays "
    "shape-form-martin/results/m0_parity/{mlp,formula}_test_nll{,_ref1056}.npy. Rule (user decision P4, 25 Nov "
    "2026): they are not copied with the rest of the package. They sit in their own lists "
    "(files_b_test_derived_inputs.txt to the same path, files_b_test_derived_records.txt into the quarantine root) "
    "and are copied to the H200 only after Gate 2c has passed. Until then `verify` FAILS if any of them exists on "
    "the H200 - at its live path, under the quarantine root, or under the same file name with the same size anywhere "
    "in the working tree, confirmed by SHA-256 (the test cache only by size and a byte hash) - and `verify "
    "--after-gate2c` requires all seven, verified by size and SHA-256. Reason: physical absence on the machine of "
    "record shows that nothing there read them before the final evaluation, without having to trust the test-access "
    "log (it has no host field, and load_sets writes its line before it opens the file, data.py:286-287) or a "
    "reading of the code. No moved step, import or campaign test reads the three test-derived inputs (static reading "
    "plus a run with them made invisible by an audit hook; see absence_before_gate2c). The raw Parquet table is "
    "deferred with them (same list files_b_test_derived_inputs.txt) but is not test-derived; see policies.raw_table.")
RAW_TABLE_POLICY = (
    "The raw Parquet hit table shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet "
    "is NOT test-derived - it is the source data, computed from no score and no split - but it CONTAINS the raw hits of "
    "all 99,689 events, the 14,956 test events included, because the train/val/test split is made only when the cache "
    "is built (sfm/data.py:150-160, seeded per momentum bin) and sfm.data.build_cache writes a test cache from it "
    "(data.py:227). The manifest therefore marks it contains_test_events = true and test_derived = false. User "
    "decision of 25 Nov 2026: it is deferred like the seven test-derived files - it travels in "
    "files_b_test_derived_inputs.txt and is copied to the H200 only after Gate 2c has passed - so that before Gate 2c "
    "no file holding test-event hits exists on the H200. Until then `verify` FAILS if it is at its live path (any "
    "content) or anywhere else in the working tree with the same name and size (confirmed by one SHA-256 of the "
    "2.6 GB file); a Parquet file of another size (for example a train/val-only extract) is reported as information. "
    "`verify --after-gate2c` requires it present with its size and SHA-256 (and warns if its mtime differs from "
    "checksums.json). No moved step, import or campaign test reads it while the caches are copied.")
TEST_SUMMARY_NOW = [
    {"path": f"{PION}/test_v3.json",
     "why_now": "read by every legacy formula load (shapelab.final_round / final_kwargs, shape-form-proton/"
                "shapelab.py:358-386, via sfm.legacy.load_component_formula, families.load_legacy_core, "
                "registry._default_bench_models): it defines the final formula's structure (variant, variant_kwargs, "
                "degrees)",
     "test_content": "summary numbers of the earlier study only: n_events, the test means of 10 models, paired "
                     "differences with SE, gap-closed fractions and two 20-entry per-momentum-bin mean differences; "
                     "no per-event or per-hit values. The headline test means are already printed in PLAN.md."},
    {"path": META_REL,
     "why_now": "read by sfm.data.load_meta (data.py:300) and registry.validation_reference (registry.py:144-145, "
                "the train/val SHA-256 keys the validation fingerprint); it holds the reference hashes of both caches",
     "test_content": "summary metadata only: counts.test, files.test (SHA-256 and size of the test cache) and "
                     "splits.test (aggregate statistics of the test split: events, hits, extremes of p, theta, zb, "
                     "z_local, r and evis, hits per event min/median/max, events per momentum bin); no event "
                     "identities, no per-event or per-hit arrays, no scores."},
]
TEST_EVENTS_IN_PARQUET = (
    "State after the user decision of 25 Nov 2026: the Parquet raw table, which holds the hits of ALL 99,689 events "
    "including the 14,956 test events, is DEFERRED to after Gate 2c (list files_b_test_derived_inputs.txt; "
    "policies.raw_table). Before Gate 2c, therefore, no file holding test-event hits or per-event test information is "
    "on the H200: neither the seven test-derived files nor the raw table, and verify checks both. build_cache.py "
    "cannot run there (its only source is absent: sfm/data.py:55 raises FileNotFoundError before anything is "
    "written), so no test cache can be rebuilt either. The 15 GB CSV is never copied. What this costs: the "
    "zero-training geometry arm needs the absolute shower origin of the validation events, which only the Parquet "
    "file holds (the cache stores zb, theta and phi, not the origin); it must be computed on the laptop from "
    "train/val rows only and shipped as a small file keyed by event number (group c entry 'absolute shower origin', "
    "not yet existing). After Gate 2c, any code that reads the Parquet file must still keep to the event numbers it "
    "is entitled to.")
CSV_POLICY = ("No *.csv is listed, stat'ed for the package or hashed (the 15 GB raw CSV is not part of the package; "
              "only the Parquet copy is, after Gate 2c). sha256_file() and the table expansion refuse any .csv path; "
              "the absence search of verify skips *.csv names before any stat.")


def is_deferred(e: dict) -> bool:
    """Copied only after Gate 2c: test-derived files and files that contain test events (the raw Parquet table)."""
    return bool(e.get("test_derived")) or bool(e.get("contains_test_events"))


def list_for(group: str, deferred: bool) -> dict:
    return next(l for l in LISTS if l["group"] == group and l["deferred"] == bool(deferred))


def _dest_root(l: dict) -> str:
    return "<repo>/" + (f"{l['root']}/" if l["root"] else "")


def rsync_command(l: dict) -> str:
    return f"rsync -a --files-from={TRANSFER_REL}/{l['file']} <repo>/ <user>@<h200>:{_dest_root(l)}"


def check_command(l: dict) -> str:
    """The `sha256sum -c` one-liner of one SHA256SUMS section, run from <repo>/ on the receiving machine."""
    sed = f"sed -n '/^# BEGIN {l['key']}$/,/^# END {l['key']}$/p'"
    if not l["root"]:
        return f"{sed} {TRANSFER_REL}/SHA256SUMS | sha256sum -c"
    return f"(cd {l['root']} && {sed} {posixpath.relpath(TRANSFER_REL, l['root'])}/SHA256SUMS | sha256sum -c)"


class RefusedFile(Exception):
    """A path this script must never list or open (any *.csv)."""


class PackageError(Exception):
    """A check of `list` failed; nothing is written."""


# ======================================================================================================================
# Group a: tracked files (arrive with git). `laptop_record` + `hazard`: what the file makes the code do on the H200.
# ======================================================================================================================
def _a(pattern, count=1, kind="code", role="", laptop_record=False, hazard=None, package=False):
    return {"pattern": pattern, "count": count, "kind": kind, "role": role, "laptop_record": laptop_record,
            "hazard": hazard, "package": package}


M2B = f"{MARTIN}/results/m2b_mlp"
GROUP_A = [
    # ---- code ----------------------------------------------------------------------------------------------------
    _a(f"{MARTIN}/common.py", role="metric of record (common.test_nll / event_nll), Showers, FieldModel"),
    _a(f"{MARTIN}/src/sfm/*.py", 10, role="sfm package: paths, data, legacy, registry, metric, stats, train, "
       "refstream, gpulock, __init__"),
    _a(f"{MARTIN}/src/sfm/_philox_emul.c", role="CUDA Philox emulator source; its hash names cache/philox_emul_<sha16>.so "
       "(refstream.library_path)"),
    _a(f"{MARTIN}/src/sfm/models/*.py", 6, role="model factories (field, ensemble, table, paramnet, dist_factory)"),
    _a(f"{MARTIN}/src/sfm/families/*.py", 7, role="Track A distribution families (load_legacy_core, build_variant)"),
    _a(f"{MARTIN}/src/sfm/sr/*.py", 3, role="PySR modules (not moved; imported by tests/test_sr_*.py)"),
    *[_a(f"{MARTIN}/scripts/{s}.py", role=r) for s, r in (
        ("run_m2a", "Track A driver (phases templates ... stage1x; guards/stage2 need the missing run_m2a_report.py)"),
        ("run_m2a_lib", "Track A helpers (Spec, run_spec, registered_row, val_array, on_gpu)"),
        ("run_m2b_lib", "Track B helpers (registered_row, run_on_gpu)"),
        ("run_m2b_fit", "Track B fits from configs (s2_seed13-15, b0_seed1-2)"),
        ("run_m2b_ensembles", "Track B ensembles, subsets, scaling table"),
        ("run_m2b_report", "Track B report -> results/m2b_mlp/summary.json (no argparse)"),
        ("fit", "generic config runner (--dry-run builds only)"),
        ("fit_queue", "queue of configs (--dry-run lists only)"),
        ("build_cache", "cache build (no argparse: ANY invocation rebuilds both caches; use only into scratch dirs)"),
        ("parity", "Gate 0 parity (default mode and --reference-stream OPEN THE TEST SPLIT; never run before Gate 2c)"),
        ("mc_noise_val", "validation-only fold-noise study (no argparse; writes results/m0_parity/mc_noise_val.json)"),
        ("check_protected", "protected-file check against results/m0_parity/checksums.json"),
        ("make_transfer_package", "this script (list / verify)"))
      if s != "make_transfer_package"],
    _a(f"{MARTIN}/scripts/make_transfer_package.py", role="this script (list / verify)", package=True),
    *[_a(f"shape-form-proton/{s}.py", role="imported read-only by sfm.legacy (shapelab.build_final, ParticleShape)")
      for s in ("shapelab", "particle_shape", "proton_shape", "families", "protonlab")],
    _a(f"{MARTIN}/results/gates/gate0_checks/philox_emul.c", role="Gate 0 emulator; compared by "
       "tests/test_refstream.py::test_c_source_is_the_gate0_emulator"),
    _a(f"{MARTIN}/results/m1_audit/checks/c_side.py", role="source of the ported exit_depth / acceptances "
       "(dist_factory); not imported at run time"),
    _a("pyproject.toml", kind="env", role="dependencies (requires-python >= 3.11; no .python-version)"),
    _a("uv.lock", kind="env", role="locked versions (torch 2.14.0 PyPI wheel, +cu130 on the laptop; numpy 2.5.3)"),
    _a(".gitignore", kind="env", role="root ignore rules (*.parquet, *.csv, shape-form-proton/**/*.{pt,npz,npy,json,log})"),
    _a(f"{MARTIN}/.gitignore", kind="env", role="campaign ignore rules (cache/, results/**/*.{pt,npy,npz})"),
    _a(f"{MARTIN}/tests/conftest.py", kind="test", role="pytest setup"),
    *[_a(f"{MARTIN}/tests/test_{t}.py", kind="test", role="campaign test") for t in (
        "data", "dist_factory", "families_cellint", "families_components", "families_halo", "families_lat",
        "families_long", "metric", "models", "refstream", "registry", "sr_operators", "sr_search", "stats", "train")],
    _a(f"{MARTIN}/tests/test_transfer_package.py", kind="test", role="test of this script", package=True),
    # ---- configs -------------------------------------------------------------------------------------------------
    _a(f"{MARTIN}/configs/m2b/*.toml", 8, kind="config", role="Track B run files (s2_seed11-15, b0_seed0-2)"),
    _a(f"{MARTIN}/configs/examples/*.toml", 3, kind="config", role="config templates for AUDIT section 4.2 runs"),
    # ---- documents (read by people) ------------------------------------------------------------------------------
    *[_a(f"{MARTIN}/{d}", kind="doc", role=r) for d, r in (
        ("PLAN.md", "campaign plan (sections 13-16)"), ("README.md", "workspace README"),
        ("results/m1_audit/AUDIT.md", "pre-registered run lists (section 4.1 Track A, 4.2 Track B)"),
        ("results/m1_audit/SUMMARY.md", "Milestone 1 summary"), ("results/gates/gate0.md", "Gate 0 report"),
        ("results/m2a_dist/STATUS.md", "Track A status"), ("results/m2b_mlp/STATUS.md", "Track B status + resume commands"),
        ("transfer/STATUS.md", "transfer package status"), ("transfer/REVIEW_STATUS.md", "statistics review of the move"))],
    # ---- tracked laptop records (hazards on another GPU) ---------------------------------------------------------
    _a(REGISTRY_REL, kind="record", laptop_record=True, role="registry: 8 laptop rows (stream: laptop grid 144)",
       hazard="Read by registered_row (run_m2b_lib.py:51, run_m2a_lib.py:159; match by checkpoint SHA-256 only) and "
              "latest() (select_1se registry.py:521, run_m2b_ensembles._arr :162, run_m2b_report.py:42, "
              "run_m2a_lib.val_array :174). On the H200, re-running a registered run's script reuses the laptop row "
              "('already registered; not evaluated again'); until H200 rows are appended, every table/ensemble/1-SE "
              "step reads the laptop rows' val_nll_path: FileNotFoundError while the laptop arrays are quarantined "
              "(correct), silent laptop/H200 mixing if they were copied to the live path. evaluate() appends."),
    _a(f"{MARTIN}/results/test_access.log", kind="record", laptop_record=True,
       role="test-access log: the two laptop lines (Gate 0 parity, C1)",
       hazard="Appended by every load_sets(..., allow_test=True) (data.py:253-258). The line holds UTC time, script and "
              "reason but no host/GPU: on the H200 the file starts with the two laptop lines and a test access there "
              "appends a third line to the H200 copy; Gate 2c's 'exactly two lines' must be restated per machine and "
              "the two working copies' logs must be merged by hand."),
    _a(f"{MARTIN}/results/registry_refs/guard_spec.json", kind="record", laptop_record=True,
       role="guard spec (K=1024, bulk, validation fingerprint 731a1997...; no GPU/stream identity)",
       hazard="_bench_guard_arrays (registry.py:323-338) reuses results/registry_refs/*.npy whenever this spec matches "
              "and the four arrays exist: laptop arrays at the live path would be reused silently. With them "
              "quarantined, the first guarded evaluate() on the H200 recomputes them and rewrites this file with "
              "identical content (no git diff)."),
    _a(f"{M2B}/*/done.json", 8, kind="record", laptop_record=True,
       role="run done-markers of s1_seed11-15, s2_seed11-12, b0_seed0",
       hazard="Skip markers: run_m2b_fit.py:40, fit.py:43, fit_queue.py:68 and run_m2b_stage1.py:233 skip these "
              "runs. The laptop networks are kept (intended), but they can be "
              "re-scored only by a new script (registry.evaluate directly), never through the run scripts. "
              "run_m2b_report.py:56 reads them for training details."),
    _a(f"{M2B}/*/fit_done.json", 3, kind="record", laptop_record=True,
       role="sfm.train fit markers of b0_seed0, s2_seed11, s2_seed12 (config_hash, model_sha256)",
       hazard="train.fit (train.py:407-416) returns the stored model.pt instead of training whenever this file exists "
              "(after checking model.pt's SHA-256). Retraining b0_seed0 on the H200 (user decision, 25 Nov 2026) therefore "
              "needs another run name/out_dir; deleting done.json alone would silently reload the laptop network."),
    _a(f"{M2B}/*/resources.json", 8, kind="record", laptop_record=True, role="per-segment fit/eval seconds and memory",
       hazard="Read-modify-write by run_m2b_fit.py:79-85 only for a run being executed (none of these). Laptop timings."),
    _a(f"{M2B}/s1_seed*/stage1_monitor.json", 5, kind="record", laptop_record=True, role="stage-1 per-epoch monitors",
       hazard="Read by run_m2b_stage1.py:75-76 only when stage 1 resumes; done.json prevents that. Laptop record."),
    _a(f"{M2B}/summary.json", kind="record", laptop_record=True, role="Track B report table (laptop numbers)",
       hazard="Not read by code; rewritten (git diff) by run_m2b_report.py:133 on the H200. Until then it shows laptop "
              "numbers (other sample points)."),
    *[_a(f"{MARTIN}/results/m0_parity/{j}.json", kind="record", laptop_record=True, role=r, hazard=h) for j, r, h in (
        ("parity", "Gate 0 parity record (laptop, grid 144)",
         "Its existence makes parity.py refuse to run without --overwrite (parity.py:342): keep it that way."),
        ("c1_reference_stream", "C1 record (emulated grid 1056, laptop)", "Existence check for parity.py --reference-stream."),
        ("cache_build", "cache build record (does NOT hold the cache hashes)",
         "Rewritten by ANY invocation of build_cache.py (build_cache.py:64-66) unless SFM_RESULTS_DIR points elsewhere."),
        ("mc_noise_val", "validation fold-noise record (laptop)", "Rewritten by mc_noise_val.py (no argparse)."),
        ("checksums", "protected-file manifest (1997 files + Parquet/CSV size+mtime)",
         "check_protected.py verify on the H200 reports every protected file that was not copied as missing (exit 1)."),
        ("checksums_verify", "last protected-file verification (laptop)", "Not read by code."))],
    # ---- this package (untracked when `list` runs; committed together with the outputs) --------------------------
    *[_a(f"{TRANSFER_REL}/{n}", kind="package", package=True, role="output of make_transfer_package.py list")
      for n in OUTPUT_NAMES],
]


# ======================================================================================================================
# Group b: git-ignored, copied by hand. b1 -> live path; b2 -> quarantine root (comparison only).
# Every entry names the code that reads it. `count` = expected number of files of a glob pattern.
# `test_derived`: per-event or per-hit information about test events -> copied only after Gate 2c (P4).
# `contains_test_events`: not derived from test events but holds their raw hits (the Parquet table) -> also deferred.
# `test_note`: why a file whose name or content mentions the test split travels now / is not test-derived.
# ======================================================================================================================
def _b(pattern, group, role, readers, stream_dependent, stream, decision, regeneration, count=1, hazard=None,
       cross_check=None, opaque=False, test_derived=False, test_note=None, contains_test_events=False):
    return {"pattern": pattern, "count": count, "group": group, "role": role, "readers": readers,
            "stream_dependent": stream_dependent, "stream": stream, "decision": decision,
            "regeneration": regeneration, "hazard": hazard, "cross_check": cross_check, "opaque": opaque,
            "test_derived": test_derived, "test_note": test_note, "contains_test_events": contains_test_events}


COPY_LIVE = "copy to the same repo-relative path (rsync -a, keeps mtime); verify size + SHA-256"
COPY_Q = (f"copy ONLY into {QUARANTINE_REL}/<same path> (comparison partner); never to the live path; verify there")
AFTER_G2C = ("TEST-DERIVED: not copied now; copied ONLY AFTER Gate 2c has passed (files_b_test_derived_*.txt); "
             "before that, verify fails if it exists on the H200")
AFTER_G2C_RAW = ("DEFERRED (contains test events; not test-derived): not copied now; copied ONLY AFTER Gate 2c has "
                 "passed (files_b_test_derived_inputs.txt); before that, verify fails if it exists on the H200")
RUNS8 = ("s1_seed11", "s1_seed12", "s1_seed13", "s1_seed14", "s1_seed15", "s2_seed11", "s2_seed12", "b0_seed0")
GUARDED3 = ("b0_seed0", "s2_seed11", "s2_seed12")
NATIVE144 = "laptop native stream (RTX 4060 Laptop, launch grid 144), CUDA generator seed 12345"

GROUP_B1 = [
    _b("shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet", "b1",
       "raw hit table (Parquet copy of the CSV) of ALL events, test events included. No moved step reads it while "
       "both caches are copied",
       ["sfm/paths.py:41-43 parquet_path", "sfm/data.py:51-55 parquet_getter <- build_cache (data.py:214-217)",
        "sfm/data.py:327, :367 exactness checks (build_cache.py only)", "scripts/build_cache.py:32 (rebuild)",
        "scripts/parity.py:471-472 (raw-file check, default mode only)",
        "scripts/check_protected.py:84-86 (size + mtime against checksums.json stat_only)",
        "Gate 0 record scripts results/gates/gate0_checks/c05, c06, c11a, c12, c15 and the laptop geometry audit "
        "results/m2c_theta/geometry/explore/e1-e3 (not moved)"],
       False, "none (data)", AFTER_G2C_RAW + ". Then: " + COPY_LIVE + ". On the H200 it is needed by no moved step; "
       "the zero-training geometry arm gets the absolute shower origin as a small laptop-made file instead (group c)",
       {"command": "not regenerated (source data)", "laptop_time": None, "reproducibility": "n/a"},
       cross_check={"kind": "checksums_stat"}, contains_test_events=True,
       test_note="NOT test-derived, but it CONTAINS the raw hits of all events including the 14,956 test events "
                 "(the split is made at cache build, data.py:150-160): deferred to after Gate 2c (user decision of "
                 "25 Nov 2026); see policies.raw_table"),
    _b(f"{MARTIN}/cache/pion_s1_trainval.pt", "b1", "train/val cache (69,783 + 14,950 events)",
       ["sfm/data.py:292 load_sets (every step)", "sfm/registry.py:153 validation fingerprint, :344 _default_bench_models",
        "scripts/run_m2a_lib.py:111 data_cpu", "scripts/run_m2b_fit.py:46", "scripts/fit.py:50"],
       False, "none (deterministic CPU build from the Parquet file)",
       "COPY (do not rebuild); verify SHA-256 = cache/pion_s1_meta.json files.trainval.sha256",
       {"command": "uv run python -B shape-form-martin/scripts/build_cache.py (no argparse; rewrites BOTH caches, the "
                   "meta file and the tracked results/m0_parity/cache_build.json)",
        "laptop_time": "12.0 s build, 40.1 s with both exactness checks, 4.05 GB peak RSS (cache_build.json)",
        "reproducibility": "not guaranteed bit-identical on another CPU (float64 np.cos/np.sin of zb and R, "
                           "data.py:135-136, common.py:144; torch.log for lnp); only the byte copy carries Gate 0/C1"},
       cross_check={"kind": "meta", "key": "trainval"},
       test_note="train and val events only; its geometry constants come from all rows (data.py:82-86) and its "
                 "per-event 'index' is the position in the full event list (the complement locates the test "
                 "events, without hits or scores)"),
    _b(TEST_CACHE_REL, "b1", "TEST cache (14,956 events) - OPAQUE BYTE STREAM, hash and size only",
       ["sfm/data.py:287 load_sets(..., allow_test=True, reason=...) only: scripts/parity.py:318,368 and the future "
        "final_test.py (after Gate 2c)"],
       False, "none (deterministic CPU build from the Parquet file)",
       AFTER_G2C + ". COPY (do not rebuild); verify SHA-256 = cache/pion_s1_meta.json files.test.sha256 by a plain "
       "byte hash; stays unopened as data until the final evaluation after Gate 2c",
       {"command": "only as a side effect of build_cache.py (data.py:227), which also rewrites the copied files",
        "laptop_time": "(part of the 12.0 s build)",
        "reproducibility": "as the train/val cache; C1 validated these exact bytes"},
       cross_check={"kind": "meta", "key": "test"}, opaque=True, test_derived=True),
    _b(META_REL, "b1", "cache metadata (geometry, counts, per-file SHA-256 of both caches)",
       ["sfm/data.py:300 load_meta", "sfm/registry.py:144-145 validation_reference (trainval SHA-256 = fingerprint "
        "key), :161 counts"],
       False, "none", "COPY; never byte-reproducible (created_utc, Parquet mtime_ns), and it carries the reference "
       "hashes of both caches", {"command": "build_cache.py (see above)", "laptop_time": None,
                                 "reproducibility": "never byte-identical"},
       test_note="travels NOW although it mentions the test split: summary metadata only (counts.test, files.test "
                 "hash and size, splits.test aggregate statistics); needed by load_meta and validation_reference; "
                 "see policies.test_summary_metadata_now"),
    _b(f"{PION}/mlp_std.pt", "b1", "benchmark MLP checkpoint (fine-tuned 4x32)",
       ["sfm/legacy.py:45 load_mlp_std", "sfm/models/field.py:70 mlp_std_start (factories field_mlp_std, field_acc, "
        "residual_table, ensemble include_mlp_std)", "sfm/registry.py:345 _default_bench_models (guard arrays)",
        "scripts/run_m2b_ensembles.py:71 _member_shas"],
       False, "none (weights)", COPY_LIVE, {"command": "not regenerable (earlier study)", "laptop_time": None,
                                             "reproducibility": "n/a"}, cross_check={"kind": "checksums"}),
    _b(f"{PION}/new3_std_n3.pt", "b1", "benchmark component formula checkpoint (101 parameters)",
       ["shape-form-proton/shapelab.py:397 build_final <- sfm/legacy.py:56 load_component_formula, "
        "sfm/families/components.py:526 load_legacy_core (every Track A model via dist_factory.legacy_core :221)"],
       False, "none (weights)", COPY_LIVE, {"command": "not regenerable (earlier study)", "laptop_time": None,
                                             "reproducibility": "n/a"}, cross_check={"kind": "checksums"}),
    _b(f"{PION}/abar.pt", "b1", "stored acceptance template A-bar (128 x 32)",
       ["sfm/legacy.py:66 load_abar", "sfm/models/field.py:78-82 acceptance_template('stored') (also registry.py:454 "
        "guards)", "shape-form-proton/shapelab.py:396 build_final", "scripts/run_m2a.py:211 (templates phase)"],
       False, "none", COPY_LIVE, {"command": "not regenerable (earlier study)", "laptop_time": None,
                                  "reproducibility": "n/a"}, cross_check={"kind": "checksums"}),
    _b(f"{PION}/test_v3.json", "b1", "final-formula definition of the earlier study (degree, variant); stored "
       "RESULTS, not test data",
       ["shape-form-proton/shapelab.py:363 final_round (exists), :377 final_kwargs"],
       False, "none", COPY_LIVE, {"command": "not regenerable", "laptop_time": None, "reproducibility": "n/a"},
       cross_check={"kind": "checksums"},
       test_note="travels NOW although it holds test numbers: summary results only (test means, paired differences "
                 "with SE, gap-closed fractions, per-momentum-bin mean differences; no per-event values); the legacy "
                 "adapter reads the formula's structure from it; see policies.test_summary_metadata_now"),
    _b(f"{PION}/mlp_std.json", "b1", "stored validation mean of mlp_std (reference machine)",
       ["scripts/parity.py:91 stored_val_means"], True, "reference machine of the earlier study (grid 1056), scalar",
       COPY_LIVE + "; validation-only parity target on the H200", {"command": "not regenerable", "laptop_time": None,
                                                                   "reproducibility": "n/a"},
       cross_check={"kind": "checksums"},
       test_note="validation only: keys val (18.118084) and val_report_weights; no test content"),
    _b(f"{PION}/new3_std.json", "b1", "stored validation means of the formula rungs (reference machine)",
       ["scripts/parity.py:92 stored_val_means"], True, "reference machine of the earlier study (grid 1056), scalars",
       COPY_LIVE + "; validation-only parity target on the H200", {"command": "not regenerable", "laptop_time": None,
                                                                   "reproducibility": "n/a"},
       cross_check={"kind": "checksums"},
       test_note="validation only: per-rung params and validation means (rungs 1-6), n_star, chosen, diff_vs_best "
                 "(paired difference and SE on validation, shape-form-proton/scripts/particle_pipeline.py:121-126); "
                 "no test content"),
    _b(f"{PION}/test_v3_nll.npz", "b1", "stored per-event TEST arrays of the earlier study (not a laptop record)",
       ["scripts/parity.py:317 (C1), :407 (default mode)", "results/gates/gate0_checks/c08, c08b, c14, c16 (Gate 0 "
        "records, not moved)"], True,
       "reference machine of the earlier study (grid 1056), test split",
       AFTER_G2C + ". Then: " + COPY_LIVE + "; pairing partner for a labelled secondary test column and the "
       "abort-on-fail check of the final test if the H200 grid is 1056",
       {"command": "not regenerable", "laptop_time": None, "reproducibility": "n/a"},
       cross_check={"kind": "checksums"}, test_derived=True),
    _b(f"{MARTIN}/results/m0_parity/val_event_number.npy", "b1", "validation event numbers in pairing order",
       ["sfm/registry.py:142-143 validation_reference (every evaluate / check_validation)",
        "tests/test_registry.py real-data test"], False, "none (event numbers)",
       COPY_LIVE + "; written only by parity.py default mode (parity.py:414), which opens the test split",
       {"command": "no command exists (parity.py default mode opens test)", "laptop_time": None,
        "reproducibility": "bit-identical (= cache extras['val']['event_number'])"}),
    _b(f"{MARTIN}/results/m0_parity/test_event_number.npy", "b1", "test event numbers in test pairing order (order "
       "check of the final test)",
       ["scripts/parity.py:316 (C1 order check)", "future final_test.py", "results/gates/gate0_checks/c05, c08, c14, "
        "c16 (Gate 0 records, not moved)"], False, "none (event numbers)",
       AFTER_G2C + ". Then: " + COPY_LIVE, {"command": "no command exists (parity.py default mode opens test)",
                                           "laptop_time": None, "reproducibility": "bit-identical"},
       test_derived=True),
    _b(f"{MARTIN}/results/m0_parity/abar_front.pt", "b1", "acceptance template, front face only (A3f, B4f)",
       ["sfm/models/field.py:78-82 acceptance_template('front') (field_acc template='front', component "
        "template='front')", "scripts/run_m2a.py:213 (templates phase comparison)"], False,
       "none (deterministic mean_acceptance, no RNG; GPU float arithmetic)",
       COPY_LIVE + "; written only by parity.py default mode (parity.py:504), which opens the test split",
       {"command": "no command exists (parity.py default mode opens test)", "laptop_time": None,
        "reproducibility": "deterministic up to GPU float rounding"}),
    _b(f"{MARTIN}/results/m0_parity/abar_front_back.pt", "b1", "acceptance template, front + back (recomputed)",
       ["sfm/models/field.py:78-82 acceptance_template('front_back')"], False,
       "none (deterministic mean_acceptance, no RNG; GPU float arithmetic)",
       COPY_LIVE + "; written only by parity.py default mode (parity.py:505)",
       {"command": "no command exists (parity.py default mode opens test)", "laptop_time": None,
        "reproducibility": "deterministic up to GPU float rounding"}),
    *[_b(f"{M2B}/{r}/model.pt", "b1", f"checkpoint of the registered Track B network {r}",
         ["sfm/models/__init__.py:95 load_run (factories from_run :172, ensemble :165, residual_table :157)",
          "scripts/run_m2b_lib.py:51-67 registered_row (SHA-256)", "scripts/run_m2b_ensembles.py:69 _member_shas",
          "sfm/train.py:407-414 (SHA-256 against fit_done.json if fit() is called)"]
         + (["configs/m2b/s2_seed%s.toml (factory from_run): needed for the fine-tune" % r[-2:]]
            if r in ("s1_seed13", "s1_seed14", "s1_seed15") else []),
         False, "none (weights; trained on the laptop, identified by SHA-256)",
         COPY_LIVE + "; SHA-256 must equal checkpoint_sha256 of its registry row",
         {"command": "not regenerable (training is not bit-reproducible)", "laptop_time": None,
          "reproducibility": "n/a"}, cross_check={"kind": "registry_checkpoint"}) for r in RUNS8],
]

GROUP_B2 = [
    *[_b(f"{MARTIN}/results/m0_parity/{m}_val_nll.npy", "b2", f"Gate 0 benchmark pairing array ({m}, validation)",
         ["sfm/registry.py:92-94,410-414 evaluate (pairing partner of EVERY row)",
          "scripts/run_m2a.py:172 (A0 drift rule)" if m == "formula" else "scripts/run_m2b_ensembles.py:177 "
          "(scaling table)"],
         True, NATIVE144 + ", K=64; written OUTSIDE deterministic() (parity.py:412)", COPY_Q,
         {"command": "H200 version: no command exists yet (validation-only common.test_nll inside deterministic())",
          "laptop_time": "1.48 s (mlp) / 0.90 s (formula) on validation (parity.json)",
          "reproducibility": "machine-dependent (fold stream of the launch grid)"},
         hazard="DANGEROUS at the live path: every H200 registry row would pair against laptop points silently; "
                "absent, evaluate() fails loudly (FileNotFoundError)") for m in ("mlp", "formula")],
    *[_b(f"{MARTIN}/results/m0_parity/{m}_test_nll.npy", "b2", f"Gate 0 benchmark TEST array ({m}), laptop",
         ["no reader in moved code (written by parity.py:412; read by the Gate 0 record scripts "
          "results/gates/gate0_checks/c08, c08b, c14)"], True, NATIVE144 + ", K=64, test split",
         AFTER_G2C + ". Then: " + COPY_Q,
         {"command": "not regenerated (Gate 0 record)", "laptop_time": "1.37 s / 0.89 s (parity.json)",
          "reproducibility": "machine-dependent"}, test_derived=True) for m in ("mlp", "formula")],
    *[_b(f"{MARTIN}/results/m0_parity/{m}_{s}_nll_ref1056.npy", "b2",
         f"laptop emulation of the reference stream ({m}, {s})", ["no reader in code (written by parity.py:%s)"
                                                                 % ("314/466" if s == "val" else "324")],
         True, "emulated reference stream, launch grid 1056 (laptop CPU emulator + laptop GPU arithmetic)",
         (AFTER_G2C + ". Then: " if s == "test" else "") + COPY_Q
         + "; expected values of the H200's own common.test_nll if its grid is 1056 (tolerance 1e-4 per event)",
         {"command": "not regenerated (C1 / Gate 0 record)", "laptop_time": "C1 total 7.1 s",
          "reproducibility": "emulated stream is machine-independent; arithmetic agrees to ~3e-5 per event"},
         test_derived=(s == "test"))
      for s in ("val", "test") for m in ("mlp", "formula")],
    *[_b(f"{MARTIN}/results/registry_refs/{m}_val_{k}.npy", "b2", f"benchmark guard array ({m}, {k})",
         ["sfm/registry.py:323-338 _bench_guard_arrays (reused when guard_spec.json matches)"], True,
         NATIVE144 + (", K=1024 chunk 64" if k == "k1024" else ", bulk metric on the stratum zb >= 1221 mm")
         + ", inside deterministic()", COPY_Q,
         {"command": "automatic at the first registry.evaluate(..., guards=True) when absent",
          "laptop_time": "about 27 s for all four (b0_seed0 guarded evaluation 42.8 s vs 16.1 s)",
          "reproducibility": "machine-dependent"},
         hazard="DANGEROUS at the live path: guard_spec.json is tracked and has no GPU identity, so laptop arrays "
                "would be reused silently (shown by the trace)") for m in ("mlp", "formula")
      for k in ("k1024", "bulk_stratum")],
    *[_b(f"{M2B}/{r}/val_nll.npy", "b2", f"per-event validation NLL of the registered run {r}",
         ["sfm/registry.py:542 select_1se, :273-281 _save_array (re-scoring compares)",
          "scripts/run_m2b_ensembles.py:166 _arr", "scripts/run_m2b_report.py:46", "scripts/run_m2a_lib.py:179 val_array"],
         True, NATIVE144 + ", K=64, inside deterministic()", COPY_Q,
         {"command": "re-scoring: no command exists yet", "laptop_time": "1.5 s (no guards) / 16.1 s (guards) per run",
          "reproducibility": "machine-dependent"},
         hazard="At the live path the laptop rows would stay valid (their SHA-256 matches) and be mixed with H200 "
                "arrays; re-scoring would raise FileExistsError unless replace=True",
         cross_check={"kind": "registry_array"}) for r in RUNS8],
    *[_b(f"{M2B}/{r}/val_nll_k1024.npy", "b2", f"guard array K=1024 of {r}",
         ["no reader in code (written by registry.py:355)"], True, NATIVE144 + ", K=1024, inside deterministic()",
         COPY_Q, {"command": "re-scoring with guards: no command exists yet", "laptop_time": "(in 16.1 s)",
                  "reproducibility": "machine-dependent"}) for r in GUARDED3],
    *[_b(f"{M2B}/{r}/val_bulk_zb1221.npy", "b2", f"guard array bulk/stratum of {r}",
         ["scripts/run_m2b_report.py:87-90 (bulk_mean)", "written by registry.py:356"], True,
         NATIVE144 + ", bulk metric on zb >= 1221 mm", COPY_Q,
         {"command": "re-scoring with guards: no command exists yet", "laptop_time": "(in 16.1 s)",
          "reproducibility": "machine-dependent"}) for r in GUARDED3],
    *[_b(f"{M2B}/{r}/val_mass_monitors.npz", "b2", f"mass monitors of {r} (deterministic map functionals)",
         ["no reader in code (written by registry.py:358)"], False,
         "none (no fold points; laptop GPU float arithmetic)", COPY_Q + "; free functional check of the copied "
         "checkpoint after re-scoring (should agree to rounding)",
         {"command": "re-scoring with guards: no command exists yet", "laptop_time": "(in 16.1 s)",
          "reproducibility": "deterministic up to GPU float rounding"}) for r in GUARDED3],
    _b(f"{MARTIN}/results/gates/gate0_checks/val_grid1056_*.npy", "b2",
       "Gate 0 validation arrays on the emulated reference stream (13 models)",
       ["tests/test_refstream.py:113-115 (val_grid1056_mlp_std.npy, tolerance 1e-4; skipped if absent)",
        "results/gates/gate0_checks/c08_parity_stats.py, c08b_pair_tail.py, c14_stats_review.py"],
       True, "emulated reference stream, launch grid 1056 (computed outside deterministic())", COPY_Q,
       {"command": "not regenerated (Gate 0 record, c04_confirm_reference_stream.py)", "laptop_time": None,
        "reproducibility": "emulated stream machine-independent; arithmetic to ~3e-5"}, count=13),
    _b(f"{MARTIN}/results/gates/gate0_checks/val_grid144_*.npy", "b2",
       "Gate 0 validation arrays on the emulated laptop stream (13 models)",
       ["results/gates/gate0_checks/c08_parity_stats.py, c08b_pair_tail.py, c14_stats_review.py"], True,
       "emulated launch grid 144 (= laptop native stream)", COPY_Q,
       {"command": "not regenerated (Gate 0 record)", "laptop_time": None, "reproducibility": "as above"}, count=13),
]

# ======================================================================================================================
# Group c: regenerated on the H200 (not copied)
# ======================================================================================================================
def _c(path, command, laptop_time, reproducibility, notes=""):
    return {"path": path, "command": command, "laptop_time": laptop_time, "reproducibility": reproducibility,
            "notes": notes}


NOCMD = "no command exists yet"
GROUP_C = [
    _c(".venv/", "uv sync --python 3.12   (then confirm torch.version.cuda and torch.cuda.is_available())",
       "not measured", "machine-dependent install (uv.lock pins torch 2.14.0; laptop wheel +cu130, needs a CUDA-13 "
       "capable driver)"),
    _c(f"{MARTIN}/cache/philox_emul_622ce7d283034712.so", "automatic at the first emulator use (sfm.refstream._load: "
       "gcc -O2 -fopenmp -shared -fPIC -ffp-contract=off); name = hash of the C source + flags", "seconds (not measured)",
       "binary is machine-dependent; its output bits are machine-independent (integer Philox)", "needs gcc with OpenMP"),
    _c(f"{MARTIN}/cache/pion_s1_val_fingerprint.json", "automatic in sfm.registry.validation_reference "
       "(registry.py:152-160) when absent or keyed to another trainval SHA-256", "seconds",
       "bit-identical with the copied cache: fingerprint must equal 731a199763cc31c498986574083e95c9c6c61e09d83f69b294"
       "cee13cf07858a7 (tracked in results/registry_refs/guard_spec.json)", "a free check of the copied train/val cache"),
    _c(f"{MARTIN}/cache/gpu.lock", "created by the first sfm.gpulock.gpu_lock (or SFM_GPU_LOCK)", "-", "-"),
    _c(f"{MARTIN}/results/m0_parity/mlp_val_nll.npy (H200 version)", NOCMD + "; needs a validation-only script: "
       "load_sets(('train','val')), legacy.load_mlp_std(), common.test_nll inside sfm.metric.deterministic(), "
       "np.save; never opens test; records GPU name and refstream.local_grid()", "1.48 s (parity.json)",
       "machine-dependent (stream); bit-reproducible on one machine inside deterministic()"),
    _c(f"{MARTIN}/results/m0_parity/formula_val_nll.npy (H200 version)", NOCMD + " (same script, "
       "legacy.load_component_formula(train))", "0.90 s (parity.json)", "machine-dependent (stream)"),
    _c(f"{MARTIN}/results/registry_refs/{{mlp,formula}}_val_{{k1024,bulk_stratum}}.npy (H200 versions)",
       "automatic at the first registry.evaluate(..., guards=True) when the files are absent", "about 27 s",
       "machine-dependent", "create once, by one process, before anything else is registered"),
    _c(f"{M2B}/<8 registered runs>/{{val_nll.npy,val_nll_k1024.npy,val_bulk_zb1221.npy,val_mass_monitors.npz}} + 8 "
       "H200 registry rows", NOCMD + "; needs a re-scoring script: check model.pt SHA-256 against the laptop row, "
       "models.load_run, registry.evaluate(guards=True for b0_seed0/s2_seed11/s2_seed12, False for s1_*) "
       "(run_m2b_fit/run_m2b_stage1 skip on done.json and registered_row reuses laptop rows)",
       "5 x 1.5 s + 3 x 16.1 s + ~27 s once = about 1.5 min", "machine-dependent; mass monitors to rounding"),
    _c(f"{MARTIN}/results/m2a_dist/templates/{{back_corrected.pt,full_box.pt,templates_all.npz,templates.json,"
       "train_theta.npy}", "uv run python -B shape-form-martin/scripts/run_m2a.py templates", "never run",
       "train_theta.npy bit-identical (float64 copy of cache theta); templates GPU float32 sums: identical up to "
       "rounding", "read by A3c, A3x, A9, A9r and B4x"),
    _c(f"{MARTIN}/results/m2a_dist/{{plr.json,decisions.json,gpu_time.jsonl}} and <run>/{{model.pt,checkpoint.pt,"
       "fit_done.json,done.json,init.json,val_nll.npy,...}", "uv run python -B shape-form-martin/scripts/run_m2a.py "
       "parity plr stage0 stage1p stage1x   (BLOCKED on a grid-1056 GPU by FORMULA_OF_RECORD, run_m2a.py:43/260/266)",
       "estimate only: AUDIT 4.1 about 4 GPU-hours; _est_seconds defaults 0.055 s/step (poly), 0.07 s/step x K/3 "
       "(mixture)", "machine-dependent"),
    _c("Track A report / guards / Stage 2 (A11, A11k, A11n)", NOCMD + "; phases guards/stage2 import the missing "
       "scripts/run_m2a_report.py (ImportError)", "-", "machine-dependent"),
    _c(f"{M2B}/{{s2_seed13,s2_seed14,s2_seed15,b0_seed1,b0_seed2}}/*", "uv run python -B "
       "shape-form-martin/scripts/run_m2b_fit.py shape-form-martin/configs/m2b/s2_seed13.toml "
       "shape-form-martin/configs/m2b/s2_seed14.toml shape-form-martin/configs/m2b/s2_seed15.toml "
       "shape-form-martin/configs/m2b/b0_seed1.toml shape-form-martin/configs/m2b/b0_seed2.toml   "
       "(precondition: H200 pairing arrays)", "238 s fit + 16.1 s guarded evaluation each, about 21 min for five",
       "machine-dependent (training fold points; not bit-reproducible)"),
    _c(f"{M2B}/{{E5_s2,E6_s2_mlpstd,E2_s2_seeds11-12,E3_s2_seeds11-13,E4_s2_seeds11-14,E5_s1}}/*, "
       "ensemble_subsets/*.npy, ensemble_subsets/timing.json, ensemble_scaling.json",
       "uv run python -B shape-form-martin/scripts/run_m2b_ensembles.py   (--table-only: table from saved arrays; "
       "precondition: the 8 runs re-scored)", "never run on the laptop (no measurement)", "machine-dependent"),
    _c(f"{M2B}/summary.json (tracked, rewritten)", "uv run python -B shape-form-martin/scripts/run_m2b_report.py "
       "(no argparse; CPU only)", "seconds (not measured)", "deterministic given the arrays"),
    _c("remaining Track B runs (AUDIT 4.2)", "configs + scripts/fit.py or fit_queue.py work for B2a/b/c, B5f "
       "(field_mlp_std), B3s (field), B4f/B4s (field_acc front/stored); " + NOCMD + " for B4x (field_acc rejects "
       "full_box), B6a/b/c (no factory), B1d/B1f (distillation), B-ceil (train_fraction)",
       "estimate only: about 2 GPU-hours (PLAN 15)", "machine-dependent"),
    _c("theta track Theta-1..Theta-3", NOCMD + " (own harness forward(lnp, aux); Theta-1b needs the absolute origin "
       "in the cache; user decision of 25 Nov 2026: zero-training geometry arm only, after the geometry audit)", "-", "machine-dependent"),
    _c(f"{MARTIN}/results/m2c_theta/geometry/<absolute shower origin per event> (file name not fixed; NOT YET "
       "EXISTING INPUT, made on the LAPTOP and shipped, not regenerated on the H200)",
       NOCMD + ": a laptop script that reads ONLY the train/val rows of the Parquet file (row selection by the event "
       "numbers of cache/pion_s1_trainval.pt, as the geometry audit does) and writes, keyed by event_number, the "
       "absolute origin (x0, y0, z0) of the shower frame = energy-weighted centroid of the lowest-z hit layer over "
       "ALL rows of the event, late hits included (results/m2c_theta/geometry/STATUS.md section 1.1); no test row",
       "not measured", "deterministic (float64 sums over the Parquet rows); machine-independent up to summation order",
       "Needed by the zero-training geometry arm (absolute origin of the VALIDATION events; AUDIT section 4.3 Theta1b: "
       "'needs the absolute origin in the cache'). The cache stores zb, theta and phi but not the origin "
       "(sfm/data.py:132-135 computes z0 only to derive zb), and the Parquet file is absent on the H200 until after "
       "Gate 2c (policies.raw_table), so the origin must travel as a small file (about 15,000 validation events x "
       "3 float64 + int64 event numbers, well under 1 MB): commit it, or add it to files_b_inputs.txt with a reader "
       "in code (none exists yet). The instrumented-volume constants of the geometry audit (layers, lattice, stepped "
       "octagon outline; STATUS.md section 1.3) are numbers that travel as text or code."),
    _c(f"{MARTIN}/results/final/LOCK.json and the final test arrays", NOCMD + " (final_test.py and LOCK.json "
       "missing; after Gate 2c)", "-", "machine-dependent"),
]

# ======================================================================================================================
# Considered and excluded
# ======================================================================================================================
EXCLUDED = [
    {"path": f"{M2B}/{{b0_seed0,s2_seed11,s2_seed12}}/checkpoint.pt", "reason": "read by train.fit only when "
     "fit_done.json is absent (train.py:407,436); fit_done.json is tracked and present. Carries laptop CUDA RNG and "
     "fold-generator states (train.py:233-235, 427): must never be resumed on another machine"},
    {"path": f"{M2B}/s1_seed*/stage1_fit.pt", "reason": "read by run_m2b_stage1.py only for a seed without done.json "
     "(run_m2b_stage1.py:233); all five have tracked done.json"},
    {"path": f"{M2B}/*/fit.log, {M2B}/logs/*.log, {M2B}/s2_seed13/fit.log", "reason": "logs; train.fit appends to "
     "fit.log; s2_seed13 holds one start line and restarts from step 0"},
    {"path": f"{MARTIN}/results/gates/gate0_checks/{{philox_emul.so,events_table.npz,protected/}}",
     "reason": "Gate 0 artefacts; no moved step reads them"},
    {"path": f"{MARTIN}/results/m1_audit/**/*.npz (10 files)", "reason": "laptop-stream Milestone 1 records, read "
     "only by the M1 audit scripts (not moved); dist_factory ports c_side.py and reads no npz (traced); M1 "
     "conclusions carry over through the tracked AUDIT.md"},
    {"path": f"{MARTIN}/results/m2c_theta/geometry/explore/train_columns.npz", "reason": "geometry audit (stays on "
     "the laptop)"},
    {"path": f"{PION}/<all other files>", "reason": "never opened by any traced entry point"},
    {"path": "shape-form-original/data/*.csv", "reason": "never read (Parquet copy used); refused by this script"},
    {"path": ".venv/julia_env, ~/.julia", "reason": "PySR searches do not move"},
    {"path": "**/__pycache__/", "reason": "bytecode (all scripts run with -B)"},
]

# ======================================================================================================================
# Machine-specific constants and assumptions (file, line, text that must be on that line; checked by `list`)
# ======================================================================================================================
def _k(file, line, must, what, consequence):
    return {"file": file, "line": line, "must_contain": must, "what": what, "consequence_on_h200": consequence}


S, T = f"{MARTIN}/scripts", f"{MARTIN}/tests"
SRC = f"{MARTIN}/src/sfm"
CONSTANTS = [
    _k(f"{MARTIN}/common.py", 313, "manual_seed(12345)", "common.test_nll draws its fold points from the CUDA "
       "Philox generator; the stream depends on the launch grid SMs x (maxThreadsPerSM // 256)",
       "H200: 132 x 8 = 1056 = REFERENCE_GRID (prediction; read refstream.local_grid()). Its validation numbers would "
       "be the earlier study's 18.118084 / 18.120407, not the laptop's 18.117857 / 18.120061. A MIG slice differs"),
    _k(f"{MARTIN}/common.py", 64, 'torch.device("cuda" if torch.cuda.is_available() else "cpu")',
       "silent CPU fallback", "if the CUDA build/driver fails, every script runs on the CPU (another fold stream, very "
       "slow) and run_on_gpu/on_gpu skip the memory check (free = inf, run_m2b_lib.py:121, run_m2a_lib.py:128)"),
    _k(f"{S}/run_m2a.py", 43, "FORMULA_OF_RECORD = 18.120061", "laptop formula validation mean as the Track A parity "
       "target (checked at :260 to 1e-5, stop at :266)", "with grid 1056 the value is ~18.120407 (diff ~3.5e-4): the "
       "parity phase registers A0_init (registry write), records passed:false in decisions.json and raises "
       "SystemExit; Track A cannot proceed until the criterion is restated machine-independently"),
    _k(f"{S}/run_m2a.py", 172, 'np.load(L.paths.M0_DIR / "formula_val_nll.npy")', "A0 drift rule (|A0 - formula| > "
       "2e-4 -> stop, :313) pairs with the benchmark array", "laptop array at the live path: false STOP (+3.5e-4 "
       "stream difference); absent: FileNotFoundError; needs the H200 array"),
    _k(f"{S}/run_m2a.py", 44, "BUDGET_S = 4.0 * 3600.0", "4 GPU-hour budget; with _est_seconds defaults 0.07 / "
       "0.055 s per step (:371, :374) and 180 s per guard evaluation (:381)", "the pre-registered drop order depends "
       "on machine speed (and on concurrency, since gpu_time.jsonl sums wall seconds): a fast H200 drops fewer runs"),
    _k(f"{SRC}/registry.py", 94, 'm0 / f"{b}_val_nll.npy"', "benchmark pairing arrays = 'this machine's stream'; "
       "rows record only device 'cuda' and the torch version (:446)", "laptop arrays at the live path: every H200 "
       "row pairs silently against laptop points (fake +-2.3e-4 vs MLP, 3.5e-4 vs formula); absent: loud error"),
    _k(f"{SRC}/registry.py", 328, "ok = spec_file.exists()", "guard spec has no GPU/stream identity (_guard_spec "
       ":296)", "laptop registry_refs/*.npy at the live path are reused silently; absent: recomputed"),
    _k(f"{S}/run_m2b_lib.py", 62, 'r.get("checkpoint_sha256") != want', "registered_row matches by checkpoint "
       "SHA-256 only (also run_m2a_lib.py:166)", "re-running a registered run's script reuses the laptop row; "
       "re-scoring is impossible through the existing scripts"),
    _k(f"{S}/run_m2b_ensembles.py", 177, 'np.load(lib.paths.M0_DIR / "mlp_val_nll.npy")', "scaling table pairs with "
       "the benchmark array and the latest rows' arrays (:166; also run_m2b_report.py:42-49, registry.py:542)",
       "before re-scoring: laptop arrays of seeds 11/12 mixed with H200 arrays (if copied) or FileNotFoundError"),
    _k(f"{SRC}/refstream.py", 51, "LOCAL_GRID = 144", "launch grid of the laptop GPU", "tests that assume it fail "
       "(see tests_expected_to_fail_on_h200)"),
    _k(f"{T}/test_refstream.py", 52, "refstream.local_grid() == refstream.LOCAL_GRID", "asserts the local grid is "
       "144", "fails on the H200 (1056)"),
    _k(f"{T}/test_refstream.py", 76, "(a == b).float().mean()) < 0.05", "assumes the local stream differs from the "
       "grid-1056 stream", "fails on a grid-1056 GPU (the streams are identical)"),
    _k(f"{T}/test_refstream.py", 113, 'GATE0 / "val_grid1056_mlp_std.npy"', "uses a Gate 0 laptop record if present "
       "at its live path", "with the record quarantined this sub-check is skipped silently (the test still checks "
       "the stored mean to 4e-6)"),
    _k(f"{S}/audit_lib.py", 48, "RECORD = {", "laptop validation numbers of record", "M1 audit re-runs on the H200 "
       "would compare against laptop numbers (not a moved step)"),
    _k(f"{SRC}/registry.py", 440, '"checkpoint": str(ckpt)', "absolute laptop paths in rows (also train.py:661 "
       "out_dir, config_path in model.pt meta)", "informational only; no loader uses them"),
    _k(f"{SRC}/train.py", 235, "torch.cuda.get_rng_state_all()", "checkpoint.pt stores CUDA RNG states; the fold "
       "generator lives on the device (:427); resume checks only config_hash (:438)", "resuming a laptop "
       "checkpoint.pt on the H200 continues on another stream (hybrid fit); no partial laptop checkpoint is needed"),
    _k(f"{SRC}/refstream.py", 64, "def local_grid(device: int = 0)", "always cuda:0 (with common.DEVICE)",
       "on a multi-GPU node every job runs on GPU 0 (cache/gpu.lock serialises them); MIG changes the grid"),
    _k(f"{SRC}/refstream.py", 89, 'subprocess.run(["gcc"', "gcc with -fopenmp must exist (CFLAGS :55)",
       "without it the first emulator use fails (test_refstream, reference-stream checks)"),
    _k(f"{S}/run_m2b_lib.py", 110, "min_free_mb: float = 3000.0", "free-memory threshold tuned for an 8 GB GPU "
       "(run_m2a_lib.py:121: 2500)", "harmless on 141 GB"),
    _k(f"{T}/test_families_components.py", 532, "assert mem < 1500 and ms < 150", "cost budget written for the "
       "RTX 4060", "passes, or skips on a busy GPU (not verified on an H200)"),
    _k(f"{S}/check_protected.py", 49, '"m0_parity" / "checksums.json"', "protected-file manifest incl. Parquet size "
       "and mtime", "verify on the H200 reports every protected file that was not copied as missing (exit 1); keep "
       "the Parquet mtime (rsync -a)"),
    _k(f"{SRC}/data.py", 135, "np.cos(theta)", "float64 numpy transcendental functions in the cache build (also "
       "common.py:144 rotation)", "numpy uses AVX-512 SVML kernels where available (the laptop i7-14650HX has no "
       "AVX-512): a rebuild on the H200 host is not guaranteed bit-identical -> copy the caches"),
    _k(f"{SRC}/metric.py", 79, "def deterministic():", "bit-reproducibility per machine only", "H200 arrays are "
       "reproducible on the H200 and differ from the laptop's"),
    _k(f"{S}/parity.py", 412, 'np.save(OUT / f"{name}_{s}_nll.npy"', "laptop pairing arrays were computed outside "
       "deterministic(); registry arrays inside (registry.py:418)", "the H200 recomputation should use "
       "deterministic() (differs from an outside run by up to ~2.7e-5 per event)"),
    _k(f"{SRC}/data.py", 258, "{script}\\t{reason}", "test-access log line has no host or GPU", "lines written on "
       "the two machines cannot be told apart; Gate 2c's 'exactly two lines' must be restated per machine"),
    _k(f"{SRC}/gpulock.py", 32, 'paths.CACHE_DIR / "gpu.lock"', "one flock file per working copy (or SFM_GPU_LOCK)",
       "two checkouts on one node do not share the lock unless SFM_GPU_LOCK points to one file; flock may not work "
       "on a network filesystem"),
    _k(f"{SRC}/paths.py", 26, "def _env_path", "SFM_DATA_DIR / SFM_RESULTS_DIR / SFM_CACHE_DIR overrides (:31-33)",
       "the manifest assumes the defaults: setting any of them moves where the code looks for the b1 files"),
    _k("pyproject.toml", 5, 'requires-python = ">=3.11"', "no .python-version; the laptop venv is CPython 3.12",
       "uv may choose another interpreter on the H200: use uv sync --python 3.12"),
    _k("uv.lock", 2450, 'version = "2.14.0"', "torch 2.14.0 from PyPI (laptop: 2.14.0+cu130, CUDA 13.0)",
       "the H200 driver must support CUDA 13.0, otherwise torch.cuda.is_available() is False (silent CPU fallback)"),
]

TESTS_ON_H200 = [
    {"test": "tests/test_refstream.py::test_local_grid_constant_matches_this_gpu", "expected": "FAIL",
     "why": "asserts local_grid() == LOCAL_GRID == 144; the H200 grid is 1056 (any GPU but the RTX 4060 Laptop fails)"},
    {"test": "tests/test_refstream.py::test_other_grids_differ_only_beyond_their_thread_count", "expected": "FAIL "
     "if the H200 grid is 1056", "why": "requires the local stream to differ from the grid-1056 stream"},
    {"test": "tests/test_refstream.py (library, bitwise emulation, local/reference stream tests)", "expected": "PASS "
     "if gcc with OpenMP exists, else FAIL", "why": "the .so is compiled on first use; "
     "test_emulation_reproduces_torch_rand_bitwise is the check that the H200 torch build has the same Philox layout"},
    {"test": "tests/test_families_components.py::test_forward_backward_cost_at_batch_256", "expected": "PASS or SKIP",
     "why": "budget written for the RTX 4060; skipped when the GPU is busy"},
    {"test": "tests/test_data.py, test_metric.py, test_registry.py real-data tests", "expected": "PASS with the b1 "
     "files; skip (cache absent) or FAIL (test_registry needs results/m0_parity/val_event_number.npy) without them",
     "why": "they read the copied cache, meta and event numbers"},
    {"test": "tests/test_dist_factory.py template test", "expected": "SKIP until run_m2a.py templates", "why":
     "HAVE_TEMPLATES"},
    {"test": "all campaign tests, before Gate 2c (the eight deferred files absent)", "expected": "unaffected",
     "why": "no test reads test_v3_nll.npz, test_event_number.npy, the test cache or the Parquet file; "
            "tests/test_data.py::test_test_split_is_locked raises PermissionError / ValueError before any log write "
            "or open (data.py:283, :285); tests/test_data.py::test_paths_layout only checks the suffix of "
            "paths.parquet_path() (:17, no stat). Laptop check with the eight made invisible (audit hook on open, "
            "stat wrapper) and under strace: collection of all 458 tests and tests/test_data.py (5 passed) touched "
            "none of them; no test imports pyarrow.parquet"},
]

# ======================================================================================================================
# What the ABSENCE of the test-derived files before Gate 2c changes (file, line, text on that line; checked by `list`)
# ======================================================================================================================
def _x(file, line, must, what, consequence):
    return {"file": file, "line": line, "must_contain": must, "what": what, "consequence_before_gate2c": consequence}


G0 = f"{MARTIN}/results/gates/gate0_checks"
ABSENCE = [
    _x(f"{SRC}/data.py", 286, "_log_test_access(str(reason).strip())", "load_sets(('test',), allow_test=True, "
       "reason=...) appends the test-access log line BEFORE it opens the test cache (:287, the only reader of "
       "pion_s1_test.pt)", "an allowed test load fails with FileNotFoundError at :287 after the log line has been "
       "written: on the H200 such a line before Gate 2c marks an attempted access (no data read); investigate it. "
       "The locked calls (no allow_test, :283; empty reason, :285) raise before logging or opening"),
    _x(f"{S}/parity.py", 316, 'np.load(OUT / "test_event_number.npy")', "C1 (--reference-stream; refused without "
       "--overwrite at :300 because c1_reference_stream.json is tracked)", "with --overwrite: FileNotFoundError here, "
       "before load_sets(test) at :318 (no log line), but only after :314 has written "
       "{mlp,formula}_val_nll_ref1056.npy at their live paths"),
    _x(f"{S}/parity.py", 368, "allow_test=True, reason=TEST_REASON", "Gate 0 default mode (refused without "
       "--overwrite at :342 because parity.json is tracked)", "with --overwrite: a test-access log line, then "
       "FileNotFoundError (data.py:286-287), before test_v3_nll.npz (:407) is reached. parity.py must not run on the "
       "H200 before Gate 2c in any case"),
    _x(f"{S}/check_protected.py", 152, "fail = bool(a or r or c", "verify compares shape-form-proton/ and "
       "shape-form-original/ with checksums.json (1997 hashed files; 185 tracked, 1812 git-ignored, 7 of them in the "
       "package; stat_only: the CSV and the Parquet file, size + mtime, :84-86)", "already FAILS on the H200 (1805 "
       "ignored protected files are never copied, and the CSV in stat_only is missing); before Gate 2c "
       "test_v3_nll.npz is one more 'removed' hashed entry (1806) and the Parquet file a second 'removed' stat_only "
       "entry. Exit code 1 either way; it also rewrites the tracked results/m0_parity/checksums_verify.json (:155)"),
    _x(f"{SRC}/data.py", 55, "pf = pq.ParquetFile(str(path or paths.parquet_path()))", "parquet_getter, the first "
       "I/O of build_cache (data.py:216-217; also the exactness checks :327, :367), i.e. of scripts/build_cache.py:32",
       "FileNotFoundError ('Failed to open local file', shown on the laptop with a missing scratch path) before "
       "anything is written (the cache files at :226-227, the meta file at :246, cache_build.json at "
       "build_cache.py:64-66 all come later): the caches cannot be rebuilt on the H200 before Gate 2c - intended, "
       "since a rebuild would also write a test cache"),
    _x(f"{S}/parity.py", 472, "pq.ParquetFile(str(paths.parquet_path()))", "raw-file check of the Gate 0 default "
       "mode", "not reached before Gate 2c: the default mode stops earlier at :368 (test cache absent) and is refused "
       "without --overwrite at :342"),
    _x(f"{SRC}/data.py", 132, 'z0 = first["z"] - first["z_trans"]', "the cache build computes the global z of the "
       "shower origin only to derive zb (:135); the cache stores zb, theta and phi but no absolute origin",
       "the zero-training geometry arm (absolute shower origin of the validation events) cannot get the origin on the "
       "H200 before Gate 2c: it must be computed on the laptop from train/val rows and shipped as a small file keyed "
       "by event number (group c, not yet existing)"),
    _x(f"{G0}/c06_cache_fidelity.py", 28, "pq.ParquetFile(str(paths.parquet_path()))", "Gate 0 record scripts and "
       "the laptop geometry audit that read the Parquet file (also gate0_checks/c05_split_integrity.py:19, "
       "c11a_hits_1001.py:14, c12_units_lowp.py:55, c15_geometry_train_only.py:11; "
       "results/m2c_theta/geometry/explore/e1.py:15, e2.py:11, e3.py:12)", "FileNotFoundError on the H200 before "
       "Gate 2c (not moved steps)"),
    _x(f"{T}/test_data.py", 17, 'paths.parquet_path().suffix == ".parquet"', "test_paths_layout",
       "passes: a path computation only (paths.py:41-43), no stat or open; run with the eight hidden: 5 passed"),
    _x(f"{G0}/c08_parity_stats.py", 117, "test_v3_nll.npz", "Gate 0 record scripts that read the test-derived files "
       "(also c05_split_integrity.py:102, c08_parity_stats.py:110/118, c08b_pair_tail.py:15-16, "
       "c14_stats_review.py:119-121, c16_test_order_fingerprint.py:15/19)", "FileNotFoundError on the H200 (not moved "
       "steps; the four laptop test arrays they read would be quarantined in any case)"),
    _x(f"{SRC}/data.py", 168, '"test": d / f"{CACHE_STEM}_test.pt"', "cache_files() returns the test path without "
       "opening it", "no effect: no code branches on cache_files()['test'].exists() (grep); load_meta still reports "
       "files.test from the copied meta file"),
    _x(f"{T}/test_data.py", 34, 'data.load_sets(("test",), device="cpu")', "test_test_split_is_locked",
       "passes: PermissionError at data.py:283 (and ValueError at :285) before any log write or open; run on the "
       "laptop with the eight hidden: 5 passed"),
]
ABSENCE_UNAFFECTED = (
    "Unaffected (read-only run on the laptop CPU, 25 Nov 2026, with the eight deferred files - the seven test-derived "
    "files and the raw Parquet table - made invisible: an audit hook raised FileNotFoundError on any open and a stat "
    "wrapper on any stat of them; writes under the repo and *.csv opens were refused; the whole run was also traced "
    "with strace, which sees C-level opens such as pyarrow's: no file syscall named any of the eight or a .csv, and "
    "pyarrow.parquet was never imported by the code under test): imports of all sfm modules; "
    "data.load_sets(('train','val')) from the copied cache and load_meta; "
    "legacy.load_mlp_std, load_component_formula (shapelab.build_final, reads test_v3.json, which travels now), "
    "load_abar, families.load_legacy_core; registry.validation_reference, check_validation, read_registry, "
    "select_1se, the benchmark arrays and _bench_guard_arrays (reuse path); acceptance templates stored/front/"
    "front_back; the emulator library; Track A spec lists and builds with skip_init=True; Track B factories "
    "field_mlp_std, from_run, ensemble, field_acc, residual_table, field; load_run of the 8 networks; fit.py "
    "--dry-run (s2_seed13, b0_seed1); fit_queue.py --dry-run; run_m2b_fit.py config start-up; import and --help of "
    "run_m2a.py, run_m2b_fit.py, run_m2b_ensembles.py, run_m2b_stage1.py, import of run_m2b_report.py; collection "
    "of all 458 campaign tests and a run of tests/test_data.py. Result: zero opens or stats of any of the eight.")


# ======================================================================================================================
# Hashing (the ONLY place where group-b files are opened)
# ======================================================================================================================
def _refuse_csv(path) -> None:
    if str(path).lower().endswith(".csv"):
        raise RefusedFile(f"refused: {path} is a .csv (never listed, opened or hashed by this script)")


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes, read in 4 MiB chunks in binary mode.

    This is the only routine that opens a group-(b) file. The test cache is handled here as an opaque byte stream:
    its bytes are hashed, nothing is interpreted (no unpickling, no torch). Any *.csv is refused before opening."""
    _refuse_csv(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


_NPY_DTYPES = {"<f4": "float32", "<f8": "float64", "<i8": "int64", "<i4": "int32", "|b1": "bool", "<f2": "float16"}


def npy_array_sha256(path) -> str:
    """``sfm.registry._sha256_array`` of a .npy file without numpy: sha256(f"{dtype}|{shape}|" + raw bytes)."""
    _refuse_csv(path)
    if str(path).endswith(TEST_CACHE_REL) or not str(path).endswith(".npy"):
        raise RefusedFile(f"npy_array_sha256 only reads .npy arrays, not {path}")
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path}: not a .npy file")
        major, _minor = f.read(2)
        hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
        raw = f.read()
    if header.get("fortran_order") or header["descr"] not in _NPY_DTYPES:
        raise ValueError(f"{path}: unsupported .npy header {header}")
    shape = tuple(header["shape"])
    return hashlib.sha256(f"{_NPY_DTYPES[header['descr']]}|{shape}|".encode() + raw).hexdigest()


# ======================================================================================================================
# git helpers
# ======================================================================================================================
def _git(repo: Path, *args, check=False) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise PackageError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p


def git_tracked(repo: Path, rel: str) -> bool:
    """`git ls-files --error-unmatch -- <path>` (exit 0 = tracked)."""
    return _git(repo, "ls-files", "--error-unmatch", "--", rel).returncode == 0


def git_ignore_rule(repo: Path, rel: str):
    """`git check-ignore -v -- <path>`: the rule '<source>:<line>:<pattern>' that ignores it, or None."""
    p = _git(repo, "check-ignore", "-v", "--", rel)
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip().split("\t")[0]
    if p.returncode not in (0, 1):
        raise PackageError(f"git check-ignore failed for {rel}: {p.stderr.strip()}")
    return None


# ======================================================================================================================
# table expansion (glob patterns only with an expected count; never a blind walk)
# ======================================================================================================================
def expand(repo: Path, pattern: str, count: int) -> list:
    _refuse_csv(pattern)
    if any(ch in pattern for ch in "*?["):
        found = sorted(p.relative_to(repo).as_posix() for p in repo.glob(pattern) if p.is_file())
    else:
        found = [pattern] if (repo / pattern).is_file() else []
    for rel in found:
        _refuse_csv(rel)
    if len(found) != count:
        raise PackageError(f"{pattern}: expected {count} file(s), found {len(found)}: {found[:5]}")
    return found


def _load_json(p: Path):
    return json.loads(p.read_text())


class CrossChecks:
    """Recorded reference values: cache meta (b1, ignored), checksums.json and registry.jsonl (tracked)."""

    def __init__(self, repo: Path):
        self.repo = repo
        self._meta = self._ck = self._reg = None

    def meta(self):
        if self._meta is None:
            self._meta = _load_json(self.repo / META_REL)
        return self._meta

    def checksums(self):
        if self._ck is None:
            self._ck = _load_json(self.repo / CHECKSUMS_REL)
        return self._ck

    def registry_latest(self):
        if self._reg is None:
            rows = [json.loads(l) for l in (self.repo / REGISTRY_REL).read_text().splitlines() if l.strip()]
            self._reg = {}
            for r in rows:
                self._reg[(r["stage"], r["name"])] = r
        return self._reg

    def check(self, spec: dict, rel: str, size: int, sha: str) -> dict:
        kind = spec["kind"]
        if kind == "meta":
            rec = self.meta()["files"][spec["key"]]
            exp = {"sha256": rec["sha256"], "bytes": rec["bytes"]}
            ok = exp["sha256"] == sha and exp["bytes"] == size
            src = f"{META_REL} files.{spec['key']}"
        elif kind == "checksums":
            rec = self.checksums()["files"].get(rel)
            exp = rec and {"sha256": rec["sha256"], "bytes": rec["size"]}
            ok = bool(rec) and exp["sha256"] == sha and exp["bytes"] == size
            src = f"{CHECKSUMS_REL} files[{rel}]"
        elif kind == "checksums_stat":
            rec = self.checksums()["stat_only"].get(rel)
            st = (self.repo / rel).stat()
            exp = rec and {"bytes": rec["size"], "mtime_ns": rec["mtime_ns"]}
            ok = bool(rec) and exp["bytes"] == size and exp["mtime_ns"] == st.st_mtime_ns
            src = f"{CHECKSUMS_REL} stat_only[{rel}] (size + mtime; no hash recorded)"
        elif kind == "registry_checkpoint":
            name = Path(rel).parent.name
            row = self.registry_latest().get(("m2b_mlp", name))
            exp = row and {"sha256": row["checkpoint_sha256"]}
            ok = bool(row) and exp["sha256"] == sha
            src = f"{REGISTRY_REL} m2b_mlp/{name} checkpoint_sha256"
        elif kind == "registry_array":
            name = Path(rel).parent.name
            row = self.registry_latest().get(("m2b_mlp", name))
            exp = row and {"val_nll_sha256": row["val_nll_sha256"]}
            got = npy_array_sha256(self.repo / rel)
            ok = bool(row) and exp["val_nll_sha256"] == got and row["val_nll_path"] == f"m2b_mlp/{name}/val_nll.npy"
            src = f"{REGISTRY_REL} m2b_mlp/{name} val_nll_sha256 (array hash, sfm.registry._sha256_array)"
        else:
            raise PackageError(f"unknown cross-check kind {kind}")
        return {"source": src, "expected": exp, "equal": bool(ok)}


def _hazard_list() -> list:
    """One entry per group-a table row with a hazard (pattern, number of files, what it makes the code do)."""
    return [{"pattern": e["pattern"], "files": e["count"], "hazard": e["hazard"]} for e in GROUP_A if e.get("hazard")]


# ======================================================================================================================
# list
# ======================================================================================================================
def build_manifest(repo: Path, log=print) -> dict:
    head = _git(repo, "rev-parse", "HEAD", check=True).stdout.strip()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip())
    problems = []

    # ---- group a ----
    a_files = []
    for e in GROUP_A:
        try:
            paths_ = expand(repo, e["pattern"], e["count"]) if not e["package"] else [e["pattern"]]
        except PackageError as err:
            problems.append(str(err))
            continue
        for rel in paths_:
            p = repo / rel
            tracked = git_tracked(repo, rel)
            item = {"path": rel, "kind": e["kind"], "role": e["role"], "laptop_record": e["laptop_record"],
                    "hazard": e["hazard"], "package": e["package"],
                    "bytes": p.stat().st_size if p.is_file() and rel.split("/")[-1] not in OUTPUT_NAMES else None,
                    "git": {"tracked": tracked}}
            if e["package"] and not tracked:
                rule = git_ignore_rule(repo, rel)
                if rule:
                    problems.append(f"{rel}: package file is git-ignored ({rule})")
                item["git"]["status"] = "untracked: part of this package, to be committed with the outputs"
                if rel.split("/")[-1] not in OUTPUT_NAMES and not p.is_file():
                    problems.append(f"{rel}: package file missing")
            elif not tracked:
                rule = git_ignore_rule(repo, rel)
                problems.append(f"{rel}: listed in group a but not tracked ({'ignored by ' + rule if rule else 'untracked and NOT ignored'})")
            a_files.append(item)

    # ---- groups b1 / b2 ----
    xc = CrossChecks(repo)
    b = {"b1": [], "b2": []}
    for e in GROUP_B1 + GROUP_B2:
        try:
            paths_ = expand(repo, e["pattern"], e["count"])
        except PackageError as err:
            problems.append(str(err))
            continue
        if e.get("test_derived") and e.get("test_note"):
            problems.append(f"{e['pattern']}: test_derived and test_note are exclusive")
        if e.get("test_derived") and e.get("contains_test_events"):
            problems.append(f"{e['pattern']}: test_derived and contains_test_events are exclusive (a test-derived "
                            "file is deferred by that flag alone)")
        if e.get("contains_test_events") and e["group"] != "b1":
            problems.append(f"{e['pattern']}: contains_test_events is only defined for group b1 inputs")
        for rel in paths_:
            name = rel.split("/")[-1]
            if rel == TEST_CACHE_REL and not e.get("test_derived"):
                problems.append(f"{rel}: the test cache must be test_derived (copied only after Gate 2c)")
            if _TEST_WORD.search(name.lower()) and not e.get("test_derived") and not e.get("test_note"):
                problems.append(f"{rel}: name mentions 'test' but the entry is neither test_derived nor explained by "
                                "a test_note")
            p = repo / rel
            tracked = git_tracked(repo, rel)
            rule = git_ignore_rule(repo, rel)
            if tracked:
                problems.append(f"{rel}: tracked by git, cannot be in group {e['group']}")
            if not tracked and rule is None:
                problems.append(f"{rel}: untracked and NOT ignored (neither group a nor b)")
            size = p.stat().st_size
            log(f"  hashing {rel} ({size} B){' [opaque byte stream]' if e['opaque'] else ''}")
            sha = sha256_file(p)
            td = bool(e.get("test_derived"))
            lst = list_for(e["group"], is_deferred(e))
            item = {"path": rel, "group": e["group"], "bytes": size, "sha256": sha, "role": e["role"],
                    "readers": e["readers"], "git": {"tracked": tracked, "ignored": rule is not None, "rule": rule},
                    "stream_dependent": e["stream_dependent"], "stream": e["stream"], "decision": e["decision"],
                    "test_derived": td, "contains_test_events": bool(e.get("contains_test_events")),
                    "transfer_phase": lst["phase"], "list": lst["file"],
                    "destination": rel if e["group"] == "b1" else f"{QUARANTINE_REL}/{rel}",
                    "regeneration": e["regeneration"], "hazard": e["hazard"],
                    "opaque_byte_stream": bool(e["opaque"]) or rel == TEST_CACHE_REL,
                    "test_note": e.get("test_note")}
            if e["cross_check"]:
                item["cross_check"] = xc.check(e["cross_check"], rel, size, sha)
                if not item["cross_check"]["equal"]:
                    problems.append(f"{rel}: differs from {item['cross_check']['source']}")
            b[e["group"]].append(item)
    seen = [x["path"] for g in b.values() for x in g]
    if len(seen) != len(set(seen)):
        problems.append("a path appears twice in groups b1/b2")

    # ---- group c ----
    c_items = []
    for e in GROUP_C:
        it = dict(e)
        p = e["path"]
        it["exists_now"] = (not any(ch in p for ch in "{}<> ")) and (repo / p.rstrip("/")).exists()
        c_items.append(it)

    # ---- constants and absence consequences: the cited line must still say what the table claims ----
    def cited(table, what):
        out = []
        for k in table:
            f = repo / k["file"]
            try:
                line = f.read_text().splitlines()[k["line"] - 1]
                ok = k["must_contain"] in line
            except (OSError, IndexError):
                line, ok = None, False
            if not ok:
                problems.append(f"{what} {k['file']}:{k['line']}: line does not contain {k['must_contain']!r}")
            out.append({**k, "line_text": line.strip() if line else None, "line_checked": ok})
        return out

    consts = cited(CONSTANTS, "constant")
    absence = cited(ABSENCE, "absence consequence")

    if problems:
        raise PackageError("list refused; nothing written:\n  - " + "\n  - ".join(problems))

    def tot(items):
        return {"files": len(items), "bytes": sum(i["bytes"] or 0 for i in items)}

    def by_phase(items):
        return {ph: tot([i for i in items if i["transfer_phase"] == ph]) for ph in PHASES}

    all_b = b["b1"] + b["b2"]
    lists = {}
    for l in LISTS:
        items = [x for x in b[l["group"]] if (x["transfer_phase"] == "after_gate2c") == l["deferred"]]
        lists[l["key"]] = {"file": f"{TRANSFER_REL}/{l['file']}", "group": l["group"],
                           "deferred": l["deferred"], "transfer_phase": l["phase"], "holds": l["holds"],
                           "destination_root": _dest_root(l), "totals": tot(items),
                           "test_derived_files": sum(1 for x in items if x["test_derived"]),
                           "contains_test_events_files": sum(1 for x in items if x["contains_test_events"]),
                           "rsync": rsync_command(l), "check_on_h200": check_command(l)}

    def brief(x):
        return {"path": x["path"], "group": x["group"], "list": x["list"], "destination": x["destination"],
                "bytes": x["bytes"], "test_derived": x["test_derived"],
                "contains_test_events": x["contains_test_events"]}
    test_derived = [brief(x) for x in all_b if x["test_derived"]]
    deferred = [brief(x) for x in all_b if x["transfer_phase"] == "after_gate2c"]

    return {
        "schema": SCHEMA,
        "title": "H200 transfer package (PLAN.md section 16)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": f"{MARTIN}/scripts/make_transfer_package.py list",
        "git": {"commit": head, "branch": branch, "tracked_tree_dirty": dirty,
                "note": "verify requires this commit to be an ancestor of (or equal to) the H200 checkout's HEAD"},
        "policies": {
            "test_derived": TEST_DERIVED_POLICY,
            "raw_table": RAW_TABLE_POLICY,
            "test_derived_files": test_derived,
            "deferred_files": deferred,
            "test_summary_metadata_now": TEST_SUMMARY_NOW,
            "test_events_in_raw_parquet": TEST_EVENTS_IN_PARQUET,
            "test_cache": TEST_CACHE_POLICY, "csv": CSV_POLICY, "quarantine_root": QUARANTINE_REL,
            "quarantine_rule": "group b2 files are copied ONLY under the quarantine root (repo-relative path preserved); "
                               "a stream-dependent b2 file found at its live path is a verify FAILURE",
            "cache_decision": {
                "decision": "COPY, do NOT rebuild: the train/val cache and pion_s1_meta.json now, the test cache only "
                            "after Gate 2c (test-derived); verify each by SHA-256 against the meta file (and this "
                            "manifest). Before Gate 2c a rebuild on the H200 is impossible anyway: the Parquet source "
                            "is deferred too (policies.raw_table)",
                "facts": [
                    "scripts/build_cache.py has no argument parser: any invocation (even --help) rebuilds",
                    "sfm.data.build_cache always writes BOTH pion_s1_trainval.pt and pion_s1_test.pt (data.py:226-227) "
                    "and the meta file (:246); build_cache.py then rewrites the tracked "
                    "results/m0_parity/cache_build.json (:64-66)",
                    "the meta file is never byte-identical (created_utc, Parquet mtime_ns)",
                    "the tensors are not guaranteed bit-identical on another CPU: float64 np.cos/np.sin "
                    "(data.py:135-136, common.py:144) may use AVX-512 SVML kernels on the H200 host (the laptop CPU "
                    "has no AVX-512); torch CPU log for lnp likewise",
                    "the code would not notice a different rebuild: validation_reference re-keys the fingerprint by "
                    "the new trainval SHA-256 and _bench_guard_arrays recomputes silently",
                    "Gate 0 (train/val bit-identical to the reference loader) and C1 (exact per-event test parity) "
                    "validated exactly the laptop bytes; only a byte copy carries both over; the test file stays "
                    "unopened, so a byte hash is the right check",
                    "optional cross-check without touching the live files - on the H200 ONLY AFTER Gate 2c, when the "
                    "Parquet file has been copied, because build_cache also writes a test cache (data.py:227): "
                    "SFM_CACHE_DIR=<scratch>/c SFM_RESULTS_DIR=<scratch>/r uv run python -B "
                    "shape-form-martin/scripts/build_cache.py, then compare the scratch SHA-256 values (equal = "
                    "bit-identical rebuild); delete the scratch test file"]},
        },
        "transfer_commands": {
            **{l["key"]: rsync_command(l) for l in LISTS},
            "phases": {ph: [l["key"] for l in LISTS if l["phase"] == ph] for ph in PHASES},
            "tar_alternative": f"tar -C <repo> -cf inputs.tar -T {TRANSFER_REL}/files_b_inputs.txt",
            "verify_on_h200_before_gate2c": f"python3 -B {MARTIN}/scripts/make_transfer_package.py verify   "
                                            "(from <repo>; fails if any of the eight deferred files - seven "
                                            "test-derived + the raw Parquet table - is on the machine)",
            "verify_on_h200_after_gate2c": f"python3 -B {MARTIN}/scripts/make_transfer_package.py verify "
                                           "--after-gate2c   (from <repo>, after the two after-Gate-2c lists were "
                                           "copied)",
            "verify_on_laptop": f"python3 -B {MARTIN}/scripts/make_transfer_package.py verify --source-machine",
        },
        "lists": lists,
        "totals_by_phase": by_phase(all_b),
        "groups": {
            "a": {"description": "tracked in the pushed branch; arrives with git pull (sizes are a snapshot at "
                                 "generation time; verify checks presence and tracking only)",
                  "files": a_files, "totals": tot(a_files),
                  "laptop_record_hazards": _hazard_list()},
            "b1": {"description": "git-ignored inputs; copy to the same repo-relative path (the test-derived ones and "
                                  "the raw Parquet table only after Gate 2c)", "files": b["b1"], "totals": tot(b["b1"]),
                   "totals_by_phase": by_phase(b["b1"])},
            "b2": {"description": f"git-ignored laptop records; copy ONLY under {QUARANTINE_REL}/ for comparison "
                                  "(test-derived ones only after Gate 2c)", "files": b["b2"], "totals": tot(b["b2"]),
                   "totals_by_phase": by_phase(b["b2"])},
            "c": {"description": "regenerated on the H200 (not copied)", "files": c_items,
                  "totals": {"entries": len(c_items)}},
        },
        "excluded": EXCLUDED,
        "machine_constants": consts,
        "absence_before_gate2c": {"consequences": absence, "unaffected": ABSENCE_UNAFFECTED},
        "tests_expected_to_fail_on_h200": TESTS_ON_H200,
    }


def _list_items(man: dict, l: dict) -> list:
    return [x for x in man["groups"][l["group"]]["files"] if (x["transfer_phase"] == "after_gate2c") == l["deferred"]]


def write_outputs(repo: Path, out_dir: Path, man: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for l in LISTS:
        (out_dir / l["file"]).write_text("".join(f"{x['path']}\n" for x in _list_items(man, l)))
    body = [
        f"# SHA-256 of the H200 transfer package (group b), generated at commit {man['git']['commit']} "
        f"({man['generated_utc']}) by make_transfer_package.py list.",
        "# Paths are repo-relative. Four sections; each is checked relative to its own DESTINATION ROOT on the H200,",
        "# and each header gives its list, phase, contents and the one-liner to run from <repo>/ on the H200.",
        "# Phase 'now': copy and check now. Phase 'AFTER GATE 2c': the DEFERRED files (the seven test-derived files and",
        "# the raw Parquet table, which contains the test events), copied to the H200 only after Gate 2c has passed;",
        "# before that, their absence is required (make_transfer_package.py verify).",
        f"# On the laptop (all four lists at their live source paths), from <repo>/: sha256sum -c {TRANSFER_REL}/SHA256SUMS",
        f"# {TEST_CACHE_REL} is hashed as an opaque byte stream; it stays unopened until the final evaluation "
        "after Gate 2c.",
        "# No *.csv is listed.",
    ]
    for i, l in enumerate(LISTS, 1):
        items = _list_items(man, l)
        phase = "now" if l["phase"] == "now" else "AFTER GATE 2c (deferred; do not copy before)"
        body += [
            "#",
            f"# ---- section {i}/{len(LISTS)}: {l['key']}  ({len(items)} files, {sum(x['bytes'] for x in items)} bytes, "
            f"{l['file']})",
            f"#   phase: {phase}",
            f"#   holds: {l['holds']}",
            f"#   destination root: {_dest_root(l)}",
            f"#   copy:  {rsync_command(l)}",
            f"#   check: {check_command(l)}",
            f"# BEGIN {l['key']}",
            *[f"{x['sha256']}  {x['path']}" for x in items],
            f"# END {l['key']}",
        ]
    (out_dir / "SHA256SUMS").write_text("\n".join(body) + "\n")
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")


def cmd_list(args) -> int:
    repo = Path(args.repo).resolve()
    out_dir = Path(args.out).resolve() if args.out else repo / TRANSFER_REL
    print(f"make_transfer_package list: repo {repo}")
    try:
        man = build_manifest(repo)
    except (PackageError, RefusedFile) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    write_outputs(repo, out_dir, man)
    g = man["groups"]
    print(f"commit {man['git']['commit']} (tracked tree dirty: {man['git']['tracked_tree_dirty']})")
    for k in ("a", "b1", "b2"):
        t = g[k]["totals"]
        extra = ""
        if k != "a":
            ph = g[k]["totals_by_phase"]
            extra = "   (" + ", ".join(f"{PHASE_LABEL[p]}: {ph[p]['files']} files, {ph[p]['bytes']} bytes"
                                       for p in PHASES) + ")"
        print(f"group {k:2}: {t['files']:4d} files, {t['bytes']:>12d} bytes{extra}")
    print(f"group c : {g['c']['totals']['entries']} entries (regenerated on the H200)")
    for l in LISTS:
        t = man["lists"][l["key"]]["totals"]
        print(f"list {l['file']:34} {l['group']}  {PHASE_LABEL[l['phase']]:13}  {t['files']:3d} files, "
              f"{t['bytes']:>12d} bytes  -> {_dest_root(l)}")
    for p in PHASES:
        t = man["totals_by_phase"][p]
        print(f"phase {PHASE_LABEL[p]:13}: {t['files']:3d} files, {t['bytes']:>12d} bytes")
    d = man["policies"]["deferred_files"]
    n_td, n_raw = sum(x["test_derived"] for x in d), sum(x["contains_test_events"] for x in d)
    print(f"deferred to after Gate 2c: {len(d)} files = {n_td} test-derived + {n_raw} raw table(s) with test events")
    print(f"wrote {', '.join(str(out_dir / n) for n in OUTPUT_NAMES)}")
    print(TEST_CACHE_POLICY)
    return 0


# ======================================================================================================================
# verify
# ======================================================================================================================
class Report:
    def __init__(self):
        self.lines, self.failures, self.warnings, self.checks = [], [], [], 0

    def ok(self, tag, msg):
        self.checks += 1
        self.lines.append(("OK", tag, msg))

    def fail(self, tag, msg):
        self.checks += 1
        self.failures.append(f"[{tag}] {msg}")
        self.lines.append(("FAIL", tag, msg))

    def warn(self, tag, msg):
        self.warnings.append(f"[{tag}] {msg}")
        self.lines.append(("WARN", tag, msg))

    def info(self, tag, msg):
        self.lines.append(("INFO", tag, msg))


def _check_file(rep: Report, tag: str, p: Path, rel: str, exp_bytes: int, exp_sha: str, skip_large, opaque: bool):
    _refuse_csv(rel)
    if not p.is_file():
        rep.fail(tag, f"missing: {rel}")
        return
    size = p.stat().st_size
    if size != exp_bytes:
        rep.fail(tag, f"size {size} != {exp_bytes}: {rel}")
        return
    if skip_large is not None and size > skip_large:
        rep.warn(tag, f"size OK, SHA-256 NOT checked (--skip-large-hash {skip_large}): {rel}")
        return
    got = sha256_file(p)                     # opaque byte read; the only open of the test cache
    if got != exp_sha:
        rep.fail(tag, f"SHA-256 mismatch {got[:16]}... != {exp_sha[:16]}...: {rel}")
    else:
        rep.ok(tag, f"{size:>11d} B  sha256 {got[:16]}...  {rel}" + ("  [opaque byte hash]" if opaque else "")
               + (f"  [SHA-256 of {size} B computed once]" if size > LARGE_HASH_NOTE else ""))


def find_by_name(repo: Path, names, suffixes=()) -> list:
    """Every file under the working tree whose name is in `names` or ends with one of `suffixes`: [(Path, size)].
    A cheap walk by name: directories named .git are skipped, symlinked directories are not followed, and any *.csv
    name is skipped before anything else (never stat'ed). Only matches are stat'ed; nothing is opened."""
    names, suffixes, out, stack = set(names), tuple(s.lower() for s in suffixes), [], [str(repo)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                if e.name.lower().endswith(".csv"):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name != ".git":
                            stack.append(e.path)
                    elif (e.name in names or (suffixes and e.name.lower().endswith(suffixes))) and e.is_file():
                        out.append((Path(e.path), e.stat().st_size))
                except OSError:
                    continue
    return sorted(out)


LARGE_HASH_NOTE = 1 << 27            # files above 128 MiB: say in the output that their SHA-256 was computed once


def _sha_once(p: Path, hashes: dict) -> str:
    """SHA-256 of a file, computed at most once per verify run (the raw table is 2.6 GB)."""
    key = os.path.realpath(p)
    if key not in hashes:
        hashes[key] = sha256_file(p)
    return hashes[key]


def _identity(p: Path, exp_bytes: int, exp_sha: str, skip_large, hashes=None):
    """Compare a present file with the manifest entry: size first; the SHA-256 (plain byte hash, also for the test
    cache) only if the size matches, and at most once per file. Returns (size_equal, sha_equal or None)."""
    _refuse_csv(p)
    size = p.stat().st_size
    if size != exp_bytes:
        return False, None
    if skip_large is not None and size > skip_large:
        return True, None
    return True, _sha_once(p, {} if hashes is None else hashes) == exp_sha


def _identity_text(size_eq, sha_eq, size=None) -> str:
    if not size_eq:
        return "different size from the laptop file"
    if sha_eq is None:
        return "same size as the laptop file, SHA-256 not checked (--skip-large-hash)"
    once = f"; SHA-256 of {size} B computed once" if size and size > LARGE_HASH_NOTE else ""
    return ("identical to the laptop file (size + SHA-256" + once + ")") if sha_eq else \
        ("same size, different SHA-256" + once)


def check_deferred_absent(rep: Report, repo: Path, quarantine_rel: str, deferred: list, skip_large,
                          hashes: dict) -> list:
    """Before Gate 2c: FAIL for every deferred file found on this machine. Returns the offending paths.
    - test-derived: at its live path or under the quarantine root (any content), or elsewhere in the working tree
      with the same name and size, confirmed by SHA-256 (same name and size, another hash: warning);
    - contains_test_events (the raw Parquet table): at its live path (any content), or elsewhere with the same name
      and size, confirmed by one SHA-256 (another hash: warning); any other file with its suffix (.parquet) - same
      name with another size, or another name - is information only (for example a train/val-only extract)."""
    offending = []
    raw_suffixes = {Path(x["path"]).suffix for x in deferred if x.get("contains_test_events") and Path(x["path"]).suffix}
    found = find_by_name(repo, {Path(x["path"]).name for x in deferred}, raw_suffixes)
    claimed = set()                                  # paths reported against a deferred entry (no duplicate info)
    qroot = repo / quarantine_rel
    for x in deferred:
        rel, name = x["path"], Path(x["path"]).name
        _refuse_csv(rel)
        raw = bool(x.get("contains_test_events")) and not x.get("test_derived")
        label = "RAW TABLE (contains test events)" if raw else "TEST-DERIVED FILE"
        fixed = {repo / rel: "live path"} if raw else {repo / rel: "live path", qroot / rel: "quarantine root"}
        hits = []
        for p, where in fixed.items():
            if p.is_file():
                size_eq, sha_eq = _identity(p, x["bytes"], x["sha256"], skip_large, hashes)
                hits.append(p)
                claimed.add(p)
                rep.fail("deferred-absent", f"{label} PRESENT BEFORE GATE 2c at its {where}: "
                                            f"{p.relative_to(repo).as_posix()} "
                                            f"({_identity_text(size_eq, sha_eq, x['bytes'])})")
        for p, size in found:
            if p.name != name or p in fixed:
                continue
            claimed.add(p)
            prel = p.relative_to(repo).as_posix()
            if size != x["bytes"]:
                what = ("a Parquet file of another size (for example a train/val-only extract; not the raw table)"
                        if raw else "not that file")
                rep.info("deferred-absent", f"same name as {rel} but different size, {size} B ({what}): {prel}")
                continue
            _, sha_eq = _identity(p, x["bytes"], x["sha256"], skip_large, hashes)
            once = f"; SHA-256 of {size} B computed once" if size > LARGE_HASH_NOTE and sha_eq is not None else ""
            if sha_eq is False:
                rep.warn("deferred-absent", f"same name and size as {rel} but a different SHA-256{once} (not the "
                                            f"laptop file; inspect it - a rebuilt test cache, a test array made here "
                                            f"or another copy of raw data with test events is test data too): {prel}")
                continue
            hits.append(p)
            rep.fail("deferred-absent", f"{label} PRESENT BEFORE GATE 2c elsewhere in the tree (same name and size, "
                                        f"{'same SHA-256' + once if sha_eq else 'SHA-256 not checked: treated as present'}"
                                        f"; copy of {rel}): {prel}")
        if hits:
            offending += [p.relative_to(repo).as_posix() for p in hits]
        else:
            where = "live path, same name and size elsewhere" if raw else \
                "live path, quarantine root, same name and size elsewhere"
            rep.ok("deferred-absent", f"absent ({where}): {rel}")
    for p, size in found:                             # other files with the raw table's suffix: information only
        if p not in claimed and p.suffix in raw_suffixes and p.relative_to(repo).parts[0] != ".venv":
            # (.venv holds the test fixtures of pyarrow; same-name copies there are still checked above)
            rep.info("deferred-absent", f"other {p.suffix} file, {size} B (not the raw table by name; information "
                                        f"only): {p.relative_to(repo).as_posix()}")
    return offending


def _deferred_count_text(n_td: int, n_raw: int) -> str:
    raw = "" if not n_raw else (" + the raw Parquet table" if n_raw == 1 else f" + {n_raw} raw tables")
    return f"{n_td + n_raw} checked: {n_td} test-derived{raw}"


def verify(repo: Path, manifest_path: Path, quarantine_rel: str, source_machine: bool, skip_large,
           after_gate2c: bool = False) -> dict:
    man = json.loads(manifest_path.read_text())
    rep = Report()
    b_all = man["groups"]["b1"]["files"] + man["groups"]["b2"]["files"]
    for x in b_all:
        _refuse_csv(x["path"])

    def deferred_entry(x):
        return x.get("transfer_phase") == "after_gate2c" or bool(x.get("test_derived"))
    deferred = [x for x in b_all if deferred_entry(x)]
    n_td = sum(1 for x in deferred if x.get("test_derived"))
    n_raw = len(deferred) - n_td
    hashes = {}
    check_td_here = source_machine or after_gate2c       # deferred files are expected on this machine
    commit = man["git"]["commit"]
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        rep.fail("git", f"the checkout does not contain the manifest's commit {commit}")
    elif _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        rep.fail("git", f"manifest commit {commit[:12]} is not an ancestor of HEAD {head[:12]}")
    else:
        rep.ok("git", f"checkout contains manifest commit {commit[:12]} (HEAD {head[:12]})")

    tracked = set(_git(repo, "ls-files").stdout.splitlines())
    a = man["groups"]["a"]["files"]
    missing = [x["path"] for x in a if not (repo / x["path"]).is_file()]
    untracked = [x["path"] for x in a if x["path"] not in tracked and (repo / x["path"]).is_file()]
    for m in missing:
        rep.fail("a", f"missing tracked file: {m}")
    for u in untracked:
        pkg = next(x for x in a if x["path"] == u)["package"]
        if pkg and source_machine:
            rep.warn("a", f"package file not committed yet: {u}")
        else:
            rep.fail("a", f"present but not tracked by git (group a files must arrive with git): {u}")
    if not missing:
        rep.ok("a", f"{len(a)} group-a files present ({len(a) - len(untracked)} tracked)")

    for x in man["groups"]["b1"]["files"]:
        if deferred_entry(x) and not check_td_here:
            continue                                      # before Gate 2c: absence is checked below
        _check_file(rep, "b1-deferred" if deferred_entry(x) else "b1", repo / x["path"], x["path"], x["bytes"],
                    x["sha256"], skip_large, x.get("opaque_byte_stream", False))
        exp_mtime = ((x.get("cross_check") or {}).get("expected") or {}).get("mtime_ns")
        if exp_mtime is not None and (repo / x["path"]).is_file() and (repo / x["path"]).stat().st_mtime_ns != exp_mtime:
            rep.warn("b1", f"mtime differs from {CHECKSUMS_REL} (copy with rsync -a; check_protected.py would report "
                           f"the file as modified): {x['path']}")

    qroot = repo / quarantine_rel
    if not source_machine:
        rule = git_ignore_rule(repo, f"{quarantine_rel}/probe")
        if rule:
            rep.ok("b2", f"quarantine root {quarantine_rel}/ is git-ignored ({rule})")
        else:
            rep.fail("b2", f"quarantine root {quarantine_rel}/ is NOT git-ignored: laptop records could be committed")
    for x in man["groups"]["b2"]["files"]:
        rel = x["path"]
        live = repo / rel
        tag = "b2-deferred" if deferred_entry(x) else "b2"
        if deferred_entry(x) and not check_td_here:
            continue                                      # before Gate 2c: absence is checked below
        if source_machine:
            _check_file(rep, tag, live, rel, x["bytes"], x["sha256"], skip_large, False)
            continue
        _check_file(rep, tag, qroot / rel, f"{quarantine_rel}/{rel}", x["bytes"], x["sha256"], skip_large, False)
        if live.is_file():
            same = live.stat().st_size == x["bytes"] and sha256_file(live) == x["sha256"]
            mv = f"mkdir -p {quarantine_rel}/{Path(rel).parent.as_posix()} && mv {rel} {quarantine_rel}/{rel}"
            if same and x["stream_dependent"]:
                rep.fail("b2-live", f"LAPTOP RECORD AT ITS LIVE PATH (stream: {x['stream']}): {rel}\n"
                                    f"        quarantine it with (not executed): {mv}")
            elif same:
                rep.warn("b2-live", f"laptop record (not stream-dependent) at its live path: {rel}; "
                                    f"suggested (not executed): {mv}")
            else:
                rep.info("b2-live", f"live path holds a different file (not the laptop record; e.g. produced on this "
                                    f"machine): {rel}")

    # ---- deferred files: seven test-derived (P4) + the raw Parquet table (contains test events) ----
    offending = []
    if deferred and not check_td_here:
        offending = check_deferred_absent(rep, repo, quarantine_rel, deferred, skip_large, hashes)
    d_paths = {x["path"] for x in deferred}
    d_fail = [f for f in rep.failures if any(p in f for p in d_paths)]
    count = _deferred_count_text(n_td, n_raw)
    if source_machine:
        d_res = {"mode": "source machine: checked at their live source paths (present by definition; not a failure)",
                 "absent": None, "verified": not d_fail,
                 "summary": f"deferred files: present at their source paths on the source machine (not a failure); "
                            f"correct: {'yes' if not d_fail else 'no'} ({count})"}
    elif after_gate2c:
        d_res = {"mode": "after Gate 2c: required present at their destinations", "absent": None,
                 "verified": not d_fail,
                 "summary": f"deferred files present and verified (after Gate 2c): "
                            f"{'yes' if not d_fail else 'no'} ({count})"}
    else:
        d_res = {"mode": "before Gate 2c: required ABSENT from this machine", "absent": not offending,
                 "verified": None, "offending_paths": offending,
                 "summary": f"deferred files absent: {'yes' if not offending else 'no'} ({count})"}
    d_res.update({"checked": len(deferred), "test_derived": n_td, "contains_test_events": n_raw,
                  "files": sorted(d_paths)})
    raw_entries = [x for x in deferred if x.get("contains_test_events") and not x.get("test_derived")]
    if not raw_entries:
        raw_text = None
    elif check_td_here:
        raw_text = "; ".join(f"{x['path']}: raw hit table with the test events (contains_test_events, not "
                             f"test-derived); checked by size and one SHA-256 of {x['bytes']} B" for x in raw_entries)
    else:
        raw_text = "; ".join(f"{x['path']}: raw hit table with the test events (contains_test_events, not "
                             "test-derived), deferred to after Gate 2c; this verify required it to be ABSENT (live path, "
                             "any content; the same name and size elsewhere, confirmed by one SHA-256 of "
                             f"{x['bytes']} B); Parquet files of another size or name are information only"
                             for x in raw_entries)

    for x in man["groups"]["c"]["files"]:
        p = x["path"]
        concrete = not any(ch in p for ch in "{}<> ")
        state = "present (already produced here)" if concrete and (repo / p.rstrip("/")).exists() else "to be regenerated"
        rep.info("c", f"{state}: {p}  <-  {x['command']}")

    if check_td_here:
        tc_text = (f"{TEST_CACHE_REL}: verified by byte hash only (SHA-256 of its bytes; opaque byte stream, never "
                   "opened as data). It must stay unopened until the final evaluation after Gate 2c.")
    else:
        tc_text = (f"{TEST_CACHE_REL}: test-derived, NOT copied before Gate 2c; this verify required it to be ABSENT "
                   "(live path, quarantine root, and the same name and size anywhere in the working tree; a candidate "
                   "is compared by size and, only if the size matches, by a byte hash - never opened as data). After "
                   "Gate 2c it is copied and verified by byte hash only (verify --after-gate2c), and it stays "
                   "unopened as data until the final evaluation.")
    hazards = man["groups"]["a"].get("laptop_record_hazards", [])
    mode = ("source machine" if source_machine else
            "target machine, after Gate 2c" if after_gate2c else "target machine, before Gate 2c")
    return {"ok": not rep.failures, "mode": mode,
            "phase": None if source_machine else ("after_gate2c" if after_gate2c else "before_gate2c"),
            "repo": str(repo), "manifest": str(manifest_path), "manifest_commit": commit, "head": head,
            "quarantine_root": quarantine_rel, "checks": rep.checks, "failures": rep.failures,
            "warnings": rep.warnings, "lines": [list(l) for l in rep.lines],
            "deferred": d_res, "test_cache": tc_text, "raw_table": raw_text,
            "tracked_laptop_record_hazards": hazards,
            "totals": {k: man["groups"][k]["totals"] for k in ("a", "b1", "b2", "c")},
            "totals_by_phase": man.get("totals_by_phase")}


def cmd_verify(args) -> int:
    repo = Path(args.repo).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else repo / TRANSFER_REL / "MANIFEST.json"
    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2
    try:
        res = verify(repo, manifest, args.quarantine_root, args.source_machine, args.skip_large_hash,
                     after_gate2c=args.after_gate2c)
    except RefusedFile as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    print(f"H200 transfer package verify ({res['mode']}): repo {res['repo']}")
    print(f"manifest {res['manifest']} (commit {res['manifest_commit'][:12]})")
    if not args.source_machine:
        print(f"laptop records expected under {res['quarantine_root']}/")
        print("deferred files (test-derived + raw table): " + res["deferred"]["mode"])
    for status, tag, msg in res["lines"]:
        print(f"{status:4} [{tag}] {msg}")
    print()
    print("TEST CACHE: " + res["test_cache"])
    if res["raw_table"]:
        print("RAW TABLE: " + res["raw_table"])
    print()
    print("Tracked laptop records that arrived with git (hazards on this machine):")
    for h in res["tracked_laptop_record_hazards"]:
        print(f"  - {h['pattern']} ({h['files']} file{'s' if h['files'] != 1 else ''}): {h['hazard']}")
    print()
    print(f"SUMMARY: {res['checks']} checks, {len(res['failures'])} failure(s), {len(res['warnings'])} warning(s) -> "
          f"{'PASS' if res['ok'] else 'FAIL'}")
    print(f"SUMMARY: {res['deferred']['summary']}")
    for p in res["deferred"].get("offending_paths") or []:
        print(f"  DEFERRED FILE ON THIS MACHINE: {p}")
    for f in res["failures"]:
        print(f"  FAIL {f}")
    return 0 if res["ok"] else 1


# ======================================================================================================================
def main(argv=None) -> int:
    default_repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(prog="make_transfer_package.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="(laptop) build groups b1/b2, check git status, hash, write the six outputs",
                        description="Builds groups b1/b2 from the explicit tables, checks each file with git "
                                    "(ls-files --error-unmatch / check-ignore -v), hashes it once (the test cache "
                                    "as an opaque byte stream; any *.csv is refused) and writes the four lists "
                                    "(files_b_inputs.txt, files_b_laptop_records.txt, files_b_test_derived_inputs.txt, "
                                    "files_b_test_derived_records.txt), SHA256SUMS and MANIFEST.json.")
    pl.add_argument("--repo", default=str(default_repo), help="repository root (default: this script's repo)")
    pl.add_argument("--out", default=None, help=f"output directory (default: <repo>/{TRANSFER_REL})")
    pv = sub.add_parser("verify", help="(H200) check presence, size and SHA-256 of the b1/b2 files of the phase",
                        description="Checks the 'now' b1 files at their live paths and the 'now' b2 files under the "
                                    "quarantine root (size + SHA-256), group-a presence and the manifest commit, and "
                                    "FAILS if a stream-dependent laptop record sits at its live path (prints the mv, "
                                    "never runs it). Before Gate 2c (default) it FAILS if any of the eight deferred "
                                    "files exists on this machine: a test-derived file at its live path, under the "
                                    "quarantine root, or with the same name and size anywhere in the working tree "
                                    "(confirmed by SHA-256); the raw Parquet table at its live path or with the same "
                                    "name and size elsewhere (one SHA-256; other Parquet files are information only). "
                                    "With --after-gate2c it requires all eight present and correct instead. The test "
                                    "cache is only ever compared by size and a byte hash and never opened as data.")
    pv.add_argument("--repo", default=str(default_repo), help="repository root (default: this script's repo)")
    pv.add_argument("--manifest", default=None, help=f"default: <repo>/{TRANSFER_REL}/MANIFEST.json")
    pv.add_argument("--quarantine-root", default=QUARANTINE_REL, help=f"repo-relative (default {QUARANTINE_REL})")
    pv.add_argument("--after-gate2c", action="store_true",
                    help="after Gate 2c, once files_b_test_derived_*.txt were copied: require the eight deferred files "
                         "(seven test-derived + the raw Parquet table) present with correct size and SHA-256 (test "
                         "cache: byte hash only) instead of absent")
    pv.add_argument("--source-machine", action="store_true",
                    help="on the machine where the package was made: every b1/b2 file, deferred ones included, "
                         "is checked at its live (source) path and the quarantine / live-path / absence checks are "
                         "skipped")
    pv.add_argument("--skip-large-hash", type=int, default=None, metavar="BYTES",
                    help="files larger than BYTES are checked by size only (quick pass)")
    pv.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    return cmd_list(args) if args.cmd == "list" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
