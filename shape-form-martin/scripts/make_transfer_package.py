#!/usr/bin/env python3
"""H200 transfer package (PLAN.md section 15.1 / 16): the git-ignored files the moved steps need, as rsync/tar file
lists plus checksums, and a verify mode for the machine that receives them.

    uv run python -B shape-form-martin/scripts/make_transfer_package.py list               # laptop (source machine)
    uv run python -B shape-form-martin/scripts/make_transfer_package.py verify [--json]    # H200, from the repo root
    uv run python -B shape-form-martin/scripts/make_transfer_package.py verify --source-machine   # on the laptop

Standard library only (never imports torch or numpy), so it runs before the environment exists (plain `python3`
>= 3.11 works as well).

Groups (MANIFEST.json holds every field; the tables below are the source of truth, each entry justified by a reader
in the code - there is no blind directory walk; glob patterns are only used together with an expected count):
    a    tracked in the pushed branch; arrives with `git pull`. Some are laptop RECORDS that the code trusts
         (registry rows, done-markers, guard spec, ...): they are hazards on another GPU; verify lists them.
    b1   inputs: git-ignored, needed as-is on the H200 at the SAME repo-relative path.
             rsync -a --files-from=shape-form-martin/transfer/files_b_inputs.txt <repo>/ <user>@<h200>:<repo>/
    b2   laptop records: git-ignored per-event arrays (and monitors) computed on this laptop's Monte-Carlo sample
         points. Never copied to their live path (the code would pair H200 scores with them silently); copied into
         the quarantine root, repo-relative path preserved underneath it, for comparison only:
             rsync -a --files-from=shape-form-martin/transfer/files_b_laptop_records.txt <repo>/ \\
                   <user>@<h200>:<repo>/shape-form-martin/cache/laptop_records/
    c    regenerated on the H200 (command or "no command exists yet", laptop time, reproducibility).

Test cache. shape-form-martin/cache/pion_s1_test.pt is COPIED (not rebuilt) and treated here as an OPAQUE BYTE
STREAM: the only access is a chunked binary read inside sha256_file() (SHA-256 of the file's bytes) plus os.stat()
for the size. It is never unpickled, never loaded with torch, never passed to sfm.data.load_sets. A byte hash is not
a test access (nothing is logged in results/test_access.log). The file must stay unopened as data until the final
evaluation after Gate 2c.

CSV. No *.csv is ever listed, stat'ed for the package or hashed: sha256_file() and the table expansion refuse any
path ending in .csv (the 15 GB raw CSV is not part of the package; the Parquet copy is).

Outputs of `list` (in shape-form-martin/transfer/): files_b_inputs.txt, files_b_laptop_records.txt, SHA256SUMS
(sha256sum -c format, two sections with their destination roots in the header) and MANIFEST.json.
`verify` exits non-zero on any failure, including a stream-dependent laptop record found at its LIVE path (the
exact `mv` that would quarantine it is printed, never executed).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

SCHEMA = 1
MARTIN = "shape-form-martin"
PION = "shape-form-proton/results/particles/pion"
TRANSFER_REL = f"{MARTIN}/transfer"
QUARANTINE_REL = f"{MARTIN}/cache/laptop_records"          # inside the git-ignored cache/ (shape-form-martin/.gitignore)
TEST_CACHE_REL = f"{MARTIN}/cache/pion_s1_test.pt"          # opaque byte stream (module docstring)
META_REL = f"{MARTIN}/cache/pion_s1_meta.json"
CHECKSUMS_REL = f"{MARTIN}/results/m0_parity/checksums.json"
REGISTRY_REL = f"{MARTIN}/results/registry.jsonl"
OUTPUT_NAMES = ("files_b_inputs.txt", "files_b_laptop_records.txt", "SHA256SUMS", "MANIFEST.json")
CHUNK = 1 << 22

TEST_CACHE_POLICY = (
    "shape-form-martin/cache/pion_s1_test.pt is copied byte for byte (not rebuilt) and verified by its SHA-256 and "
    "size only: the transfer script reads it as an opaque byte stream (chunked binary read in sha256_file) and never "
    "unpickles, torch-loads or load_sets() it. A byte hash is not a test access and is not logged. The file must stay "
    "unopened as data until the single final evaluation after Gate 2c.")
CSV_POLICY = ("No *.csv is listed, stat'ed for the package or hashed (the 15 GB raw CSV is not part of the package; "
              "only the Parquet copy is). sha256_file() and the table expansion refuse any .csv path.")


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
# ======================================================================================================================
def _b(pattern, group, role, readers, stream_dependent, stream, decision, regeneration, count=1, hazard=None,
       cross_check=None, opaque=False):
    return {"pattern": pattern, "count": count, "group": group, "role": role, "readers": readers,
            "stream_dependent": stream_dependent, "stream": stream, "decision": decision,
            "regeneration": regeneration, "hazard": hazard, "cross_check": cross_check, "opaque": opaque}


COPY_LIVE = "copy to the same repo-relative path (rsync -a, keeps mtime); verify size + SHA-256"
COPY_Q = (f"copy ONLY into {QUARANTINE_REL}/<same path> (comparison partner); never to the live path; verify there")
RUNS8 = ("s1_seed11", "s1_seed12", "s1_seed13", "s1_seed14", "s1_seed15", "s2_seed11", "s2_seed12", "b0_seed0")
GUARDED3 = ("b0_seed0", "s2_seed11", "s2_seed12")
NATIVE144 = "laptop native stream (RTX 4060 Laptop, launch grid 144), CUDA generator seed 12345"

GROUP_B1 = [
    _b("shape-form-original/data/pion_standalone_N100000_M0_20GeV_reg_balanced_s1_v3.parquet", "b1",
       "raw hit table (Parquet copy of the CSV). No moved step reads it while both caches are copied",
       ["sfm/paths.py:41-43 parquet_path", "sfm/data.py:51-55 parquet_getter <- build_cache (data.py:214-217)",
        "scripts/build_cache.py:32 (rebuild)", "scripts/parity.py:471-472 (raw-file check, default mode only)",
        "scripts/check_protected.py (size + mtime against checksums.json stat_only)"],
       False, "none (data)", COPY_LIVE + "; needed only for a cross-check rebuild into scratch directories and for "
       "the zero-training geometry arm (absolute shower origin, AUDIT section 4.3; user decision: no trained arm)",
       {"command": "not regenerated (source data)", "laptop_time": None, "reproducibility": "n/a"},
       cross_check={"kind": "checksums_stat"}),
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
       cross_check={"kind": "meta", "key": "trainval"}),
    _b(TEST_CACHE_REL, "b1", "TEST cache (14,956 events) - OPAQUE BYTE STREAM, hash and size only",
       ["sfm/data.py:287 load_sets(..., allow_test=True, reason=...) only: scripts/parity.py:318,368 and the future "
        "final_test.py (after Gate 2c)"],
       False, "none (deterministic CPU build from the Parquet file)",
       "COPY (do not rebuild); verify SHA-256 = cache/pion_s1_meta.json files.test.sha256 by a plain byte hash; stays "
       "unopened as data until the final evaluation after Gate 2c",
       {"command": "only as a side effect of build_cache.py (data.py:227), which also rewrites the copied files",
        "laptop_time": "(part of the 12.0 s build)",
        "reproducibility": "as the train/val cache; C1 validated these exact bytes"},
       cross_check={"kind": "meta", "key": "test"}, opaque=True),
    _b(META_REL, "b1", "cache metadata (geometry, counts, per-file SHA-256 of both caches)",
       ["sfm/data.py:300 load_meta", "sfm/registry.py:144-145 validation_reference (trainval SHA-256 = fingerprint "
        "key), :161 counts"],
       False, "none", "COPY; never byte-reproducible (created_utc, Parquet mtime_ns), and it carries the reference "
       "hashes of both caches", {"command": "build_cache.py (see above)", "laptop_time": None,
                                 "reproducibility": "never byte-identical"}),
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
       cross_check={"kind": "checksums"}),
    _b(f"{PION}/mlp_std.json", "b1", "stored validation mean of mlp_std (reference machine)",
       ["scripts/parity.py:91 stored_val_means"], True, "reference machine of the earlier study (grid 1056), scalar",
       COPY_LIVE + "; validation-only parity target on the H200", {"command": "not regenerable", "laptop_time": None,
                                                                   "reproducibility": "n/a"},
       cross_check={"kind": "checksums"}),
    _b(f"{PION}/new3_std.json", "b1", "stored validation means of the formula rungs (reference machine)",
       ["scripts/parity.py:92 stored_val_means"], True, "reference machine of the earlier study (grid 1056), scalars",
       COPY_LIVE + "; validation-only parity target on the H200", {"command": "not regenerable", "laptop_time": None,
                                                                   "reproducibility": "n/a"},
       cross_check={"kind": "checksums"}),
    _b(f"{PION}/test_v3_nll.npz", "b1", "stored per-event TEST arrays of the earlier study (not a laptop record)",
       ["scripts/parity.py:317 (C1), :407 (default mode)"], True,
       "reference machine of the earlier study (grid 1056), test split",
       COPY_LIVE + "; pairing partner for a labelled secondary test column and the abort-on-fail check of the final "
       "test if the H200 grid is 1056", {"command": "not regenerable", "laptop_time": None, "reproducibility": "n/a"},
       cross_check={"kind": "checksums"}),
    _b(f"{MARTIN}/results/m0_parity/val_event_number.npy", "b1", "validation event numbers in pairing order",
       ["sfm/registry.py:142-143 validation_reference (every evaluate / check_validation)",
        "tests/test_registry.py real-data test"], False, "none (event numbers)",
       COPY_LIVE + "; written only by parity.py default mode (parity.py:414), which opens the test split",
       {"command": "no command exists (parity.py default mode opens test)", "laptop_time": None,
        "reproducibility": "bit-identical (= cache extras['val']['event_number'])"}),
    _b(f"{MARTIN}/results/m0_parity/test_event_number.npy", "b1", "test event numbers (order check of the final "
       "test; event numbers only, reading it is not a test access)",
       ["scripts/parity.py:316 (C1 order check)", "future final_test.py"], False, "none (event numbers)",
       COPY_LIVE, {"command": "no command exists (parity.py default mode opens test)", "laptop_time": None,
                   "reproducibility": "bit-identical"}),
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
         ["no reader in code (written by parity.py:412)"], True, NATIVE144 + ", K=64, test split", COPY_Q,
         {"command": "not regenerated (Gate 0 record)", "laptop_time": "1.37 s / 0.89 s (parity.json)",
          "reproducibility": "machine-dependent"}) for m in ("mlp", "formula")],
    *[_b(f"{MARTIN}/results/m0_parity/{m}_{s}_nll_ref1056.npy", "b2",
         f"laptop emulation of the reference stream ({m}, {s})", ["no reader in code (written by parity.py:%s)"
                                                                 % ("314/466" if s == "val" else "324")],
         True, "emulated reference stream, launch grid 1056 (laptop CPU emulator + laptop GPU arithmetic)", COPY_Q
         + "; expected values of the H200's own common.test_nll if its grid is 1056 (tolerance 1e-4 per event)",
         {"command": "not regenerated (C1 / Gate 0 record)", "laptop_time": "C1 total 7.1 s",
          "reproducibility": "emulated stream is machine-independent; arithmetic agrees to ~3e-5 per event"})
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
]


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
        for rel in paths_:
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
            item = {"path": rel, "group": e["group"], "bytes": size, "sha256": sha, "role": e["role"],
                    "readers": e["readers"], "git": {"tracked": tracked, "ignored": rule is not None, "rule": rule},
                    "stream_dependent": e["stream_dependent"], "stream": e["stream"], "decision": e["decision"],
                    "destination": rel if e["group"] == "b1" else f"{QUARANTINE_REL}/{rel}",
                    "regeneration": e["regeneration"], "hazard": e["hazard"],
                    "opaque_byte_stream": bool(e["opaque"]) or rel == TEST_CACHE_REL}
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

    # ---- constants: the cited line must still say what the table claims ----
    consts = []
    for k in CONSTANTS:
        f = repo / k["file"]
        try:
            line = f.read_text().splitlines()[k["line"] - 1]
            ok = k["must_contain"] in line
        except (OSError, IndexError):
            line, ok = None, False
        if not ok:
            problems.append(f"constant {k['file']}:{k['line']}: line does not contain {k['must_contain']!r}")
        consts.append({**k, "line_text": line.strip() if line else None, "line_checked": ok})

    if problems:
        raise PackageError("list refused; nothing written:\n  - " + "\n  - ".join(problems))

    def tot(items):
        return {"files": len(items), "bytes": sum(i["bytes"] or 0 for i in items)}

    return {
        "schema": SCHEMA,
        "title": "H200 transfer package (PLAN.md section 16)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": f"{MARTIN}/scripts/make_transfer_package.py list",
        "git": {"commit": head, "branch": branch, "tracked_tree_dirty": dirty,
                "note": "verify requires this commit to be an ancestor of (or equal to) the H200 checkout's HEAD"},
        "policies": {
            "test_cache": TEST_CACHE_POLICY, "csv": CSV_POLICY, "quarantine_root": QUARANTINE_REL,
            "quarantine_rule": "group b2 files are copied ONLY under the quarantine root (repo-relative path preserved); "
                               "a stream-dependent b2 file found at its live path is a verify FAILURE",
            "cache_decision": {
                "decision": "COPY both cache files and pion_s1_meta.json; verify by SHA-256 against the meta file "
                            "(and this manifest); do NOT rebuild in place",
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
                    "optional cross-check without touching the live files: SFM_CACHE_DIR=<scratch>/c "
                    "SFM_RESULTS_DIR=<scratch>/r uv run python -B shape-form-martin/scripts/build_cache.py, then "
                    "compare the scratch SHA-256 values (equal = bit-identical rebuild); delete the scratch test file"]},
        },
        "transfer_commands": {
            "inputs": f"rsync -a --files-from={TRANSFER_REL}/files_b_inputs.txt <repo>/ <user>@<h200>:<repo>/",
            "laptop_records": f"rsync -a --files-from={TRANSFER_REL}/files_b_laptop_records.txt <repo>/ "
                              f"<user>@<h200>:<repo>/{QUARANTINE_REL}/",
            "tar_alternative": f"tar -C <repo> -cf inputs.tar -T {TRANSFER_REL}/files_b_inputs.txt",
            "verify_on_h200": f"python3 -B {MARTIN}/scripts/make_transfer_package.py verify   (from <repo>)",
            "verify_on_laptop": f"python3 -B {MARTIN}/scripts/make_transfer_package.py verify --source-machine",
        },
        "groups": {
            "a": {"description": "tracked in the pushed branch; arrives with git pull (sizes are a snapshot at "
                                 "generation time; verify checks presence and tracking only)",
                  "files": a_files, "totals": tot(a_files),
                  "laptop_record_hazards": _hazard_list()},
            "b1": {"description": "git-ignored inputs; copy to the same repo-relative path", "files": b["b1"],
                   "totals": tot(b["b1"])},
            "b2": {"description": f"git-ignored laptop records; copy ONLY under {QUARANTINE_REL}/ for comparison",
                   "files": b["b2"], "totals": tot(b["b2"])},
            "c": {"description": "regenerated on the H200 (not copied)", "files": c_items,
                  "totals": {"entries": len(c_items)}},
        },
        "excluded": EXCLUDED,
        "machine_constants": consts,
        "tests_expected_to_fail_on_h200": TESTS_ON_H200,
    }


def write_outputs(repo: Path, out_dir: Path, man: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    b1, b2 = man["groups"]["b1"]["files"], man["groups"]["b2"]["files"]
    (out_dir / "files_b_inputs.txt").write_text("".join(f"{x['path']}\n" for x in b1))
    (out_dir / "files_b_laptop_records.txt").write_text("".join(f"{x['path']}\n" for x in b2))
    hdr = [
        f"# SHA-256 of the H200 transfer package (group b), generated at commit {man['git']['commit']} "
        f"({man['generated_utc']}) by make_transfer_package.py list.",
        "# Paths are repo-relative. Two sections, each relative to its own DESTINATION ROOT on the H200:",
        f"#   inputs          ({len(b1)} files, files_b_inputs.txt)          root: <repo>/",
        f"#   laptop_records  ({len(b2)} files, files_b_laptop_records.txt)  root: <repo>/{QUARANTINE_REL}/",
        "# On the H200, from <repo>/:",
        f"#   sed -n '/^# BEGIN inputs/,/^# END inputs/p' {TRANSFER_REL}/SHA256SUMS | sha256sum -c",
        f"#   (cd {QUARANTINE_REL} && sed -n '/^# BEGIN laptop_records/,/^# END laptop_records/p' "
        "../../transfer/SHA256SUMS | sha256sum -c)",
        f"# On the laptop (both lists at their live source paths), from <repo>/: sha256sum -c {TRANSFER_REL}/SHA256SUMS",
        f"# {TEST_CACHE_REL} is hashed as an opaque byte stream; it stays unopened until the final evaluation "
        "after Gate 2c.",
        "# No *.csv is listed.",
    ]
    body = hdr + ["# BEGIN inputs"] + [f"{x['sha256']}  {x['path']}" for x in b1] + ["# END inputs",
                                                                                    "# BEGIN laptop_records"]
    body += [f"{x['sha256']}  {x['path']}" for x in b2] + ["# END laptop_records"]
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
        print(f"group {k:2}: {t['files']:4d} files, {t['bytes']:>12d} bytes")
    print(f"group c : {g['c']['totals']['entries']} entries (regenerated on the H200)")
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
        rep.ok(tag, f"{size:>11d} B  sha256 {got[:16]}...  {rel}" + ("  [opaque byte hash]" if opaque else ""))


def verify(repo: Path, manifest_path: Path, quarantine_rel: str, source_machine: bool, skip_large) -> dict:
    man = json.loads(manifest_path.read_text())
    rep = Report()
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
        _check_file(rep, "b1", repo / x["path"], x["path"], x["bytes"], x["sha256"], skip_large,
                    x.get("opaque_byte_stream", False))
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
        if source_machine:
            _check_file(rep, "b2", live, rel, x["bytes"], x["sha256"], skip_large, False)
            continue
        _check_file(rep, "b2", qroot / rel, f"{quarantine_rel}/{rel}", x["bytes"], x["sha256"], skip_large, False)
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

    for x in man["groups"]["c"]["files"]:
        p = x["path"]
        concrete = not any(ch in p for ch in "{}<> ")
        state = "present (already produced here)" if concrete and (repo / p.rstrip("/")).exists() else "to be regenerated"
        rep.info("c", f"{state}: {p}  <-  {x['command']}")

    hazards = man["groups"]["a"].get("laptop_record_hazards", [])
    return {"ok": not rep.failures, "mode": "source machine" if source_machine else "target machine",
            "repo": str(repo), "manifest": str(manifest_path), "manifest_commit": commit, "head": head,
            "quarantine_root": quarantine_rel, "checks": rep.checks, "failures": rep.failures,
            "warnings": rep.warnings, "lines": [list(l) for l in rep.lines],
            "test_cache": f"{TEST_CACHE_REL}: verified by byte hash only (SHA-256 of its bytes; opaque byte stream, "
                          "never opened as data). It must stay unopened until the final evaluation after Gate 2c.",
            "tracked_laptop_record_hazards": hazards,
            "totals": {k: man["groups"][k]["totals"] for k in ("a", "b1", "b2", "c")}}


def cmd_verify(args) -> int:
    repo = Path(args.repo).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else repo / TRANSFER_REL / "MANIFEST.json"
    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2
    try:
        res = verify(repo, manifest, args.quarantine_root, args.source_machine, args.skip_large_hash)
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
    for status, tag, msg in res["lines"]:
        print(f"{status:4} [{tag}] {msg}")
    print()
    print("TEST CACHE: " + res["test_cache"])
    print()
    print("Tracked laptop records that arrived with git (hazards on this machine):")
    for h in res["tracked_laptop_record_hazards"]:
        print(f"  - {h['pattern']} ({h['files']} file{'s' if h['files'] != 1 else ''}): {h['hazard']}")
    print()
    print(f"SUMMARY: {res['checks']} checks, {len(res['failures'])} failure(s), {len(res['warnings'])} warning(s) -> "
          f"{'PASS' if res['ok'] else 'FAIL'}")
    for f in res["failures"]:
        print(f"  FAIL {f}")
    return 0 if res["ok"] else 1


# ======================================================================================================================
def main(argv=None) -> int:
    default_repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(prog="make_transfer_package.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="(laptop) build groups b1/b2, check git status, hash, write the four outputs",
                        description="Builds groups b1/b2 from the explicit tables, checks each file with git "
                                    "(ls-files --error-unmatch / check-ignore -v), hashes it once (the test cache "
                                    "as an opaque byte stream; any *.csv is refused) and writes files_b_inputs.txt, "
                                    "files_b_laptop_records.txt, SHA256SUMS and MANIFEST.json.")
    pl.add_argument("--repo", default=str(default_repo), help="repository root (default: this script's repo)")
    pl.add_argument("--out", default=None, help=f"output directory (default: <repo>/{TRANSFER_REL})")
    pv = sub.add_parser("verify", help="(H200) check presence, size and SHA-256 of every b1/b2 file",
                        description="Checks b1 files at their live paths and b2 files under the quarantine root "
                                    "(size + SHA-256), group-a presence and the manifest commit, and FAILS if a "
                                    "stream-dependent laptop record sits at its live path (prints the mv, never runs "
                                    "it). The test cache is verified by a byte hash only and never opened as data.")
    pv.add_argument("--repo", default=str(default_repo), help="repository root (default: this script's repo)")
    pv.add_argument("--manifest", default=None, help=f"default: <repo>/{TRANSFER_REL}/MANIFEST.json")
    pv.add_argument("--quarantine-root", default=QUARANTINE_REL, help=f"repo-relative (default {QUARANTINE_REL})")
    pv.add_argument("--source-machine", action="store_true",
                    help="on the machine where the package was made: b2 files are checked at their live (source) "
                         "paths and the quarantine / live-path checks are skipped")
    pv.add_argument("--skip-large-hash", type=int, default=None, metavar="BYTES",
                    help="files larger than BYTES are checked by size only (quick pass)")
    pv.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    return cmd_list(args) if args.cmd == "list" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
