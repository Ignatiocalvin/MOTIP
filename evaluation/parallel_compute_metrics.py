"""
Parallel worker to recompute MOTA + HOTA for all cache entries in parallel.
Each worker atomically claims unclaimed entries using file locking.
Usage:
    python evaluation/parallel_compute_metrics.py [--workers N]
"""
import sys, os, json, fcntl, time, importlib.util, argparse, traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ── paths ────────────────────────────────────────────────────────────────────
MOTIP_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(MOTIP_ROOT))

# Patch numpy for motmetrics
import numpy
if not hasattr(numpy, "asfarray"):
    numpy.asfarray = lambda a, dtype=numpy.float64: numpy.asarray(a, dtype=dtype)

CACHE_FILE = MOTIP_ROOT / "evaluation" / ".metrics_cache.json"
LOCK_FILE  = MOTIP_ROOT / "evaluation" / ".metrics_cache.lock"

# ── load helpers lazily per-process ──────────────────────────────────────────
def _load_helpers():
    import motmetrics as mm
    spec = importlib.util.spec_from_file_location(
        "cam", str(MOTIP_ROOT / "evaluation" / "compute_all_metrics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    spec2 = importlib.util.spec_from_file_location(
        "hota", str(MOTIP_ROOT / "evaluation" / "compute_hota_pdestre.py"))
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)

    return mod.compute_metrics, mod2.run_hota


def _read_cache_locked(lock_fh):
    lock_fh.seek(0)
    try:
        return json.loads(lock_fh.read())
    except Exception:
        return {}


def _write_cache_locked(lock_fh, cache):
    data = json.dumps(cache, indent=2)
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(data)
    lock_fh.flush()
    os.fsync(lock_fh.fileno())


def claim_next_entry(worker_id: int):
    """Open cache with exclusive lock, find unclaimed empty entry, return key or None."""
    with open(LOCK_FILE, "r+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cache = _read_cache_locked(lf)
        for key, val in cache.items():
            if val == {} or (isinstance(val, dict) and "_claimed_by" in val and (time.time() - val.get("_ts", time.time())) > 600):  # unclaimed or stale claim (>10min)
                # Claim it with a sentinel so other workers skip it
                cache[key] = {"_claimed_by": worker_id, "_ts": time.time()}
                _write_cache_locked(lf, cache)
                fcntl.flock(lf, fcntl.LOCK_UN)
                return key
        fcntl.flock(lf, fcntl.LOCK_UN)
    return None


def save_result(key: str, result: dict):
    import shutil
    with open(LOCK_FILE, "r+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cache = _read_cache_locked(lf)
        cache[key] = result
        _write_cache_locked(lf, cache)
        # Also sync to main cache file immediately
        shutil.copy(LOCK_FILE, CACHE_FILE)
        fcntl.flock(lf, fcntl.LOCK_UN)


def release_claim(key: str):
    """Reset a claimed entry back to {} if computation failed."""
    with open(LOCK_FILE, "r+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cache = _read_cache_locked(lf)
        if isinstance(cache.get(key), dict) and "_claimed_by" in cache[key]:
            cache[key] = {}
        _write_cache_locked(lf, cache)
        fcntl.flock(lf, fcntl.LOCK_UN)


def worker_fn(worker_id: int):
    compute_metrics, run_hota = _load_helpers()

    GT_DIR = MOTIP_ROOT / "data" / "P-DESTRE" / "annotations"
    SPLITS_DIR = MOTIP_ROOT / "data" / "P-DESTRE" / "splits"

    def get_val_seqs(split: str):
        return [s.replace(".txt", "") for s in
                (SPLITS_DIR / f"{split}.txt").read_text().splitlines() if s.strip()]

    processed = 0
    while True:
        key = claim_next_entry(worker_id)
        if key is None:
            break  # nothing left

        # Parse key: "<exp_folder>/<val_split>/epoch_<N>"
        try:
            parts = key.split("/")
            ep_num = int(parts[-1].replace("epoch_", ""))
            val_split = parts[-2]
            exp_folder = "/".join(parts[:-2])
        except Exception:
            release_claim(key)
            continue

        # Find tracker dir — three possible layouts:
        #   1. outputs/exp/train/eval_during_train/epoch_N/tracker  (r50 training runs)
        #   2. outputs/exp/eval_during_train/epoch_N/tracker         (rfdetr training runs)
        #   3. outputs/exp/eval/PDESTRE_<split>/checkpoint_N/tracker (explicit eval runs e.g. Test_0)
        # For Test_ splits, check the explicit eval dir first to avoid picking up
        # the training eval dir which contains val sequences, not test sequences.
        exp_path = MOTIP_ROOT / "outputs" / exp_folder
        tracker_dir = None
        explicit_candidate = exp_path / "eval" / f"PDESTRE_{val_split}" / f"checkpoint_{ep_num}" / "tracker"
        if explicit_candidate.exists():
            tracker_dir = explicit_candidate
        if tracker_dir is None:
            for subpath in ["train/eval_during_train", "eval_during_train"]:
                candidate = exp_path / subpath / f"epoch_{ep_num}" / "tracker"
                if candidate.exists():
                    tracker_dir = candidate
                    break

        if tracker_dir is None:
            # Mark permanently as skipped so workers don't loop on it
            save_result(key, {"skipped": "tracker_dir_missing"})
            print(f"[W{worker_id}] skipped (no tracker dir): {key}", flush=True)
            continue

        val_seqs = get_val_seqs(val_split)
        t0 = time.time()
        try:
            m = compute_metrics(tracker_dir, val_seqs)
        except Exception as e:
            print(f"[W{worker_id}] motmetrics error for {key}: {e}", flush=True)
            release_claim(key)
            continue

        try:
            hota_m = run_hota(tracker_dir, val_split)
            for k2 in ("hota", "deta", "assa", "clear_mota", "clear_idf1"):
                if k2 in hota_m:
                    m[k2] = hota_m[k2]
        except Exception as e:
            print(f"[W{worker_id}] HOTA error for {key}: {e}", flush=True)

        save_result(key, m)
        elapsed = time.time() - t0
        hota_str = f"  HOTA={m['hota']:.1f}" if "hota" in m else ""
        print(f"[W{worker_id}] {key}  MOTA={m['mota']:.1f}{hota_str}  ({elapsed:.0f}s)", flush=True)
        processed += 1

    print(f"[W{worker_id}] done, processed {processed} entries", flush=True)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Ensure lock file is a copy of the cache (serves as both lock and data)
    if not LOCK_FILE.exists():
        import shutil
        shutil.copy(CACHE_FILE, LOCK_FILE)
    else:
        # Sync lock file from cache file (in case cache was updated externally)
        import shutil
        shutil.copy(CACHE_FILE, LOCK_FILE)

    # Count pending entries
    cache = json.loads(CACHE_FILE.read_text())
    pending = sum(1 for v in cache.values() if v == {})
    total = len(cache)
    done = sum(1 for v in cache.values() if v and "hota" in v)
    print(f"Cache state: {done}/{total} done, {pending} pending")

    if pending == 0:
        print("Nothing to do.")
        return

    n_workers = min(args.workers, pending)
    print(f"Launching {n_workers} parallel workers...")

    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, list(range(n_workers)))

    # Sync lock file back to cache file
    import shutil
    shutil.copy(LOCK_FILE, CACHE_FILE)

    done_after = sum(1 for v in json.loads(CACHE_FILE.read_text()).values() if v and "hota" in v)
    print(f"\nAll workers done. Cache: {done_after}/{total} entries complete.")


if __name__ == "__main__":
    main()
