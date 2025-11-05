# pytest -q -s -rA utils/test_incremental_vs_gwdama.py
from __future__ import annotations
import os, json, math, subprocess, inspect
from pathlib import Path
from datetime import datetime, timezone, timedelta

import h5py
import numpy as np
import pytest

# -----------------------------
# Configurable test parameters
# -----------------------------

TEST_START = os.environ.get("GWDA_TEST_START", "2025-08-14 00:00:00")
TEST_END   = os.environ.get("GWDA_TEST_END",   "2025-08-14 00:05:00")
APPEND_INTERVAL = int(os.environ.get("GWDA_TEST_APPEND_INTERVAL", "100"))  # 100s -> 3 chunks

FFL_PATH = Path(os.environ.get("GWDA_TEST_FFL", "/virgoData/ffl/trend.ffl"))
FFL_SPEC = os.environ.get("GWDA_TEST_FFL_SPEC", "V1trend")
CHANNELS_FILE = Path(os.environ.get("GWDA_TEST_CHANNELS", "channels/o4_channels.txt"))

MAX_CHANNELS = int(os.environ.get("GWDA_TEST_MAX_CHANNELS", "8"))
WORKER = Path(os.environ.get("GWDA_TEST_WORKER", "scripts/gwf_to_h5_incremental.py"))

# -----------------------------
# Utilities
# -----------------------------

def parse_utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

def utc_to_gps(dt: datetime) -> float:
    GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
    GPS_LEAP_SECONDS = 18  # O4 era
    return (dt - GPS_EPOCH).total_seconds() + GPS_LEAP_SECONDS

def load_first_n_channels(ch_file: Path, n: int) -> list[str]:
    out = []
    with ch_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
            if not raw:
                continue
            out.append(raw)
            if len(out) >= n:
                break
    if not out:
        raise RuntimeError(f"No channels found in {ch_file}")
    return out

def run_worker(out_root: Path, start: str, end: str, channels_file: Path,
               ffl_path: Path, ffl_spec: str, append_interval: int, resume=False, log_level="INFO"):
    cmd = [
        "python", str(WORKER),
        "--start", start,
        "--end",   end,
        "--channels-file", str(channels_file),
        "--ffl-path", str(ffl_path),
        "--ffl-spec", str(ffl_spec),
        "--out", str(out_root),
        "--append-interval", str(append_interval),
        "--log-level", log_level,
    ]
    if resume:
        cmd.append("--resume")
    print(f"\n[run] {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("[worker stdout]\n", res.stdout)
    print("[worker stderr]\n", res.stderr)
    if res.returncode != 0:
        raise AssertionError(f"worker failed (code {res.returncode}). See stdout/stderr above.")
    return res

def day_folder_for(out_root: Path, start_dt: datetime) -> Path:
    return out_root / start_dt.date().isoformat()

def open_h5(h5_path: Path):
    return h5py.File(h5_path, "r")

def h5_channels_map(h5: h5py.File) -> dict[str, h5py.Dataset]:
    g = h5.get("channels", None)
    if g is None:
        return {k: v for k, v in h5.items() if isinstance(v, h5py.Dataset)}
    return {k: v for k, v in g.items() if isinstance(v, h5py.Dataset)}

def crop_overlap(a_t0_gps: float, a_sr: float, a_n: int,
                 b_t0_gps: float, b_sr: float, b_n: int) -> tuple[slice, slice]:
    a_dt = 1.0 / a_sr
    b_dt = 1.0 / b_sr
    ref = "a" if a_dt >= b_dt else "b"
    if ref == "a":
        b_i0 = int(round((a_t0_gps - b_t0_gps) / b_dt))
        b_i1 = int(round(((a_t0_gps + (a_n - 1) * a_dt) - b_t0_gps) / b_dt)) + 1
        b_i0 = max(0, b_i0); b_i1 = min(b_n, b_i1)
        a_i0 = int(round(((b_t0_gps + b_i0 * b_dt) - a_t0_gps) / a_dt))
        a_i1 = int(round(((b_t0_gps + (b_i1 - 1) * b_dt) - a_t0_gps) / a_dt)) + 1
        a_i0 = max(0, min(a_n, a_i0)); a_i1 = max(0, min(a_n, a_i1))
        return slice(a_i0, a_i1), slice(b_i0, b_i1)
    else:
        a_sl, b_sl = crop_overlap(b_t0_gps, b_sr, b_n, a_t0_gps, a_sr, a_n)
        return b_sl, a_sl

# -----------------------------
# gwdama helpers
# -----------------------------

def gwdama_write_h5_ref(out_h5: Path, start_dt: datetime, end_dt: datetime,
                        channels: list[str], ffl_spec: str, ffl_path: Path):
    """
    Try several common signatures of gwdama.write_gwdama to produce an HDF5:
      1) GwDataManager.write_gwdama(filename=..., start=..., end=..., channels=..., ffl_spec=..., ffl_path=...)
    If all fail, raise with a detailed message.
    """
    from gwdama.io import GwDataManager

    if GwDataManager is None:
        pytest.skip("Could not import gwdama (neither 'gwdama' nor 'gwdama.io').")

    # Clean target
    if out_h5.exists():
        out_h5.unlink()

    start_gps = float(utc_to_gps(start_dt))
    end_gps   = float(utc_to_gps(end_dt))

    try:
        print("[gwdama] Trying read_gwdata + write_gwdama(data/meta)...")
        # obtain data/meta first
        dm = GwDataManager()
        result = dm.read_gwdata(
            start=start_gps, end=end_gps, channels=channels,
            ffl_spec=ffl_spec, ffl_path=str(ffl_path), return_output=True
        )
        '''if isinstance(result, tuple):
            data_dict = result[0]
            meta_dict = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        else:
            data_dict = result
            meta_dict = {}'''

        # write using a data/meta-style API (if present)
        result = dm.write_gwdama(
            filename=str(out_h5),
        )
        if out_h5.exists():
            print("[gwdama] Wrote via write_gwdama(data/meta)")
            return

    # If we reach here, we couldn't produce the file
    except Exception as e:
        raise AssertionError(f"gwdama.write_gwdama failed: {e}") from e

# -----------------------------
# Pre-check (clear skip reasons)
# -----------------------------

def _env_requirements() -> tuple[bool, str]:
    problems = []
    if not FFL_PATH.exists():
        problems.append(f"FFL not found: {FFL_PATH} (set GWDA_TEST_FFL)")
    if not CHANNELS_FILE.exists():
        problems.append(f"channels file not found: {CHANNELS_FILE} (set GWDA_TEST_CHANNELS)")
    if not WORKER.exists():
        problems.append(f"worker script not found: {WORKER} (set GWDA_TEST_WORKER)")
    from gwdama.io import GwDataManager
    if GwDataManager is None:
        problems.append("gwdama not importable (neither 'gwdama' nor 'gwdama.io')")
    ok = len(problems) == 0
    reason = " | ".join(problems) if problems else ""
    return ok, reason

# -----------------------------
# The test
# -----------------------------

def test_end_to_end_and_resume(tmp_path: Path):
    ok, reason = _env_requirements()
    if not ok:
        pytest.skip(f"Skipping test due to environment: {reason}")

    print(f"Using worker script: {WORKER}")
    subset = load_first_n_channels(CHANNELS_FILE, MAX_CHANNELS)
    print(f"Testing with {len(subset)} channels (subset of {CHANNELS_FILE})")

    small_channels_file = tmp_path / "channels_small.txt"
    small_channels_file.write_text("\n".join(subset), encoding="utf-8")

    out_root  = tmp_path / "out_worker"
    out_root_ref = tmp_path / "out_ref"
    out_root.mkdir(parents=True, exist_ok=True)
    out_root_ref.mkdir(parents=True, exist_ok=True)

    start = TEST_START
    end   = TEST_END
    start_dt = parse_utc(start)
    end_dt   = parse_utc(end)
    day_dir  = day_folder_for(out_root, start_dt)
    ymd = start_dt.strftime("%Y%m%d")
    h5_path  = day_dir / f"trend_{ymd}.h5"
    state_path = day_dir / "state_day.json"
    summary_path = day_dir / f"trend_{ymd}.summary.json"
    coverage_path = day_dir / "coverage.json"

    # 1) Partial run: first 200 seconds
    end_partial = (start_dt + timedelta(seconds=APPEND_INTERVAL*2)).strftime("%Y-%m-%d %H:%M:%S")
    run_worker(out_root, start, end_partial, small_channels_file, FFL_PATH, FFL_SPEC, APPEND_INTERVAL, resume=False, log_level="INFO")
    assert h5_path.exists(), "HDF5 not created after first run"
    assert state_path.exists(), "state_day.json missing after first run"

    size_after_first = h5_path.stat().st_size
    print(f"Partial run complete up to {end_partial} (size={size_after_first} B)")

    # 2) Resume to full 5 minutes
    run_worker(out_root, start, end, small_channels_file, FFL_PATH, FFL_SPEC, APPEND_INTERVAL, resume=True, log_level="INFO")

    st = json.loads(state_path.read_text(encoding="utf-8"))
    assert st["status"] == "completed"
    assert st["last_completed_chunk"] == 2  # 3 chunks => last index 2

    size_after_resume = h5_path.stat().st_size
    assert size_after_resume >= size_after_first, "Resume did not append as expected"
    print(f"Resume complete (size={size_after_resume} B)")

    assert summary_path.exists(), "summary json missing"
    assert coverage_path.exists(), "coverage json missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("elapsed_seconds") is not None, "elapsed_seconds missing in summary"
    print(f"Elapsed: {summary['elapsed_seconds']} s")

    # 3) gwdama reference H5 using write_gwdama
    ref_h5 = out_root_ref / f"ref_{ymd}.h5"
    gwdama_write_h5_ref(ref_h5, start_dt, end_dt, subset, FFL_SPEC, FFL_PATH)
    assert ref_h5.exists(), "reference h5 not created by gwdama write_gwdama"

    # 4) Compare worker vs gwdama reference on overlap
    with open_h5(h5_path) as hw, open_h5(ref_h5) as hr:
        cw = h5_channels_map(hw)
        cr = h5_channels_map(hr)
        common = sorted(set(cw.keys()) & set(cr.keys()))
        assert common, "No common channels between worker and reference outputs"

        for ch in common:
            ds_w = cw[ch]; ds_r = cr[ch]
            sr_w = float(ds_w.attrs.get("sample_rate", 1.0))
            sr_r = float(ds_r.attrs.get("sample_rate", 1.0))
            t0_w = float(ds_w.attrs.get("t0", 0.0))
            t0_r = float(ds_r.attrs.get("t0", 0.0))
            assert math.isclose(sr_w, sr_r, rel_tol=0, abs_tol=1e-9), f"{ch}: sample_rate mismatch {sr_w} vs {sr_r}"
            a = np.asarray(ds_w[...]); b = np.asarray(ds_r[...])
            if a.size == 0 or b.size == 0:
                continue
            a_sl, b_sl = crop_overlap(t0_w, sr_w, a.size, t0_r, sr_r, b.size)
            a_c = a[a_sl]; b_c = b[b_sl]
            n = min(a_c.size, b_c.size)
            if n == 0:
                continue
            np.testing.assert_allclose(a_c[:n], b_c[:n], rtol=1e-5, atol=1e-6, err_msg=f"data mismatch on {ch}")
    print("OK: worker output matches gwdama reference on overlapping samples")
