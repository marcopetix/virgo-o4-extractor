# pytest -q -s -rA utils/test_channels_presence_patterns.py
from __future__ import annotations
import os, json, subprocess, stat, textwrap
from pathlib import Path
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest

WORKER = Path(os.environ.get("GWDA_TEST_WORKER", "scripts/gwf_to_h5_incremental.py"))
APPEND_INTERVAL = int(os.environ.get("GWDA_TEST_APPEND_INTERVAL", "100"))  # 100 s
TEST_START = os.environ.get("GWDA_TEST_START", "2025-08-14 00:00:00")
TEST_END   = os.environ.get("GWDA_TEST_END",   "2025-08-14 00:05:00")      # 3 chunks (300 s)

def parse_utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

def utc_to_gps(dt: datetime) -> float:
    GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
    GPS_LEAP_SECONDS = 18  # O4 era
    return (dt - GPS_EPOCH).total_seconds() + GPS_LEAP_SECONDS

@pytest.mark.skipif(not WORKER.exists(), reason="worker script not found; set GWDA_TEST_WORKER")
def test_channel_presence_patterns(tmp_path: Path):
    """
    Simulate 4 channels with different presence patterns across 3 chunks:
      - V1:TEST_START   : missing at chunk 0, present at 1,2
      - V1:TEST_MIDDLE  : present at 0,2   (missing at 1)
      - V1:TEST_END     : present at 0,1   (missing at 2)
      - V1:TEST_ALL     : present at 0,1,2 (always)
    Validate:
      - ragged dataset lengths match present chunks × samples_per_chunk
      - coverage.json present_runs reflect the schedule
      - data content equals the chunk index for each present block
    """
    # --- Arrange: channels & schedule
    channels = [
        "V1:TEST_START",
        "V1:TEST_MIDDLE",
        "V1:TEST_END",
        "V1:TEST_ALL",
    ]
    presence = {
        "V1:TEST_START":  [1, 2],
        "V1:TEST_MIDDLE": [0, 2],
        "V1:TEST_END":    [0, 1],
        "V1:TEST_ALL":    [0, 1, 2],
    }

    # --- Build mock packages that return data only for scheduled chunks
    pkgroot = tmp_path / "mockpkg"
    gwdama_pkg = pkgroot / "gwdama"
    GWDAMA_pkg = pkgroot / "GWDAMA"
    gwdama_pkg.mkdir(parents=True, exist_ok=True)
    GWDAMA_pkg.mkdir(parents=True, exist_ok=True)

    stub_io = textwrap.dedent("""
        import os, json
        import numpy as np

        def _env_json(name, default):
            s = os.environ.get(name, "")
            if not s:
                return default
            try:
                return json.loads(s)
            except Exception:
                return default

        class GwDataManager:
            @staticmethod
            def read_gwdata(*, start=None, end=None, t0=None, t1=None, channels=None,
                            ffl_spec=None, ffl_path=None, return_output=False, **kwargs):
                # Resolve interval & env
                start_gps = float(start if start is not None else t0)
                env_start = float(os.environ["GWDA_MOCK_START_GPS"])
                append    = int(os.environ.get("GWDA_MOCK_APPEND_INTERVAL", "100"))
                sched     = _env_json("GWDA_MOCK_PRESENCE", {})
                sr        = float(os.environ.get("GWDA_MOCK_SR", "1.0"))
                spc       = int(round(sr * append))

                # Chunk index from start_gps
                idx = int(round((start_gps - env_start) / append))

                data, meta = {}, {}
                for ch in (channels or []):
                    if idx in set(sched.get(ch, [])):
                        data[ch] = np.full((spc,), float(idx), dtype=np.float32)
                        meta[ch] = {"sample_rate": sr, "unit": "arb"}
                return (data, meta) if return_output else data
    """)

    # gwdama/__init__.py re-exports io.GwDataManager
    (gwdama_pkg / "__init__.py").write_text("from .io import GwDataManager\n", encoding="utf-8")
    (gwdama_pkg / "io.py").write_text(stub_io, encoding="utf-8")

    # GWDAMA/__init__.py provides the same class (covers capitalized import)
    (GWDAMA_pkg / "__init__.py").write_text(stub_io, encoding="utf-8")

    # --- Dummy FrChannels (validation shim)
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    fr = bindir / "FrChannels"
    fr.write_text("#!/usr/bin/env bash\ncat \"${GWDA_MOCK_FRCH_FILE}\"\n", encoding="utf-8")
    fr.chmod(fr.stat().st_mode | stat.S_IEXEC)

    # --- Prepare files used by the worker
    start_dt = parse_utc(TEST_START)
    end_dt   = parse_utc(TEST_END)
    ymd = start_dt.strftime("%Y%m%d")
    out_root = tmp_path / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    day_dir = out_root / start_dt.date().isoformat()
    h5_path = day_dir / f"trend_{ymd}.h5"
    coverage_path = day_dir / "coverage.json"

    # channels file for the worker
    chfile = tmp_path / "channels.txt"
    chfile.write_text("\n".join(channels), encoding="utf-8")

    # dummy FFL (one line referencing a dummy frame file)
    dummy_frame = tmp_path / "frame1.gwf"
    dummy_frame.write_text("", encoding="utf-8")
    ffl = tmp_path / "trend.ffl"
    ffl.write_text(f"{dummy_frame} 0 100\n", encoding="utf-8")

    # file that FrChannels will "cat"
    frch_file = tmp_path / "frchannels_out.txt"
    frch_file.write_text("\n".join(channels) + "\n", encoding="utf-8")

    # --- Environment for the subprocess (prepend our mocks)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pkgroot) + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["GWDA_MOCK_START_GPS"] = str(utc_to_gps(start_dt))
    env["GWDA_MOCK_APPEND_INTERVAL"] = str(APPEND_INTERVAL)
    env["GWDA_MOCK_PRESENCE"] = json.dumps(presence)
    env["GWDA_MOCK_FRCH_FILE"] = str(frch_file)
    env["GWDA_MOCK_SR"] = "1.0"

    # --- Probe: ensure our mock is the one importable in this env
    probe = subprocess.run(
        ["python", "-c",
         "import gwdama, gwdama.io; "
         "from gwdama.io import GwDataManager; "
         "print('MOCK_IMPORT_OK', hasattr(GwDataManager, 'read_gwdata'))"],
        capture_output=True, text=True, env=env
    )
    assert probe.returncode == 0 and "MOCK_IMPORT_OK True" in probe.stdout, \
        f"mock import failed:\nSTDOUT:\n{probe.stdout}\nSTDERR:\n{probe.stderr}"

    # --- Run worker for the full 3 chunks
    cmd = [
        "python", str(WORKER),
        "--start", start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "--end",   end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "--channels-file", str(chfile),
        "--ffl-path", str(ffl),
        "--ffl-spec", "V1trend",
        "--out", str(out_root),
        "--append-interval", str(APPEND_INTERVAL),
        "--log-level", "INFO",
    ]
    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("[worker stdout]\n", res.stdout)
    print("[worker stderr]\n", res.stderr)
    assert res.returncode == 0, "worker failed"

    # --- Validate H5 ragged lengths
    assert h5_path.exists(), "HDF5 not created"
    with h5py.File(h5_path, "r") as f:
        grp = f["channels"] if "channels" in f else f
        for ch in channels:
            ds = grp.get(ch)
            assert ds is not None, f"{ch} dataset missing"
            samples = int(ds.shape[0])
            expected_samples = len(presence[ch]) * APPEND_INTERVAL  # 1 Hz => 100 per chunk
            assert samples == expected_samples, f"{ch}: samples={samples} expected={expected_samples}"

    # --- Validate coverage.json present_runs against schedule
    assert coverage_path.exists(), "coverage.json missing"
    cov = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert cov["n_chunks"] == 3
    assert cov["append_interval_s"] == APPEND_INTERVAL
    for ch, present_idxs in presence.items():
        runs = cov["channels"].get(ch, {}).get("present_runs", [])
        expanded = []
        for s, ln in runs:
            expanded.extend(list(range(s, s + ln)))
        assert sorted(expanded) == sorted(present_idxs), f"{ch}: coverage {expanded} != expected {present_idxs}"

    # --- Spot-check data content (each present block filled with its chunk index)
    with h5py.File(h5_path, "r") as f:
        grp = f["channels"] if "channels" in f else f
        for ch, idxs in presence.items():
            vec = np.asarray(grp[ch][...])
            expected = np.concatenate([np.full((APPEND_INTERVAL,), float(i), dtype=vec.dtype) for i in idxs]) if idxs else np.array([], dtype=vec.dtype)
            assert vec.shape == expected.shape
            assert np.allclose(vec, expected), f"{ch}: data content mismatch"
