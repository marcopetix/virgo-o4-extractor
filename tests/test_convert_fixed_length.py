# tests/test_convert_fixed_length.py
import json
from pathlib import Path
import numpy as np
import h5py
import pytest

from utils.convert_fixed_length import convert_day_fixed_length  

def write_ragged_h5(path: Path, channels, sample_rates, t0s, units=None, root_attrs=None):
    """
    Create a ragged, GWDAMA-like HDF5 with /channels/<ch> datasets (1D).
    Data is the exact concatenation order that coverage 'present_runs' expects.
    """
    with h5py.File(path, "w") as f:
        if root_attrs:
            for k, v in root_attrs.items():
                f.attrs[k] = v
        g = f.require_group("channels")
        for ch, vec in channels.items():
            d = g.create_dataset(ch, data=np.asarray(vec))
            d.attrs["channel"] = ch
            d.attrs["sample_rate"] = float(sample_rates[ch])
            d.attrs["t0"] = float(t0s[ch])
            if units and ch in units:
                d.attrs["unit"] = units[ch]

def write_cov_json(path: Path, *, n_chunks, append_interval_s, runs_by_channel):
    """
    runs_by_channel: dict { ch: [[start_chunk, run_len_chunks], ...] }
    """
    cov = {
        "n_chunks": int(n_chunks),
        "append_interval_s": float(append_interval_s),
        "channels": {ch: {"present_runs": runs} for ch, runs in runs_by_channel.items()},
    }
    path.write_text(json.dumps(cov), encoding="utf-8")

@pytest.fixture
def tmpdir(tmp_path: Path):
    # Helper names
    return {
        "src": tmp_path / "src.h5",
        "dst": tmp_path / "dst.h5",
        "cov": tmp_path / "coverage.json",
        "tmp": tmp_path,
    }

def test_basic_gap_filling_1hz(tmpdir):
    # Day layout: n_chunks=4, append=10s => total length = 40 samples @1Hz
    # Runs for chA: chunks [0] (10s) and [2,3] (20s). Gap is chunk [1] (10s) => NaNs.
    n_chunks, append = 4, 10.0
    sr = {"chA": 1.0}
    t0 = {"chA": 0.0}

    # Ragged source vector is the concatenation of present runs: length = 10 + 20 = 30
    run1 = np.arange(10)              # chunk 0
    run2 = 100 + np.arange(20)        # chunks 2-3
    ragged = {"chA": np.concatenate([run1, run2])}

    write_ragged_h5(tmpdir["src"], ragged, sr, t0)
    write_cov_json(tmpdir["cov"], n_chunks=n_chunks, append_interval_s=append,
                   runs_by_channel={"chA": [[0, 1], [2, 2]]})

    convert_day_fixed_length(tmpdir["src"], tmpdir["cov"], tmpdir["dst"], overwrite=True)

    with h5py.File(tmpdir["dst"], "r") as f:
        out = f["channels/chA"][...]
        assert out.shape == (40,)  # 4*10 @ 1 Hz
        # chunk 0 filled with run1
        np.testing.assert_array_equal(out[0:10], run1)
        # chunk 1 should be NaNs
        assert np.all(np.isnan(out[10:20]))
        # chunks 2-3 filled with run2
        np.testing.assert_array_equal(out[20:40], run2)
        # attrs are preserved/added
        assert f["channels/chA"].attrs["sample_rate"] == 1.0  # from source
        assert "t0" in f["channels/chA"].attrs                # from source
        assert f.attrs["fixed_length"] == True                # added by converter
        assert "source_path" in f.attrs                       # added by converter
        # gzip/chunked write is used (implementation detail)
        assert f["channels/chA"].chunks is not None           # chunked dataset

def test_mapping_with_non_unit_sr(tmpdir):
    # sr=2 Hz, n_chunks=3, append=5s => total samples = 3*5*2 = 30
    # Single run in the middle chunk only (start=1,len=1) -> samples [10:20] filled
    n_chunks, append = 3, 5.0
    sr = {"chB": 2.0}
    t0 = {"chB": 123.0}
    middle = np.arange(10)  # length = run_len(1) * append(5s) * sr(2) = 10 samples
    write_ragged_h5(tmpdir["src"], {"chB": middle}, sr, t0, units={"chB": "arb"})
    write_cov_json(tmpdir["cov"], n_chunks=n_chunks, append_interval_s=append,
                   runs_by_channel={"chB": [[1, 1]]})

    convert_day_fixed_length(tmpdir["src"], tmpdir["cov"], tmpdir["dst"], overwrite=True)

    with h5py.File(tmpdir["dst"], "r") as f:
        out = f["channels/chB"][...]
        assert out.shape == (30,)
        assert np.all(np.isnan(out[0:10]))
        np.testing.assert_array_equal(out[10:20], middle)   # middle chunk
        assert np.all(np.isnan(out[20:30]))
        # attribute carry-through
        assert f["channels/chB"].attrs["t0"] == 123.0
        assert f["channels/chB"].attrs["unit"] == "arb"

def test_dtype_promotion_for_non_float(tmpdir):
    # int16 source should produce float32 output to allow NaNs
    n_chunks, append = 2, 3.0
    sr = {"chI": 1.0}
    t0 = {"chI": 0.0}
    data = np.arange(3, dtype=np.int16)  # run length = 1 chunk @ 1Hz*3s
    write_ragged_h5(tmpdir["src"], {"chI": data}, sr, t0)
    write_cov_json(tmpdir["cov"], n_chunks=n_chunks, append_interval_s=append,
                   runs_by_channel={"chI": [[0, 1]]})

    convert_day_fixed_length(tmpdir["src"], tmpdir["cov"], tmpdir["dst"], overwrite=True)

    with h5py.File(tmpdir["dst"], "r") as f:
        dso = f["channels/chI"]
        assert dso.dtype == np.float32
        out = dso[...]
        np.testing.assert_array_equal(out[0:3], data.astype(np.float32))
        assert np.all(np.isnan(out[3:6]))  # second chunk is NaN gap

def test_no_runs_produces_all_nans(tmpdir):
    n_chunks, append = 2, 4.0
    sr = {"silent": 1.0}
    t0 = {"silent": 0.0}
    # Ragged file contains a dataset but coverage has no present_runs
    write_ragged_h5(tmpdir["src"], {"silent": np.array([], dtype=np.float32)}, sr, t0)
    write_cov_json(tmpdir["cov"], n_chunks=n_chunks, append_interval_s=append,
                   runs_by_channel={"silent": []})

    convert_day_fixed_length(tmpdir["src"], tmpdir["cov"], tmpdir["dst"], overwrite=True)

    with h5py.File(tmpdir["dst"], "r") as f:
        out = f["channels/silent"][...]
        assert out.shape == (8,)  # 2*4 @ 1Hz
        assert np.all(np.isnan(out))

def test_dst_exists_without_overwrite_raises(tmpdir):
    # Prepare any small setup
    write_ragged_h5(tmpdir["src"], {"x": np.array([1,2,3])}, {"x": 1.0}, {"x": 0.0})
    write_cov_json(tmpdir["cov"], n_chunks=1, append_interval_s=3.0,
                   runs_by_channel={"x": [[0,1]]})
    # Create an empty file at dst
    tmpdir["dst"].write_bytes(b"")

    with pytest.raises(FileExistsError):
        convert_day_fixed_length(tmpdir["src"], tmpdir["cov"], tmpdir["dst"], overwrite=False)

def test_truncate_if_source_shorter_than_expected(tmpdir):
    # Coverage claims 2 chunks of 5s at 1Hz (expects 10 samples),
    # but source has only 8 samples; last 2 should remain NaN.
    write_ragged_h5(tmpdir["src"], {"y": np.arange(8)}, {"y": 1.0}, {"y": 0.0})
    write_cov_json(tmpdir["cov"], n_chunks=2, append_interval_s=5.0,
                   runs_by_channel={"y": [[0, 2]]})

    convert_day_fixed_length(tmpdir["src"], tmpdir["cov"], tmpdir["dst"], overwrite=True)

    with h5py.File(tmpdir["dst"], "r") as f:
        out = f["channels/y"][...]
        assert out.shape == (10,)
        np.testing.assert_array_equal(out[:8], np.arange(8))
        assert np.all(np.isnan(out[8:10]))
