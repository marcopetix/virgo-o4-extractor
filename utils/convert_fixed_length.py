#!/usr/bin/env python3
"""
Convert a ragged GWDAMA-like day file + coverage.json into a fixed-length,
NaN-filled HDF5 (still GWDAMA-like: /channels/<channel>, attrs t0/sample_rate).

- Input H5 (ragged): datasets under /channels/<channel>
  attrs: channel (str), sample_rate (float), t0 (GPS float), unit (opt)
- Input JSON: coverage.json with "present_runs" per channel
- Output H5 (fixed-length): same layout/attrs, data filled to full day with NaNs

Notes:
- For 1 Hz trend, fixed length = n_chunks * append_interval_s.
- For other sample rates, fixed length = round(n_chunks * append_interval_s * sample_rate).
- For very high rates, this can be large; consider chunked write or per-channel selection.
"""

import argparse, json
from pathlib import Path
import numpy as np
import h5py
from datetime import datetime, timezone

def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%y-%m-%d_%Hh%Mm%Ss")

def _load_cov(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_group(f: h5py.File, name: str) -> h5py.Group:
    return f.require_group(name)

def convert_day_fixed_length(
    src_h5: Path,
    coverage_json: Path,
    dst_h5: Path,
    overwrite: bool = False,
    fill_value: float = np.nan,
) -> None:
    if dst_h5.exists() and not overwrite:
        raise FileExistsError(f"{dst_h5} exists. Use --overwrite to replace.")

    cov = _load_cov(coverage_json)
    n_chunks = int(cov["n_chunks"])
    append_interval_s = float(cov["append_interval_s"])
    channels_meta = cov["channels"]  # dict: ch -> {present_runs: [[start, len], ...]}

    with h5py.File(src_h5, "r") as fin, h5py.File(dst_h5, "w") as fout:
        # Root attrs (carry through + add a conversion marker)
        for k, v in fin.attrs.items():
            try:
                fout.attrs[k] = v
            except Exception:
                pass
        fout.attrs["dama_name"] = "gwf2h5_fixed"
        fout.attrs["time_stamp"] = _now_stamp()
        fout.attrs["fixed_length"] = True
        fout.attrs["source_path"] = str(src_h5)

        # Carry file-level time bounds if present
        for k in ("start_utc", "end_utc", "append_len_samples", "sample_rate_hz"):
            if k in fin.attrs:
                fout.attrs[k] = fin.attrs[k]

        # Ensure channels group
        g_out = _ensure_group(fout, "channels")
        g_in = fin.get("channels", fin)

        # Iterate channels that exist in source H5 (skip missing)
        for ch_name, obj in g_in.items():
            if not isinstance(obj, h5py.Dataset):
                continue

            src_ds: h5py.Dataset = obj
            # Read attrs
            sr = float(src_ds.attrs.get("sample_rate", 1.0))
            t0 = float(src_ds.attrs.get("t0", 0.0))
            unit = src_ds.attrs.get("unit", None)
            dtype = src_ds.dtype

            # Compute per-chunk sample count and total fixed length
            samples_per_chunk = int(round(sr * append_interval_s))
            total_samples = int(round(n_chunks * append_interval_s * sr))

            # Allocate output array (NaN filled if float, else promote to float32)
            if np.issubdtype(dtype, np.floating):
                out = np.full((total_samples,), fill_value, dtype=dtype)
                out_dtype = dtype
            else:
                out = np.full((total_samples,), np.nan, dtype=np.float32)
                out_dtype = np.float32  # we need NaN capability

            # Coverage for this channel (may be empty)
            runs = channels_meta.get(ch_name, {}).get("present_runs", [])
            # We will copy contiguous source blocks in run order
            src_cursor = 0  # current offset in the ragged source vector

            # Fast path: if no runs, leave as all-NaN
            if runs:
                # Pre-read full source vector (OK for 1 Hz; consider chunked reads for high-rate)
                src_vec = src_ds[...]
                src_len = int(src_vec.shape[0])

                for start_ix, run_len_chunks in runs:
                    # Destination absolute sample range for this run
                    dst_start = int(round(start_ix * append_interval_s * sr))
                    dst_len = int(round(run_len_chunks * append_interval_s * sr))

                    # Source absolute sample range (contiguous in ragged vector)
                    src_end = min(src_cursor + dst_len, src_len)
                    block = src_vec[src_cursor:src_end]
                    # Defensive: if source shorter than expected, trim destination too
                    copy_len = min(block.shape[0], dst_len, out.shape[0] - dst_start)
                    if copy_len > 0:
                        out[dst_start:dst_start + copy_len] = block[:copy_len]
                    src_cursor += dst_len  # advance by expected length regardless; remainder stays NaN

            # Create destination dataset and set attrs
            dso = g_out.create_dataset(
                ch_name, data=out, compression="gzip", compression_opts=1, chunks=True
            )
            dso.attrs["channel"] = ch_name
            dso.attrs["sample_rate"] = sr
            dso.attrs["t0"] = t0
            if unit is not None:
                dso.attrs["unit"] = unit

def main():
    ap = argparse.ArgumentParser(description="Convert ragged GWDAMA-like HDF5 + coverage.json to fixed-length NaN-filled HDF5.")
    ap.add_argument("--src-h5", required=True, type=Path, help="Source ragged HDF5 (GWDAMA-like)")
    ap.add_argument("--coverage", required=True, type=Path, help="coverage.json path from the day run")
    ap.add_argument("--dst-h5", required=True, type=Path, help="Destination fixed-length HDF5")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists")
    args = ap.parse_args()
    convert_day_fixed_length(args.src_h5, args.coverage, args.dst_h5, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
