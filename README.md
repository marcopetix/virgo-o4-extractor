# **SAT Dataset Creator**

### Incremental and Parallel GWF → HDF5 Conversion for Virgo O4 Trend & Raw Data

This repository provides a **robust, resumable, parallelizable pipeline** for converting Virgo **GWF** (Gravitational Wave Frame) files into structured **daily HDF5 datasets**.

It supports:

* Incremental extraction with **chunked reading**
* Automatic **padding of gaps** with NaN
* Per-channel metadata (sample rate, units, t0)
* Channel pruning & validation
* Per-day **coverage**, **summary**, and **statistics**
* A **parallel scheduler** to process entire O4 ranges
* Full resume capabilities

The system is based on the Virgo data management ecosystem (**gwdama**, **GWPy**) and reflects the internal naming and data conventions of the Virgo collaboration.

---

# **1. Features**

### ✔️ Incremental single-day extraction (`incremental_conversion.py`)

* Converts a single day of GWF data into `channels/<channel>.h5`
* Reads in configurable increments (`increment_size`, default: 600 s)
* Resumes automatically from `state_day.json`
* Produces:

  * Main HDF5 dataset (`V1trend_YYYYMMDD.h5`)
  * General summary JSON
  * Detailed per-channel coverage summary
  * Lists of valid/invalid/pruned channels
  * Log file + structured metadata

### ✔️ Parallel multi-day launcher (`parallel_conversion.py`)

* Spawns workers (one per day) with controlled concurrency
* Retries days on failure
* Supports smoke-test mode:

  * `limit_days = N`
  * `minutes_per_day = M`
* Maintains a global `index.json` of the entire run

### ✔️ Robust data handling

* Handles missing data with NaN padding
* Uses `gwdama` for reading
* Ensures dtype consistency
* Tracks per-channel sample rate, dtype, NaN fraction, and coverage over GWF files
* Atomically writes all JSON files
* Full logging (stdout + per-day logs)

---

# **2. Installation**

You need:

* Python 3.10+
* Virgo IGWN environment (or equivalent)
* `gwdama` + `gwpy` available in PATH
* HDF5 libraries (`h5py`)

Install Python packages (example):

```bash
pip install h5py gwpy numpy tomli
```

Clone the repository:

```bash
git clone <your_gitlab_or_github_url>
cd virgo-o4-extractor
```

---

# **3. Quickstart**

## **A. Run a single-day extraction (incremental worker)**

You can run the worker using a TOML config file or CLI.

### **Using a TOML config file**

Create `configs/o4_worker.toml`:

```toml
start_dt = "2024-04-10"
end_dt   = "2024-04-11"

channels_file = "channels/o4_channels.txt"
ffl_spec      = "V1trend"
ffl_path      = "/virgoData/ffl/trend.ffl"

out_root = "/data/procdata/rcsDatasets/OriginalSR/o4_trend_h5"

resume = true
log_level = "INFO"

increment_size = 600
max_retries = 3
compression = "lzf"
produce_summary = true
```

Run:

```bash
python -m scripts.incremental_conversion -c configs/o4_worker.toml
```

### **Using CLI**

```bash
python -m scripts.incremental_conversion \
    --start-dt "2024-04-10 00:00:00" \
    --end-dt   "2024-04-11 00:00:00" \
    --channels-file channels/o4_channels.txt \
    --ffl-path /virgoData/ffl/trend.ffl \
    --ffl-spec V1trend \
    --out /data/procdata/rcsDatasets/OriginalSR/o4_trend_h5 \
    --increment-size 600 \
    --resume
```

---

## **B. Run a multi-day parallel extraction**

Use the provided test config:


Example:

```bash
python -m scripts.parallel_conversion -c configs/o4_parallel_test.toml
```

A full-range extraction:

```toml
start_date = "2024-04-10"
end_date   = "2025-11-18"

channels_file = "channels/o4_channels.txt"
ffl_spec      = "V1trend"
ffl_path      = "/virgoData/ffl/trend.ffl"

out_dir = "/data/procdata/rcsDatasets/OriginalSR/o4_trend_h5"

resume = true
log_level = "INFO"

increment_size = 600
concurrency = 8
stagger_seconds = 5
max_retries = 3
```

Run:

```bash
python -m scripts.parallel_conversion -c configs/o4_parallel.toml
```

This will spawn up to N concurrent workers, each writing into:

```
out_dir/V1trend-YYYY-MM-DD/
```

---

# **4. Output Structure**

For each day, the worker produces a folder:

```
out_root/V1trend-2024-04-10/
```

Contents:

```
V1trend-2024-04-10/
 ├─ V1trend_20240410.h5          # daily dataset
 ├─ worker.log                   # detailed log (UTC)
 ├─ state_day.json               # resume state
 ├─ V1trend_20240410.summary.json
 └─ channels_info/
     ├─ requested_channels.txt
     ├─ valid_channels.txt
     ├─ invalid_channels.txt
     ├─ pruned_channels.txt
     └─ channels_coverage.json
```

---

# **5. HDF5 Dataset Structure**

The main file (e.g. `V1trend_20240410.h5`) contains:

```
/channels/<channel_name> : 1-D float32/float64 array
```

Attributes per dataset:

* `channel`
* `sample_rate`
* `unit`
* `t0` (GPS of first sample)

File-level attributes include start/end time, creation timestamp, increment size, etc.
(see IncrementalDataExtractor.open() in )

---

# **6. Summary Files**

### **General summary**

Location:

```
V1trend-YYYY-MM-DD/V1trend_YYYYMMDD.summary.json
```

Generated by: `build_general_summary()`


Contains:

* Time window (UTC/GPS)
* Number of chunks
* Runtime duration
* Dataset size
* Valid/invalid/pruned channels
* FFL info
* Increment size

### **Channel coverage summary**

Generated by: `build_channel_summaries()`


Includes for each channel:

* sample rate
* dtype
* total samples
* NaN fraction
* GWF coverage mask (`"101011..."`)
* Missing fraction
* Number of gaps