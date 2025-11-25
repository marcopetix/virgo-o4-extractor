from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any, List, Dict, Optional
import logging
from utils.configs import ExtractorConfig
from datetime import datetime, timezone
import h5py
import json
import numpy as np
from gwpy.time import tconvert, to_gps
from gwdama.io import GwDataManager 
from classes.extraction_summary import ChannelsSummary, IncrementalExtractionSummary
from utils.loaders import write_json_atomic

GWF_FILE_COVERAGE_SECONDS = 100         # seconds of data covered by a single GWF file

class IncrementalDataExtractor:
    """Class to handle incremental data extraction from GWF files to HDF5 format."""

    def __init__(self, config: ExtractorConfig, channels: List[str], logger: logging.Logger):
        self.config =                       config
        self.dataset_path =                 config.dataset_path
        self.increment_size =               config.increment_size
        self.max_retries =                  config.max_retries
        self.start_gps =                    config.start_gps
        self.end_gps =                      config.end_gps
        self.channels =                     channels
        self.compression =                  config.compression if config.compression is not None else "lzf" # default to lzf if not specified

        # For logging
        self.logger = logger

        # For HDF5 handling
        self._h5: Optional[h5py.File] = None
        self._channels_grp: Optional[h5py.Group] = None
        self._dsets: Dict[str, h5py.Dataset] = {}
        self._first_written: Dict[str, bool] = {}   # to stamp t0 once
        self._dtype_by_ch: Dict[str, np.dtype] = {} # remember chosen dtype
        
        # For HDF5 handling: file attributes
        self._file_attrs = {
            "start_dt": self.config.start_dt.isoformat() if self.config.start_dt else None,
            "end_dt": self.config.end_dt.isoformat() if self.config.end_dt else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "increment_size": self.config.increment_size,
        }

        # build extraction plan
        self.extraction_plan = self._build_extraction_plan()

        # load or init day state
        self.load_or_init_day_state()

        # MARK: I need these to manage resuming and starting point
        # ---------------------------------------------------------------
        self.last_completed_chunk = int(self.state.get("last_completed_chunk", -1))
        self.total_chunks = len(self.extraction_plan)

        self.starting_chunk = self.last_completed_chunk + 1
        self.logger.debug(f"Extraction plan has {len(self.extraction_plan)} chunks; starting from chunk {self.starting_chunk}")
        # ---------------------------------------------------------------

        if self.config.produce_summary:
            # build summary with all static metadata from the config
            self.summary = IncrementalExtractionSummary(
                total_chunks=self.total_chunks,
                date=self.config.start_dt.date().isoformat(),
                start_dt_utc=self.config.start_dt.isoformat(),
                end_dt_utc=self.config.end_dt.isoformat(),
                start_gps=int(self.config.start_gps),
                end_gps=int(self.config.end_gps),
                total_seconds=int(self.config.duration),
                ffl_spec=self.config.ffl_spec,
                ffl_path=str(self.config.ffl_path),
                increment_size=int(self.config.increment_size),
                started_at_utc=self._file_attrs["created_utc"],
            )

            # presence tracking for per-channel coverage
            self.per_channel_presence: Dict[str, set[int]] = defaultdict(set) # channel -> set of gwf indices 

    # ---- lifecycle ----

    def open(self) -> None:
        """Open (or create) the HDF5 file for appending data."""
        self._h5 = h5py.File(self.dataset_path, "a")
        # stamp generic attrs
        for k, v in self._file_attrs.items():
            try:
                self._h5.attrs[k] = v
            except Exception:
                pass
        # gwdama-ish header (matches gwdama convention)
        try:
            self._h5.attrs["dama_name"] = "gwf2h5_incremental"
            self._h5.attrs["time_stamp"] = tconvert(gpsordate='now')
        except Exception:
            pass
        # group
        try:
            self._channels_grp = self._h5.require_group("channels")
        except Exception:
            self._channels_grp = None
        if self.logger:
            self.logger.info("Opened HDF5 for append: %s", self.dataset_path)

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._h5 is not None:
            try:
                self._h5.flush()
            finally:
                self._h5.close()
            if self.logger:
                self.logger.info("Closed HDF5: %s", self.dataset_path)
        self._h5 = None
        self._channels_grp = None
        self._dsets.clear()
        self._first_written.clear()
        self._dtype_by_ch.clear()
    
        # ---- internals ----

    def _parent(self) -> h5py.Group:
        """Get the parent group for datasets."""
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")
        return self._channels_grp or self._h5

    def _effective_dtype(self, ch: str, arr: Optional[np.ndarray]) -> np.dtype:
        """Determine the effective dtype for a channel, caching the first seen dtype."""
        # keep dtype as read on first sight, else reuse
        if ch in self._dtype_by_ch:
            return self._dtype_by_ch[ch]
        dt = np.asarray(arr).dtype if isinstance(arr, np.ndarray) else np.dtype("float32")
        self._dtype_by_ch[ch] = dt
        return dt

    # ---- dataset handling ----

    def ensure_dataset(
        self,
        ch_name: str,
        unit: Optional[str] = None,
        sample_rate_hz: Optional[float] = None,
        example_array: Optional[np.ndarray] = None,
    ) -> h5py.Dataset:
        """
        Ensure resizable dataset /channels/<ch_name> exists.
        On resume, if it already exists in the file, re-open it instead of creating.
        """
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")

        # In-memory cache hit
        if ch_name in self._dsets:
            return self._dsets[ch_name]

        parent = self._parent()

        # reuse existing dataset on resume
        if ch_name in parent:
            dset = parent[ch_name]
            # cache & first-written flag based on presence of 't0'
            self._dsets[ch_name] = dset
            self._first_written[ch_name] = "t0" in dset.attrs

            # backfill attrs if missing
            try:
                if "channel" not in dset.attrs:
                    dset.attrs["channel"] = ch_name
                if "sample_rate" not in dset.attrs and sample_rate_hz is not None:
                    dset.attrs["sample_rate"] = float(sample_rate_hz)
                if unit is not None and "unit" not in dset.attrs:
                    dset.attrs["unit"] = str(unit)
            except Exception:
                pass

            if self.logger:
                self.logger.debug("Reusing existing dataset for '%s' (resume)", ch_name)
            return dset
    
        # Create new dataset
        dt = self._effective_dtype(ch_name, example_array)

        if self.compression == "gzip":
            dset = parent.require_dataset(
                ch_name,
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                compression="gzip",
                chunks=(self.config.increment_size,),
            )
        else:
            dset = parent.require_dataset(
                ch_name,
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                compression="lzf",
                chunks=(self.config.increment_size,),
            )

        # Set attrs
        sr = float(sample_rate_hz if sample_rate_hz is not None else np.nan) # default NaN
        try:
            dset.attrs["channel"] = ch_name
            dset.attrs["sample_rate"] = sr
            if unit is not None:
                dset.attrs["unit"] = str(unit)
            # 't0' is stamped on first successful append
        except Exception:
            pass

        self._dsets[ch_name] = dset
        self._first_written[ch_name] = False
        return dset
    
    # ---- plan ----
    def _build_extraction_plan(self):
        """
        Build a list of (t0, t1) tuples representing extraction chunks (in GPS).
        """
        plan = []
        current_t0 = self.config.start_gps
        while current_t0 < self.config.end_gps:
            current_t1 = min(current_t0 + self.config.increment_size, self.config.end_gps)
            plan.append((current_t0, current_t1))
            current_t0 = current_t1
        self.extraction_plan = plan
        return plan
    
    # ---- read ----
    def read_chunk_with_retries(
        self,
        t0: int, # GPS start time
        t1: int, # GPS end time
    ) -> tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """
        Return (data_by_channel, meta_by_channel).
        Skips individual channels that fail; on systemic failures returns ({}, {}).
        Passes GPS floats to gwdama to avoid igwn_segments intersection issues.
        """
        attempt = 0
        # Retry loop
        while True:
            # Attempt read
            try:
                dm = GwDataManager()
                # Read data with padding for gaps
                result = dm.read_gwdata(
                    start=t0, 
                    end=t1, 
                    channels=self.channels,
                    ffl_spec=self.config.ffl_spec, 
                    ffl_path=self.config.ffl_path,
                    return_output=True, 
                    pad=np.nan, 
                    gap="pad"
                )

                # Normalize output
                data_by_ch: Dict[str, np.ndarray] = {}
                meta_by_ch: Dict[str, dict] = {}

                # gwdama returns an h5py.Group for multiple channels
                if isinstance(result, h5py.Group):
                    group = result
                    for ch in self.channels:
                        if ch not in group:
                            continue
                        dset = group[ch]
                        try:
                            arr = np.asarray(dset[...])
                        except Exception:
                            continue
                        if arr.ndim != 1:
                            continue
                        data_by_ch[ch] = arr
                        sr = dset.attrs.get("sample_rate", np.nan)
                        unit = dset.attrs.get("unit", None)
                        try:
                            sr = float(sr)
                        except Exception:
                            sr = np.nan
                        meta_by_ch[ch] = {
                            "sample_rate": sr,
                            "unit": unit,
                        }

                # Single-channel case (just in case)
                elif isinstance(result, h5py.Dataset):
                    dset = result
                    ch = dset.attrs.get("channel", None) or (self.channels[0] if self.channels else "unknown")
                    arr = np.asarray(dset[...])
                    if arr.ndim == 1:
                        data_by_ch[ch] = arr
                        sr = dset.attrs.get("sample_rate", np.nan)
                        unit = dset.attrs.get("unit", None)
                        try:
                            sr = float(sr)
                        except Exception:
                            sr = np.nan
                        meta_by_ch[ch] = {
                            "sample_rate": sr,
                            "unit": unit,
                        }

                # If nothing readable
                if not data_by_ch:
                    self.logger.warning("No data read from gwdama for [%d, %d)", t0, t1)
                return data_by_ch, meta_by_ch

            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    self.logger.warning("Read failed (no more retries): %s", e)
                    return {}, {}
                self.logger.warning("Read failed (retry %d/%d): %s", attempt, self.max_retries, e)

    # ---- append & flush ----

    def _write_block(self, dset: h5py.Dataset, block: np.ndarray) -> int:
        """Append a block of data to the dataset, resizing as needed."""
        old = int(dset.shape[0])
        n = int(block.shape[0])
        dset.resize((old + n,))
        dset[old:old + n] = block.astype(dset.dtype, copy=False)
        return n
    
    def append_chunk(
        self,
        data_by_ch: Dict[str, np.ndarray],
        meta_by_ch: Optional[Dict[str, Dict[str, Any]]] = None,
        chunk_t0_utc: Optional[datetime] = None,
    ) -> None:
        """
        Append one window for channels present in data_by_ch (with NaN padding).
        Returns per-channel written sample counts.
        """
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")

        written: Dict[str, int] = {}

        for ch, arr in data_by_ch.items():
            if not isinstance(arr, np.ndarray):
                continue # skip invalid
            md = (meta_by_ch or {}).get(ch, {}) # per-channel metadata
            sr = float(md.get("sample_rate", None)) # default sample rate
            unit = md.get("unit", None) # optional unit

            # ensure dataset
            dset = self.ensure_dataset(
                ch_name=ch,
                unit=unit,
                sample_rate_hz=sr,
                example_array=arr,
            )

            # stamp t0 on FIRST successful write only
            if not self._first_written.get(ch, False) and chunk_t0_utc is not None:
                if "t0" not in dset.attrs:  # double check
                    dset.attrs["t0"] = float(to_gps(chunk_t0_utc)) 
                self._first_written[ch] = True 

            # write block
            block = np.asarray(arr)
            written[ch] = self._write_block(dset, block)

            # This could be used for logging per-channel appends
            # return written

    def flush(self) -> None:
        """Flush HDF5 file to disk."""
        if self._h5 is not None:
            self._h5.flush()
    
    # ---- state ----
    def load_or_init_day_state(self) -> None:
        """
        Create a minimal day state if missing; otherwise load existing.
        Schema:
        {
            "date": "YYYY-MM-DD",
            "h5_path": ".../trend_YYYYmmdd.h5",
            "increment_size": 100,
            "last_completed_chunk": -1,
            "status": "running|completed|failed",
            "error": null
        }
        """
        # if existing, load and return
        if self.config.state_path.is_file():
            with self.config.state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            self.state = state
            return

        # otherwise initialize new state
        self.state = {
            "date": self.config.start_dt.date().isoformat(),
            "h5_path": str(self.config.dataset_path),
            "increment_size": int(self.config.increment_size),
            "last_completed_chunk": -1,
            "status": "running",
            "error": None,
        }

        write_json_atomic(self.config.state_path, self.state)

    def update_day_state(self, **fields) -> None:
        self.state.update(fields)
        write_json_atomic(self.config.state_path, self.state)

    # ---- resume ----
    def check_resume(self) -> None:
        """Check and handle resuming from previous runs."""
        # Handle resuming from previous runs
        if self.config.resume:
            if self.state.get("status") == "completed" and self.last_completed_chunk >= self.total_chunks - 1:
                self.logger.info("Day already completed for current plan: %s", self.config.day_folder)
                sys.exit(0)

    def identify_start_chunk(self) -> None:
        """Identify current starting time for extraction."""
        # Identify current starting time for extraction
        self.starting_chunk = 0 if not self.config.resume else (self.last_completed_chunk + 1)
        if self.starting_chunk >= self.total_chunks:
            self.logger.info("Nothing to do: start_idx=%d >= total_chunks=%d", self.starting_chunk, self.total_chunks)
            sys.exit(0)

        self.logger.info("Resume: last_completed_chunk=%d, starting_from=%d, total_chunks=%d",
                self.last_completed_chunk, self.starting_chunk, self.total_chunks)

    # ---- main loop ----

    def run_extraction(self) -> None:
        """
        Run the incremental extraction from start_chunk to end.
        Updates day state as it progresses.
        """

        # check HDF5 is open
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")
        
        # check resume state
        self.check_resume()

        # identify starting chunk
        self.identify_start_chunk()

        for chunk_idx in range(self.starting_chunk, self.total_chunks):
            t0, t1 = self.extraction_plan[chunk_idx]

            # convert to UTC datetime for stamping
            t0_dt = datetime.fromtimestamp(t0, tz=timezone.utc)
            t1_dt = datetime.fromtimestamp(t1, tz=timezone.utc)

            # Read chunks of data with retry (returns channel-wise data and stats)
            data_by_ch, meta_by_ch = self.read_chunk_with_retries(t0, t1)
            
            # Check for empty read
            if not data_by_ch:
                # No data read for this chunk -> log and continue (still update state to skip it next time)
                self.logger.warning("No data read for chunk %d (%s to %s)", chunk_idx, t0_dt, t1_dt)
                self.update_day_state(last_completed_chunk=chunk_idx, status="running")
                self.last_completed_chunk = chunk_idx
                continue

            # Append data
            self.append_chunk(
                data_by_ch=data_by_ch,
                meta_by_ch=meta_by_ch,
                chunk_t0_utc=t0_dt,
            )

            # Flush to disk
            self.flush()

            # Update day state
            self.update_day_state(
                last_completed_chunk=chunk_idx,
                status="running",
            )

            self.logger.info("Completed chunk %d/%d (%s to %s)", chunk_idx + 1, self.total_chunks, t0_dt, t1_dt)

            # Update presence tracking for summary (this works on the gwf files level)
            if self.config.produce_summary:
                self.compute_gwf_presence(t0, data_by_ch, meta_by_ch)
        
        self.flush()

        # prune empty channels
        self.prune_empty_channels()

        if self.config.produce_summary:
            # 1) general summary JSON in the main day folder
            self.build_general_summary(self.config.summary_path)

            # 2) channels_info folder with txt lists + coverage json
            channels_info_dir = self.config.day_folder / "channels_info"
            coverage_path = channels_info_dir / "channels_coverage.json"
            self.build_channel_summaries(coverage_path)
            self.write_channel_lists(channels_info_dir)

        self.logger.info("Extraction completed: all %d chunks processed", self.total_chunks)

    # ---- extra operations ----

    def prune_empty_channels(self, threshold: int = 0) -> None:
        """
        Prune channels with total samples <= threshold.
        Returns list of pruned channel names.
        """
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")
        
        pruned: List[str] = []

        for ch, dset in list(self._dsets.items()):
            total_samples = int(dset.shape[0])
            if total_samples <= threshold:
                # delete dataset
                del self._parent()[ch]
                pruned.append(ch)
                del self._dsets[ch]
                del self._first_written[ch]
                del self._dtype_by_ch[ch]
                if self.logger:
                    self.logger.info("Pruned empty channel '%s' (samples=%d)", ch, total_samples)

        if self.config.produce_summary:
            # update summary (channels info)
            self.summary.pruned_channels = len(pruned)
            self.summary.pruned_channels_list.extend(pruned)

    def compute_gwf_presence(self, t0, data_by_ch, meta_by_ch) -> None:
        gwf_index = (t0 - self.start_gps) // GWF_FILE_COVERAGE_SECONDS
        for ch, arr in data_by_ch.items():
            channel_sample_rate = meta_by_ch.get(ch, {}).get("sample_rate", None)

            if (
                channel_sample_rate is None
                or (isinstance(channel_sample_rate, float) and not math.isfinite(channel_sample_rate))
                or channel_sample_rate <= 0
            ):
                samples_per_gwf = 0
            else:
                samples_per_gwf = int(channel_sample_rate * GWF_FILE_COVERAGE_SECONDS)

            samples_read = len(arr)
            n_gwf_files_covered = samples_read // samples_per_gwf if samples_per_gwf > 0 else 0

            for i in range(n_gwf_files_covered):
                start = i * samples_per_gwf
                end = start + samples_per_gwf
                segment = arr[start:end]
                if not np.all(np.isnan(segment)):
                    self.per_channel_presence[ch].add(gwf_index + i)


    # ---- summary ----

    def build_general_summary(self, summary_path=None) -> Dict[str, Any]:
        """Build and optionally write the extraction summary to a JSON file."""
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")
        
        # update general info
        now_utc = datetime.now(timezone.utc)
        self.summary.completed_at_utc = now_utc.isoformat()
        self.summary.elapsed_time_seconds = (
            now_utc - datetime.fromisoformat(self.summary.started_at_utc)
        ).total_seconds()
        self.summary.dataset_size_bytes = self._h5.id.get_filesize()

        # update channels info
        self.summary.requested_channels = len(self.channels)
        self.summary.requested_channels_list = list(self.channels)
        self.summary.valid_channels = len(self._dsets)
        self.summary.valid_channels_list = list(self._dsets.keys())
        # invalid_channels/pruned_channels already maintained elsewhere

        # write to file if requested
        if summary_path is not None:
            try:
                payload = self.summary.to_dict_general()
                # safer write
                write_json_atomic(Path(summary_path), payload)
                if self.logger:
                    self.logger.info("Wrote summary to %s", summary_path)
            except Exception as e:
                if self.logger:
                    self.logger.error("Failed to write summary to %s: %s", summary_path, e)

        return self.summary.to_dict_general()

    def build_channel_summaries(self, summary_path=None) -> Dict[str, Dict[str, Any]]:
        """Build per-channel coverage summaries."""
        if self._h5 is None:
            raise RuntimeError("HDF5 file is not open")

        summaries: Dict[str, Dict[str, Any]] = {}

        # Number of GWF slots over the day
        num_gwf = 0
        if self.summary.total_seconds is not None:
            num_gwf = int(math.ceil(self.summary.total_seconds / GWF_FILE_COVERAGE_SECONDS))

        for ch, dset in self._dsets.items():
            data = dset[...]  # full array
            total_samples = int(data.shape[0])
            nan_samples = int(np.count_nonzero(np.isnan(data)))
            non_nan_samples = total_samples - nan_samples
            nan_fraction = (nan_samples / total_samples) if total_samples > 0 else 1.0

            ch_summary = ChannelsSummary(
                channel_name=ch,
                sample_rate=float(dset.attrs.get("sample_rate", np.nan)),
                dtype=str(dset.dtype),
                total_samples=total_samples,
                nan_samples=nan_samples,
                non_nan_samples=non_nan_samples,
                nan_fraction=nan_fraction,
            )

            # presence indices recorded during extraction (may be missing if no summary)
            presence = set()
            if hasattr(self, "per_channel_presence"):
                presence = self.per_channel_presence.get(ch, set())

            ch_summary.calculate_gaps_and_missing(
                presence_indices=presence,
                number_of_gwf=num_gwf,
            )

            summaries[ch] = ch_summary.to_dict()

        if summary_path is not None:
            try:
                write_json_atomic(Path(summary_path), summaries)
                if self.logger:
                    self.logger.info("Wrote channel summaries to %s", summary_path)
            except Exception as e:
                if self.logger:
                    self.logger.error("Failed to write channel summaries to %s: %s", summary_path, e)

        return summaries
    
    def write_channel_lists(self, channels_info_dir: Path) -> None:
        """Write requested/valid/invalid/pruned channel lists as .txt files."""
        channels_info_dir.mkdir(parents=True, exist_ok=True)

        def _write_list(path: Path, items: List[str]) -> None:
            with path.open("w", encoding="utf-8") as f:
                for ch in items:
                    f.write(f"{ch}\n")

        _write_list(channels_info_dir / "requested_channels.txt",
                    self.summary.requested_channels_list)
        _write_list(channels_info_dir / "valid_channels.txt",
                    self.summary.valid_channels_list)
        _write_list(channels_info_dir / "invalid_channels.txt",
                    self.summary.invalid_channels_list)
        _write_list(channels_info_dir / "pruned_channels.txt",
                    self.summary.pruned_channels_list)
