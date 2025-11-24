from collections.abc import Set
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.loaders import write_json_atomic
import numpy as np

@dataclass
class ChannelsSummary:
    """
    Stats for a single channel throughout extraction and coverage over GWF files.
    """
    channel_name: str
    sample_rate: Optional[float] = None
    dtype: Optional[str] = None

    # sample-level stats
    total_samples: int = 0
    nan_samples: int = 0
    non_nan_samples: int = 0
    nan_fraction: float = 0.0  # nan_samples / total_samples

    # GWF-level coverage
    gwf_gaps: int = 0                 # number of missing gwf slots (zeros in mask)
    gwf_mask: str = ""                # bitstring over GWF files, "1" = present, "0" = missing
    missing_fraction: float = 0.0     # gwf_gaps / number_of_gwf

    def calculate_gaps_and_missing(
        self,
        presence_indices: Set[int],
        number_of_gwf: int,
    ) -> None:
        """
        Calculate GWF-level coverage from the set of indices where this channel
        had any non-NaN data.

        presence_indices: set of gwf indices with data (0-based)
        number_of_gwf:    total number of gwf "slots" over the day
        """
        if number_of_gwf <= 0:
            self.gwf_mask = ""
            self.gwf_gaps = 0
            self.missing_fraction = 1.0
            return

        mask_bits = ["0"] * number_of_gwf
        for i in presence_indices:
            if 0 <= i < number_of_gwf:
                mask_bits[i] = "1"

        self.gwf_mask = "".join(mask_bits)
        self.gwf_gaps = self.gwf_mask.count("0")
        self.missing_fraction = self.gwf_gaps / float(number_of_gwf)

    def to_dict(self) -> Dict:
        """
        JSON-serializable representation for channels_coverage.json.
        """
        return asdict(self)
    
class IncrementalExtractionSummary:
    """Class to summarize the incremental extraction process."""
    def __init__(self, **kwargs):
        
        # Time window info
        self.date: str = None
        self.start_dt_utc: str = None
        self.end_dt_utc: str = None
        self.start_gps: int = None
        self.end_gps: int = None
        self.total_seconds: float = None

        # ffl info
        self.ffl_spec: str = None
        self.ffl_path: str = None

        # Incremental extraction info
        self.increment_size: int = None
        self.total_chunks: int = None
        self.started_at_utc: str = None
        self.completed_at_utc: str = None
        self.elapsed_time_seconds: float = 0.0
        self.dataset_size_bytes: int = 0

        # Channels info
        self.requested_channels: int = 0
        self.requested_channels_list: list[str] = []
        self.valid_channels: int = 0
        self.valid_channels_list: list[str] = []
        self.invalid_channels: int = 0
        self.invalid_channels_list: list[str] = []
        self.pruned_channels: int = 0
        self.pruned_channels_list: list[str] = []
        self.channels_summary: dict[str, ChannelsSummary] = {}

        # Apply any fields passed from the worker
        self.update_summary_general(**kwargs)

        
    # update methods
    def update_summary_general(self, **kwargs):
        """Update general summary attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def update_channel_summary(self, channel_name: str, **kwargs):
        """Update per-channel summary attributes."""
        if channel_name not in self.channels_summary:
            self.channels_summary[channel_name] = ChannelsSummary(channel_name=channel_name)
        ch_summary = self.channels_summary[channel_name]
        for key, value in kwargs.items():
            if hasattr(ch_summary, key):
                setattr(ch_summary, key, value)
    
    def to_dict_general(self) -> dict:
        return {
            "date": self.date,
            "start_dt_utc": self.start_dt_utc,
            "end_dt_utc": self.end_dt_utc,
            "start_gps": self.start_gps,
            "end_gps": self.end_gps,
            "total_seconds": self.total_seconds,
            "ffl_spec": self.ffl_spec,
            "ffl_path": self.ffl_path,
            "increment_size": self.increment_size,
            "total_chunks": self.total_chunks,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "dataset_size_bytes": self.dataset_size_bytes,
            "requested_channels": self.requested_channels,
            "valid_channels": self.valid_channels,
            "invalid_channels": self.invalid_channels,
            "pruned_channels": self.pruned_channels,
        }

    def to_dict(self) -> dict:
        # for backwards compatibility with build_general_summary()
        return self.to_dict_general()