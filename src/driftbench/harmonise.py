"""Phase 1 -- feature and label harmonisation.

Two feature tracks:
  * native      -- each dataset's own cleaned feature set (within-dataset, Demo A)
  * common-core -- the intersection of canonical CICFlowMeter features present
                   in both datasets (cross-dataset transfer, Demo B)

Two label levels:
  * native      -- the dataset's own labels, preserved
  * family      -- a broad harmonised taxonomy shared across datasets
"""
from __future__ import annotations
from typing import List, Sequence
import pandas as pd

# Canonical CICFlowMeter features (normalised names per cleaning.normalise_columns).
# The common core is the intersection of THIS list with the columns actually
# present in both cleaned datasets, so it is robust to per-release naming drift.
CANONICAL_CORE: List[str] = [
    "flow_duration",
    "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean", "fwd_packet_length_std",
    "bwd_packet_length_max", "bwd_packet_length_min", "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes/s", "flow_packets/s",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fwd_header_length", "bwd_header_length",
    "fwd_packets/s", "bwd_packets/s",
    "min_packet_length", "max_packet_length", "packet_length_mean",
    "packet_length_std", "packet_length_variance",
    "fin_flag_count", "syn_flag_count", "rst_flag_count", "psh_flag_count",
    "ack_flag_count", "urg_flag_count", "cwe_flag_count", "ece_flag_count",
    "down/up_ratio", "average_packet_size",
    "subflow_fwd_packets", "subflow_fwd_bytes", "subflow_bwd_packets", "subflow_bwd_bytes",
    "init_win_bytes_forward", "init_win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
]


def common_core(*feature_frames: pd.DataFrame) -> List[str]:
    """Return canonical-core features present across all given frames, in order."""
    present = set.intersection(*[set(f.columns) for f in feature_frames]) if feature_frames else set()
    return [c for c in CANONICAL_CORE if c in present]


def to_common_core(df: pd.DataFrame, core: Sequence[str]) -> pd.DataFrame:
    """Select the agreed common-core columns (assumes names already normalised)."""
    return df.loc[:, list(core)].copy()


# --- Label harmonisation -----------------------------------------------------
# Substring rules, evaluated top-to-bottom on the lowercased native label.
_FAMILY_RULES = [
    ("benign", ("benign", "normal")),
    ("ddos", ("ddos", "drdos", "ldap", "mssql", "netbios", "ntp", "snmp",
              "ssdp", "udplag", "syn", "tftp", "udp")),  # 2019 reflection/exploitation families
    ("dos", ("dos",)),                                   # plain DoS (after ddos rule)
    ("botnet", ("bot", "mirai")),
    ("bruteforce", ("bruteforce", "brute-force", "brute_force", "ftp", "ssh", "password")),
    ("web", ("web", "xss", "sql")),
    ("infiltration", ("infiltration", "infilteration")),
    ("recon", ("portscan", "port_scan", "scan", "recon", "heartbleed")),
    ("injection", ("injection", "exploit")),
]


def _family_of(label: str) -> str:
    s = str(label).strip().lower()
    for fam, toks in _FAMILY_RULES:
        if any(t in s for t in toks):
            return fam
    return "other"


def harmonise_labels(labels: pd.Series) -> pd.Series:
    """Map native labels to the broad family taxonomy (Series of strings)."""
    return labels.map(_family_of).astype("category")


def shared_families(labels_a: pd.Series, labels_b: pd.Series) -> List[str]:
    """Families present in BOTH label sets -- the valid cross-dataset label space."""
    fa = set(harmonise_labels(labels_a).unique())
    fb = set(harmonise_labels(labels_b).unique())
    return sorted(fa & fb)
