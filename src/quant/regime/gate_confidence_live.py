from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _env(name: str, default: str) -> str:
    return str(os.getenv(name, default))


def _to_utc_ns(values: Any) -> pd.Series:
    """Normalize timestamps to a consistent dtype for merge_asof."""
    s = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(s, pd.DatetimeIndex):
        s = pd.Series(s)
    elif not isinstance(s, pd.Series):
        s = pd.Series(s)
    # Round-trip through string to force unified datetime64[ns, UTC].
    return pd.to_datetime(s.astype("string"), utc=True, errors="coerce")


def _parse_horizons(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out or [5, 30, 120, 240]


def _wilson_ci(n_success: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p_hat = n_success / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _latest_now_from_voxel_map(path: Path) -> Tuple[pd.Timestamp, int]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        last_idx = max(0, pf.num_row_groups - 1)
        tbl = pf.read_row_group(last_idx, columns=["ts", "voxel_id"])
        pdf = tbl.to_pandas()
        if pdf.empty:
            raise ValueError("empty last row group")
        row = pdf.tail(1).iloc[0]
        ts = pd.to_datetime(row["ts"], utc=True, errors="coerce")
        if pd.isna(ts):
            raise ValueError("invalid ts in last row group")
        return (ts, int(row["voxel_id"]))
    except Exception:
        df = pd.read_parquet(path, columns=["ts", "voxel_id"])
        if df.empty:
            raise ValueError("empty voxel_map parquet")
        row = df.tail(1).iloc[0]
        ts = pd.to_datetime(row["ts"], utc=True, errors="coerce")
        if pd.isna(ts):
            raise ValueError("invalid ts in voxel_map parquet")
        return (ts, int(row["voxel_id"]))


@dataclass
class _EngineState:
    voxel_map_ts: np.ndarray
    voxel_map_vid: np.ndarray
    all_vids: np.ndarray
    id2idx: Dict[int, int]
    adj_voxel: List[List[Tuple[int, float]]]
    gate_vec_voxel: np.ndarray
    voxel_gate_rate: Dict[int, float]
    voxel_gate_n: Dict[int, int]
    voxel_gate_ci: Dict[int, Tuple[float, float]]
    basins_map: Dict[int, int]
    basin_ids: np.ndarray
    bid2idx: Dict[int, int]
    adj_basin: List[List[Tuple[int, float]]]
    gate_vec_basin: np.ndarray
    basin_gate_rate: Dict[int, float]
    basins_stats: Dict[int, Dict[str, float]]
    artifacts_mtime: Dict[str, float]
    daily_gate_mtime: float


class GateConfidenceLiveEngine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_refresh = 0.0
        self.state: Optional[_EngineState] = None
        self.cache_key = _env("GATE_CONF_CACHE_KEY", "default")

    def _paths(self) -> Dict[str, Path]:
        art_dir = Path(_env("GATE_CONF_ARTIFACT_DIR", ""))
        return {
            "artifact_dir": art_dir,
            "voxel_map": art_dir / "voxel_map.parquet",
            "voxel_stats": art_dir / "voxel_stats.parquet",
            "transitions_topk": art_dir / "transitions_topk.parquet",
            "basins": art_dir / "basins_v02_components.parquet",
            "gate_daily": Path(_env("GATE_DAILY_PATH", "")),
        }

    def _mtime(self, p: Path) -> float:
        try:
            return float(p.stat().st_mtime)
        except Exception:
            return -1.0

    def _needs_refresh(self) -> bool:
        cache_sec = max(0.0, float(_env("GATE_CONF_CACHE_SEC", "30")))
        now = time.time()
        if self.state is None:
            return True
        if now - self.last_refresh < cache_sec:
            return False

        p = self._paths()
        current_art = {
            "voxel_map": self._mtime(p["voxel_map"]),
            "voxel_stats": self._mtime(p["voxel_stats"]),
            "transitions_topk": self._mtime(p["transitions_topk"]),
            "basins": self._mtime(p["basins"]),
        }
        if current_art != self.state.artifacts_mtime:
            return True
        if self._mtime(p["gate_daily"]) != self.state.daily_gate_mtime:
            return True
        return False

    def _load_aligned(self, voxel_map_df: pd.DataFrame) -> pd.DataFrame:
        p = self._paths()
        gate_ts_col = _env("GATE_DAILY_TS_COL", "ts")
        gate_col = _env("GATE_DAILY_COL", "gate_on_2of3")
        gate_path = p["gate_daily"]
        if not gate_path.exists():
            raise FileNotFoundError(f"missing gate file: {gate_path}")

        if gate_path.suffix == ".parquet":
            gate = pd.read_parquet(gate_path)
        else:
            gate = pd.read_csv(gate_path)
        if gate_ts_col not in gate.columns:
            raise ValueError(f"gate ts column not found: {gate_ts_col}")
        if gate_col not in gate.columns:
            raise ValueError(f"gate column not found: {gate_col}")

        g = gate[[gate_ts_col, gate_col]].rename(columns={gate_ts_col: "ts", gate_col: "gate_on"}).copy()
        g["ts"] = _to_utc_ns(g["ts"])
        g = g.dropna(subset=["ts"]).sort_values("ts")
        g["gate_on"] = pd.to_numeric(g["gate_on"], errors="coerce").fillna(0).astype(int).clip(0, 1)

        vm = voxel_map_df[["ts", "voxel_id"]].copy()
        vm["ts"] = _to_utc_ns(vm["ts"])
        vm = vm.dropna(subset=["ts"]).sort_values("ts")

        aligned = pd.merge_asof(vm, g, on="ts", direction="backward")
        aligned["gate_on"] = aligned["gate_on"].fillna(0).astype(int).clip(0, 1)

        if _env("GATE_CONF_WRITE_ALIGNED", "0").strip().lower() in {"1", "true", "yes", "on"}:
            cache_dir = Path(_env("GATE_CONF_CACHE_DIR", "data/live/cache"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            aligned.to_parquet(cache_dir / f"aligned_gate_{self.cache_key}.parquet", index=False)
        return aligned

    def _build_sparse(self, transitions_topk: pd.DataFrame, all_vids: np.ndarray) -> Tuple[Dict[int, int], List[List[Tuple[int, float]]]]:
        id2idx = {int(v): i for i, v in enumerate(all_vids)}
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(len(all_vids))]
        for _, row in transitions_topk.iterrows():
            try:
                fid = int(row["from_voxel_id"])
                tid = int(row["to_voxel_id"])
                p = float(row["p"])
            except Exception:
                continue
            if p <= 0:
                continue
            if fid in id2idx and tid in id2idx:
                adj[id2idx[fid]].append((id2idx[tid], p))
        return id2idx, adj

    def _propagate(self, adj: List[List[Tuple[int, float]]], start_idx: int, n_steps: int, n_nodes: int) -> np.ndarray:
        v = np.zeros(n_nodes, dtype=np.float64)
        v[start_idx] = 1.0
        for _ in range(max(0, int(n_steps))):
            nxt = np.zeros(n_nodes, dtype=np.float64)
            for i in range(n_nodes):
                vi = v[i]
                if vi <= 1e-15:
                    continue
                row_sum = 0.0
                for j, p in adj[i]:
                    nxt[j] += vi * p
                    row_sum += p
                self_p = max(0.0, 1.0 - row_sum)
                nxt[i] += vi * self_p
            v = nxt
        return v

    def _build_basin_sparse(
        self,
        transitions_topk: pd.DataFrame,
        voxel_stats: pd.DataFrame,
        basins_df: pd.DataFrame,
        noise_id: int = -1,
    ) -> Tuple[np.ndarray, Dict[int, int], List[List[Tuple[int, float]]], Dict[int, int]]:
        b = basins_df[["voxel_id", "basin_id"]].copy()
        b["basin_id"] = pd.to_numeric(b["basin_id"], errors="coerce").fillna(noise_id).astype(int)
        v2b = dict(zip(b["voxel_id"].astype(int), b["basin_id"].astype(int)))

        pi_map = dict(
            zip(
                voxel_stats["voxel_id"].astype(int),
                pd.to_numeric(voxel_stats["pi"], errors="coerce").fillna(0.0).astype(float),
            )
        )
        flow: Dict[Tuple[int, int], float] = {}
        for _, row in transitions_topk.iterrows():
            try:
                fv = int(row["from_voxel_id"])
                tv = int(row["to_voxel_id"])
                p = float(row["p"])
            except Exception:
                continue
            if p <= 0:
                continue
            fb = int(v2b.get(fv, noise_id))
            tb = int(v2b.get(tv, noise_id))
            if fb == noise_id or tb == noise_id:
                continue
            flow[(fb, tb)] = flow.get((fb, tb), 0.0) + pi_map.get(fv, 0.0) * p

        basin_set = set()
        for fb, tb in flow.keys():
            basin_set.add(int(fb))
            basin_set.add(int(tb))
        basin_ids = np.array(sorted(basin_set), dtype=int)
        bid2idx = {int(bid): i for i, bid in enumerate(basin_ids)}
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(len(basin_ids))]

        row_sum: Dict[int, float] = {}
        for (fb, _), val in flow.items():
            row_sum[fb] = row_sum.get(fb, 0.0) + val
        for (fb, tb), val in flow.items():
            s = row_sum.get(fb, 0.0)
            if s > 0 and fb in bid2idx and tb in bid2idx:
                adj[bid2idx[fb]].append((bid2idx[tb], float(val / s)))
        return basin_ids, bid2idx, adj, v2b

    def _refresh(self) -> None:
        p = self._paths()
        for k in ("voxel_map", "voxel_stats", "transitions_topk", "basins", "gate_daily"):
            if not p[k].exists():
                raise FileNotFoundError(f"missing required live confidence file: {p[k]}")

        voxel_map_df = pd.read_parquet(p["voxel_map"], columns=["ts", "voxel_id"])
        voxel_map_df["ts"] = _to_utc_ns(voxel_map_df["ts"])
        voxel_map_df = voxel_map_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        voxel_stats = pd.read_parquet(p["voxel_stats"], columns=["voxel_id", "pi"])
        transitions_topk = pd.read_parquet(p["transitions_topk"], columns=["from_voxel_id", "to_voxel_id", "p"])
        basins_df = pd.read_parquet(p["basins"], columns=["voxel_id", "basin_id"])

        aligned = self._load_aligned(voxel_map_df)
        g = aligned.groupby("voxel_id", as_index=False).agg(
            n=("gate_on", "count"),
            gate_sum=("gate_on", "sum"),
        )
        g["gate_on_rate"] = g["gate_sum"] / g["n"].clip(lower=1)
        voxel_gate_rate = dict(zip(g["voxel_id"].astype(int), g["gate_on_rate"].astype(float)))
        voxel_gate_n = dict(zip(g["voxel_id"].astype(int), g["n"].astype(int)))
        voxel_gate_ci = {}
        for _, row in g.iterrows():
            lo, hi = _wilson_ci(int(row["gate_sum"]), int(row["n"]))
            voxel_gate_ci[int(row["voxel_id"])] = (float(lo), float(hi))

        all_vids = np.array(sorted(voxel_stats["voxel_id"].astype(int).unique().tolist()), dtype=int)
        id2idx, adj_voxel = self._build_sparse(transitions_topk, all_vids)
        gate_vec_voxel = np.array([voxel_gate_rate.get(int(v), 0.0) for v in all_vids], dtype=np.float64)

        basin_ids, bid2idx, adj_basin, v2b = self._build_basin_sparse(transitions_topk, voxel_stats, basins_df, noise_id=-1)

        bjoin = basins_df[["voxel_id", "basin_id"]].copy()
        bjoin["basin_id"] = pd.to_numeric(bjoin["basin_id"], errors="coerce").fillna(-1).astype(int)
        bjoin = bjoin[bjoin["basin_id"] != -1]
        stats_df = bjoin.merge(g[["voxel_id", "gate_on_rate", "n"]], on="voxel_id", how="left").merge(
            voxel_stats[["voxel_id", "pi"]],
            on="voxel_id",
            how="left",
        )
        stats_df["gate_on_rate"] = pd.to_numeric(stats_df["gate_on_rate"], errors="coerce").fillna(0.0)
        stats_df["n"] = pd.to_numeric(stats_df["n"], errors="coerce").fillna(0).astype(int)
        stats_df["pi"] = pd.to_numeric(stats_df["pi"], errors="coerce").fillna(0.0)

        basin_gate_rate: Dict[int, float] = {}
        basins_stats: Dict[int, Dict[str, float]] = {}
        for bid, chunk in stats_df.groupby("basin_id"):
            pi_sum = float(chunk["pi"].sum())
            wrate = float(np.dot(chunk["pi"].to_numpy(dtype=float), chunk["gate_on_rate"].to_numpy(dtype=float)) / max(pi_sum, 1e-12))
            basin_gate_rate[int(bid)] = wrate
            basins_stats[int(bid)] = {
                "n_voxels": float(len(chunk)),
                "mass_pi": pi_sum,
                "gate_on_rate": wrate,
            }
        gate_vec_basin = np.array([basin_gate_rate.get(int(b), 0.0) for b in basin_ids], dtype=np.float64)

        st = _EngineState(
            voxel_map_ts=voxel_map_df["ts"].to_numpy(),
            voxel_map_vid=voxel_map_df["voxel_id"].astype(int).to_numpy(),
            all_vids=all_vids,
            id2idx=id2idx,
            adj_voxel=adj_voxel,
            gate_vec_voxel=gate_vec_voxel,
            voxel_gate_rate=voxel_gate_rate,
            voxel_gate_n=voxel_gate_n,
            voxel_gate_ci=voxel_gate_ci,
            basins_map={int(k): int(v) for k, v in v2b.items()},
            basin_ids=basin_ids,
            bid2idx=bid2idx,
            adj_basin=adj_basin,
            gate_vec_basin=gate_vec_basin,
            basin_gate_rate=basin_gate_rate,
            basins_stats=basins_stats,
            artifacts_mtime={
                "voxel_map": self._mtime(p["voxel_map"]),
                "voxel_stats": self._mtime(p["voxel_stats"]),
                "transitions_topk": self._mtime(p["transitions_topk"]),
                "basins": self._mtime(p["basins"]),
            },
            daily_gate_mtime=self._mtime(p["gate_daily"]),
        )
        self.state = st
        self.last_refresh = time.time()

    def _resolve_now(self, st: _EngineState) -> Tuple[pd.Timestamp, int]:
        mode = _env("GATE_CONF_NOW_MODE", "last_ts").strip().lower()
        if mode == "server_time_asof":
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            idx = int(np.searchsorted(st.voxel_map_ts, now.to_datetime64(), side="right") - 1)
            idx = max(0, min(idx, len(st.voxel_map_vid) - 1))
            ts = pd.Timestamp(st.voxel_map_ts[idx])
            vid = int(st.voxel_map_vid[idx])
            return (ts.tz_localize("UTC") if ts.tzinfo is None else ts, vid)

        # last_ts mode with robust parquet tail reader
        p = self._paths()["voxel_map"]
        return _latest_now_from_voxel_map(p)

    def get_latest(self) -> Dict[str, Any]:
        with self.lock:
            if self._needs_refresh():
                self._refresh()
            if self.state is None:
                raise RuntimeError("gate confidence state not initialized")
            st = self.state

        now_ts, now_vid = self._resolve_now(st)
        now_bid = int(st.basins_map.get(now_vid, -1))

        horizons = _parse_horizons(_env("GATE_CONF_HORIZONS_MINUTES", "5,30,120,240"))
        gate_on_means = _env("GATE_ON_MEANS", "trend").strip().lower()
        out_h = []
        for h in horizons:
            conf_v = float("nan")
            conf_b = float("nan")

            if now_vid in st.id2idx:
                dist_v = self._propagate(st.adj_voxel, st.id2idx[now_vid], int(h), len(st.all_vids))
                conf_v = float(np.dot(dist_v, st.gate_vec_voxel))
            else:
                conf_v = float(st.voxel_gate_rate.get(now_vid, 0.0))

            if now_bid in st.bid2idx and len(st.basin_ids) > 0:
                dist_b = self._propagate(st.adj_basin, st.bid2idx[now_bid], int(h), len(st.basin_ids))
                conf_b = float(np.dot(dist_b, st.gate_vec_basin))

            if gate_on_means == "countertrend":
                p_counter_v = conf_v
                p_trend_v = 1.0 - conf_v
                p_counter_b = conf_b if np.isfinite(conf_b) else float("nan")
                p_trend_b = 1.0 - conf_b if np.isfinite(conf_b) else float("nan")
            else:
                p_trend_v = conf_v
                p_counter_v = 1.0 - conf_v
                p_trend_b = conf_b if np.isfinite(conf_b) else float("nan")
                p_counter_b = 1.0 - conf_b if np.isfinite(conf_b) else float("nan")

            out_h.append(
                {
                    "minutes": int(h),
                    "p_gate_on_voxel": round(conf_v, 6),
                    "p_gate_on_basin": round(conf_b, 6) if np.isfinite(conf_b) else None,
                    "p_trend_voxel": round(float(p_trend_v), 6),
                    "p_countertrend_voxel": round(float(p_counter_v), 6),
                    "p_trend_basin": round(float(p_trend_b), 6) if np.isfinite(p_trend_b) else None,
                    "p_countertrend_basin": round(float(p_counter_b), 6) if np.isfinite(p_counter_b) else None,
                }
            )

        selected_min = int(_env("GATE_CONF_SELECTED_MINUTES", "30"))
        selected = next((x for x in out_h if int(x["minutes"]) == selected_min), out_h[0] if out_h else None)

        n_now = int(st.voxel_gate_n.get(now_vid, 0))
        ci_lo, ci_hi = st.voxel_gate_ci.get(now_vid, (0.0, 1.0))

        return {
            "now_ts": pd.Timestamp(now_ts).isoformat(),
            "now_voxel_id": int(now_vid),
            "now_basin_id": int(now_bid),
            "gate_on_means": gate_on_means if gate_on_means in {"trend", "countertrend"} else "trend",
            "horizons": out_h,
            "selected_minutes": selected_min,
            "selected_p_trend": None if selected is None else selected["p_trend_voxel"],
            "selected_p_countertrend": None if selected is None else selected["p_countertrend_voxel"],
            "now_gate_on_rate_voxel": round(float(st.voxel_gate_rate.get(now_vid, 0.0)), 6),
            "now_gate_on_rate_basin": round(float(st.basin_gate_rate.get(now_bid, float("nan"))), 6)
            if now_bid in st.basin_gate_rate
            else None,
            "now_voxel_n": n_now,
            "now_voxel_ci_lo": round(float(ci_lo), 6),
            "now_voxel_ci_hi": round(float(ci_hi), 6),
        }


_ENGINE = GateConfidenceLiveEngine()


def get_live_gate_confidence() -> Dict[str, Any]:
    return _ENGINE.get_latest()
