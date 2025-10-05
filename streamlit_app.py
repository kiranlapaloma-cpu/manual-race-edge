# ======================= Batch 1 — Core & UI Setup =======================
# Unified 100m + 200m app (Grid-aware core, header normalization, finish derivation, stub planner)

import io, math, re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Race Edge — Unified (100m + 200m)", layout="wide")

# -------------------- Small helpers --------------------
def as_num(x): 
    return pd.to_numeric(x, errors="coerce")

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower().strip())

@st.cache_data(show_spinner=False)
def _hash_df_for_cache(df: pd.DataFrame) -> str:
    # lightweight content hash for caching downstream steps
    return str(pd.util.hash_pandas_object(df.head(100), index=True).sum())

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### Upload")
    up = st.file_uploader(
        "CSV/XLSX with splits (either 100m or 200m grid). Include Finish split or a race total.",
        type=["csv", "xlsx", "xls"]
    )
    race_distance_input = st.number_input(
        "Race Distance (m)", min_value=800, max_value=4200, step=50, value=1600,
        help="Any distance is fine (e.g., 1160, 1250, 1450, 1750, 1900). Stub will be handled."
    )
    FORCE_GRID = st.selectbox("Grid (auto-detect by default)", ["Auto", "100m", "200m"], index=0)
    SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
    DEBUG = st.toggle("Debug info", value=False)

if not up:
    st.stop()

# -------------------- Load file --------------------
try:
    raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    st.success("File loaded.")
except Exception as e:
    st.error("Failed to read file.")
    st.exception(e)
    st.stop()

# -------------------- Time parser --------------------
def parse_time_to_seconds(v):
    """Accepts 12.34 / 12,34 / m:ss(.xx) / h:mm:ss(.xx) / strings; returns seconds float or NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except Exception:
            return np.nan
        if len(parts) == 3:
            h, m, sec = parts
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = parts
            return m * 60 + sec
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

# -------------------- Header normalization --------------------
FINISH_SPLIT_ALIASES = {
    "finishsplit","finish100","finish200","last100","last200","final100","final200",
    "home100","home200","tofinish","final","fin200","fin100"
}
FINISH_TOTAL_ALIASES = {
    "finishtime","racetime","totaltime","overalltime","finaltime","time","total"
}
POS_ALIASES = ["finishpos","finishposition","position","pos","finpos"]
HORSE_ALIASES = ["horse","runner","name","horse_name","runner_name"]

@st.cache_data(show_spinner=False)
def normalize_headers(df: pd.DataFrame, D_m: int) -> tuple[pd.DataFrame, list[str]]:
    rename = {}
    notes = []
    for col in list(df.columns):
        raw = str(col).strip()
        cx = _canon(raw)

        if cx in POS_ALIASES or re.fullmatch(r'(?i)finish\s*pos(ition)?|pos(ition)?|fin_pos|finishpos', raw):
            rename[col] = "Finish_Pos"; continue

        if cx in HORSE_ALIASES:
            rename[col] = "Horse"; continue

        # Finish split vs total
        if (cx in FINISH_SPLIT_ALIASES) or re.fullmatch(r'(?i)(finish\s*split|finish\s*100|finish\s*200|last\s*100|last\s*200|final\s*100|final\s*200|home\s*100|home\s*200)', raw):
            rename[col] = "Finish_Time"; continue
        if (cx in FINISH_TOTAL_ALIASES) or re.fullmatch(r'(?i)(finish\s*time|race\s*time|total\s*time|overall\s*time|final\s*time|time)', raw):
            rename[col] = "RaceTotal_raw"; continue

        # Generic "{metres}[m] ... (time|split)?" → "{metres}_Time"
        m = re.match(r'(?i).*?(\d{2,4})\s*m?\s*[_\s-]?\s*(time|split)?$', raw)
        if m:
            metres = int(m.group(1))
            if 1 <= metres <= int(D_m):
                rename[col] = f"{metres}_Time"; continue

        # Camel cases like "1600mTime"
        m2 = re.match(r'(?i)^(\d{2,4})m(Time|Split)$', raw)
        if m2:
            metres = int(m2.group(1))
            if 1 <= metres <= int(D_m):
                rename[col] = f"{metres}_Time"; continue

    out = df.rename(columns=rename)
    # Drop any duplicate columns keeping the first occurrence
    out = out.loc[:, ~out.columns.duplicated()]

    # Coerce Finish_Pos to numeric (digits only)
    if "Finish_Pos" in out.columns:
        out["Finish_Pos"] = out["Finish_Pos"].astype(str).str.extract(r'(\d+)')[0].astype(float)

    # Note renames
    for k, v in rename.items():
        if k != v:
            notes.append(f"`{k}` → `{v}`")

    return out, notes

work, alias_notes = normalize_headers(raw.copy(), int(race_distance_input))

# -------------------- Finish vs Total derivation (grid-agnostic) --------------------
def coerce_times_and_derive_finish(df: pd.DataFrame, D_m: int) -> pd.DataFrame:
    g = df.copy()

    # Coerce *_Time + RaceTotal_raw
    for c in list(g.columns):
        if c.endswith("_Time") or c == "RaceTotal_raw":
            g[c] = g[c].apply(parse_time_to_seconds)

    # If Finish_Time looks like a total (>40s), reclassify as RaceTotal_s
    if "Finish_Time" in g.columns:
        mask_totalish = g["Finish_Time"] > 40.0
        if mask_totalish.any():
            g.loc[mask_totalish, "RaceTotal_s"] = g.loc[mask_totalish, "Finish_Time"]
            g.loc[mask_totalish, "Finish_Time"] = np.nan

    # Convert RaceTotal_raw → RaceTotal_s
    if "RaceTotal_raw" in g.columns:
        g["RaceTotal_s"] = g["RaceTotal_raw"].apply(parse_time_to_seconds)
        g.drop(columns=["RaceTotal_raw"], inplace=True, errors="ignore")

    # If Finish_Time missing, derive from total − sum(other splits)
    need_finish = ("Finish_Time" not in g.columns) or g["Finish_Time"].isna().all()
    if need_finish and "RaceTotal_s" in g.columns:
        # We don't know grid yet; sum all *_Time except Finish
        split_cols = [c for c in g.columns if c.endswith("_Time") and c != "Finish_Time"]
        if split_cols:
            sums = g[split_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            g["Finish_Time"] = g["RaceTotal_s"] - sums
            # keep plausible range for the last split (we'll validate again after grid detect)
            g.loc[(g["Finish_Time"] < 5.0) | (g["Finish_Time"] > 35.0), "Finish_Time"] = np.nan

    return g

work = coerce_times_and_derive_finish(work, int(race_distance_input))

# -------------------- Grid detection (100m vs 200m) --------------------
def collect_markers(df: pd.DataFrame) -> list[int]:
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                marks.append(int(str(c).split("_")[0]))
            except Exception:
                pass
    return sorted(set(marks), reverse=True)

def detect_grid(markers: list[int]) -> str:
    if len(markers) < 3:
        return "unknown"
    diffs = []
    for i in range(len(markers) - 1):
        d = markers[i] - markers[i + 1]
        if 50 <= d <= 400:
            diffs.append(d)
    if not diffs:
        return "unknown"
    # count closeness to 100 or 200
    cnt100 = sum(1 for d in diffs if abs(d - 100) <= 20)
    cnt200 = sum(1 for d in diffs if abs(d - 200) <= 40)
    if cnt100 >= cnt200 and cnt100 >= max(1, 0.7 * len(diffs)):
        return "100m"
    if cnt200 > cnt100 and cnt200 >= max(1, 0.7 * len(diffs)):
        return "200m"
    # fallback: prefer finer grid
    return "100m" if cnt100 >= cnt200 else "200m"

markers = collect_markers(work)
auto_grid = detect_grid(markers)
grid = {"Auto": auto_grid, "100m": "100m", "200m": "200m"}[FORCE_GRID]

# -------------------- Stub-aware segment planner --------------------
def build_segment_plan_200(D: int) -> list[tuple[int, float]]:
    """Return ordered segments: list of (end_marker, length_m), excluding Finish (200→0)."""
    rem = D % 200
    segs = []
    if rem > 0:
        first_end = D - rem
        if first_end >= 200:
            segs.append((first_end, float(rem)))  # D → (D-rem) stub
    else:
        first_end = D - 200
        if first_end >= 200:
            segs.append((first_end, 200.0))
    m = first_end - 200
    while m >= 200:
        segs.append((m, 200.0))
        m -= 200
    return segs  # e.g., [(1000,160.0),(800,200.0),...,(200,200.0)]

def build_segment_plan_100(D: int) -> list[tuple[int, float]]:
    """Return ordered segments: list of (end_marker, length_m), excluding Finish (100→0)."""
    segs = []
    # 100m grid has no leading stub by definition; just walk by 100s
    for m in range(D - 100, 99, -100):
        segs.append((m, 100.0))
    return segs  # e.g., [(1500,100),(1400,100),...,(100,100)]

D = int(race_distance_input)
if grid == "200m":
    seg_plan = build_segment_plan_200(D)
    finish_len_expected = 200.0
elif grid == "100m":
    seg_plan = build_segment_plan_100(D)
    finish_len_expected = 100.0
else:
    # unknown → try best-effort using markers: if highest gap ~100 → 100m else 200m
    tentative = "100m" if auto_grid == "100m" else "200m"
    if tentative == "100m":
        seg_plan = build_segment_plan_100(D); finish_len_expected = 100.0
    else:
        seg_plan = build_segment_plan_200(D); finish_len_expected = 200.0
    grid = tentative

# -------------------- Finish sanity recheck (now that grid known) --------------------
if "Finish_Time" in work.columns:
    if finish_len_expected == 100.0:
        # typical ~11–13s; keep a broad window but exclude absurd values
        work.loc[(work["Finish_Time"] < 6.0) | (work["Finish_Time"] > 25.0), "Finish_Time"] = np.nan
    else:
        # typical ~22–25s; keep a broad window
        work.loc[(work["Finish_Time"] < 9.0) | (work["Finish_Time"] > 35.0), "Finish_Time"] = np.nan

# -------------------- Display: raw & detection summary --------------------
st.markdown("### Raw Table (normalized)")
st.dataframe(work.head(12), use_container_width=True)

det_msg = f"**Detected grid:** `{grid}`  •  **Markers found:** {len(markers)}  •  **Finish len (expected):** {int(finish_len_expected)} m"
if alias_notes:
    det_msg += "  •  Aliases: " + "; ".join(alias_notes)
st.info(det_msg)

if DEBUG:
    st.write("Seg plan (end_marker, length_m):", seg_plan)

# Save to session for downstream batches
st.session_state["__raw_df_hash__"] = _hash_df_for_cache(work)
st.session_state["work_df"] = work
st.session_state["race_distance_m"] = D
st.session_state["grid"] = grid
st.session_state["seg_plan"] = seg_plan
st.session_state["finish_len_expected"] = finish_len_expected
st.session_state["SHOW_WARNINGS"] = SHOW_WARNINGS
st.session_state["DEBUG"] = DEBUG

# ======================= Batch 2 — Metric Computation Engine =======================
# Uses Batch 1 session state: work_df, race_distance_m, grid, seg_plan, finish_len_expected

import math
import numpy as np
import pandas as pd
import streamlit as st

# ---- Guard: ensure Batch 1 ran
for k in ["work_df", "race_distance_m", "grid", "seg_plan", "finish_len_expected", "__raw_df_hash__"]:
    if k not in st.session_state:
        st.error("Batch 1 must run before Batch 2. Please paste/run Batch 1 first.")
        st.stop()

work = st.session_state["work_df"].copy()
D = int(st.session_state["race_distance_m"])
grid = st.session_state["grid"]
seg_plan = list(st.session_state["seg_plan"])            # [(end_marker, length_m), ...]
finish_len_expected = float(st.session_state["finish_len_expected"])
SHOW_WARNINGS = st.session_state.get("SHOW_WARNINGS", True)
DEBUG = st.session_state.get("DEBUG", False)
RAW_HASH = st.session_state["__raw_df_hash__"]

# -------------------- small helpers (Batch-2 local) --------------------
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def mad_std(x):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def winsorize(s, p_lo=0.10, p_hi=0.90):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(p_lo); hi = s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)

def _lerp(a, b, t): 
    return a + (b - a) * float(t)

def _interpolate_weights(dm, a_dm, a_w, b_dm, b_w):
    span = float(b_dm - a_dm)
    t = 0.0 if span <= 0 else (float(dm) - a_dm) / span
    return {
        "F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
        "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
        "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
        "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t),
    }

# -------------------- distance + context PI weights (unchanged maths) --------------------
def pi_weights_distance_and_context(distance_m: float,
                                    acc_median: float | None,
                                    grd_median: float | None) -> dict:
    dm = float(distance_m or 1200)

    if dm <= 1000:
        base = {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17}
    elif dm < 1100:
        base = _interpolate_weights(
            dm,
            1000, {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17},
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20}
        )
    elif dm < 1200:
        base = _interpolate_weights(
            dm,
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20},
            1200, {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
        )
    elif dm == 1200:
        base = {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
    else:
        shift_units = max(0.0, (dm - 1200.0) / 100.0) * 0.01
        grind = min(0.25 + shift_units, 0.40)
        F200, ACC = 0.08, 0.30
        ts = max(0.0, 1.0 - F200 - ACC - grind)
        base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":grind}

    acc_med = float(acc_median) if acc_median is not None and math.isfinite(acc_median) else None
    grd_med = float(grd_median) if grd_median is not None and math.isfinite(grd_median) else None
    if acc_med is not None and grd_med is not None:
        bias = acc_med - grd_med
        scale = math.tanh(abs(bias) / 6.0)
        max_shift = 0.02 * scale

        F200 = base["F200_idx"]; ts = base["tsSPI"]; ACC = base["Accel"]; GR = base["Grind"]
        if bias > 0:
            delta = min(max_shift, ACC - 0.26)
            ACC -= delta; GR += delta
        elif bias < 0:
            delta = min(max_shift, GR - 0.18)
            GR  -= delta; ACC += delta
        GR = min(GR, 0.40)
        ts = max(0.0, 1.0 - F200 - ACC - GR)
        base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":GR}

    s = sum(base.values())
    if abs(s - 1.0) > 1e-6:
        base = {k: v / s for k, v in base.items()}
    return base

# -------------------- Stage builders (grid-aware) --------------------
def stage_columns_and_lengths(grid: str, D: int):
    cols = {}
    lens = {}

    if grid == "100m":
        # F200: (D-100)+(D-200) if present
        f200_cols = [c for c in [f"{D-100}_Time", f"{D-200}_Time"] if c in work.columns]
        f200_lens = [100.0 for _ in f200_cols]

        # tsSPI: (D-300) down to 600 step 100 (clip to existing)
        tssp_cols = [f"{m}_Time" for m in range(D-300, 599, -100) if m >= 600 and f"{m}_Time" in work.columns]
        tssp_lens = [100.0 for _ in tssp_cols]

        # Accel: 500 + 400 + 300 + 200 (existing only)
        accel_cols = [c for c in [f"{m}_Time" for m in [500, 400, 300, 200]] if c in work.columns]
        accel_lens = [100.0 for _ in accel_cols]

        # Grind: 100 + Finish
        grind_cols = [c for c in ["100_Time", "Finish_Time"] if c in work.columns]
        # use finish_len_expected from Batch-1 (should be 100 here but future-proof)
        grind_lens = []
        for c in grind_cols:
            grind_lens.append(100.0 if c == "100_Time" else float(finish_len_expected))

    else:  # 200m grid
        # F200: the FIRST split (stub length if any)
        if len(seg_plan) > 0:
            first_end, first_len = seg_plan[0]
            f200_cols = [f"{first_end}_Time"] if f"{first_end}_Time" in work.columns else []
            f200_lens = [float(first_len) for _ in f200_cols]
        else:
            f200_cols, f200_lens = [], []

        # tsSPI: (D-400) down to 600 step 200
        tssp_cols = [f"{m}_Time" for m in range(D-400, 599, -200) if m >= 600 and f"{m}_Time" in work.columns]
        tssp_lens = [200.0 for _ in tssp_cols]

        # Accel: 400 + 200
        accel_cols = [c for c in ["400_Time", "200_Time"] if c in work.columns]
        accel_lens = [200.0 for _ in accel_cols]

        # Grind: Finish only (200 m)
        grind_cols = [c for c in ["Finish_Time"] if c in work.columns]
        grind_lens = [float(finish_len_expected) for _ in grind_cols]

    cols["F200"], lens["F200"] = f200_cols, f200_lens
    cols["tsSPI"], lens["tsSPI"] = tssp_cols, tssp_lens
    cols["Accel"], lens["Accel"] = accel_cols, accel_lens
    cols["Grind"], lens["Grind"] = grind_cols, grind_lens
    return cols, lens

def stage_speed_row(row, cols, lens):
    """Distance-correct speed: sum(lengths of available pieces) / sum(times)."""
    if not cols:
        return np.nan
    times = []
    dsum = 0.0
    for c, L in zip(cols, lens):
        t = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(t) and float(t) > 0:
            times.append(float(t))
            dsum += float(L)
    if dsum <= 0 or len(times) == 0:
        return np.nan
    return dsum / sum(times)

# -------------------- RaceTime_s computation (grid-aware) --------------------
def compute_race_time(df: pd.DataFrame, grid: str, D: int) -> pd.Series:
    if "RaceTotal_s" in df.columns and df["RaceTotal_s"].notna().any():
        return pd.to_numeric(df["RaceTotal_s"], errors="coerce")

    if grid == "200m":
        split_cols = [f"{end}_Time" for (end, _) in seg_plan if f"{end}_Time" in df.columns]
    else:
        split_cols = [f"{m}_Time" for m in range(D-100, 99, -100) if f"{m}_Time" in df.columns]

    cols = split_cols + (["Finish_Time"] if "Finish_Time" in df.columns else [])
    if not cols:
        return pd.Series(np.nan, index=df.index)

    tmp = df[cols].apply(pd.to_numeric, errors="coerce")
    tmp = tmp.where(tmp > 0)  # treat <=0 as missing
    return tmp.sum(axis=1)

# -------------------- Speed→Index stabilizers --------------------
def speed_to_index(spd_series: pd.Series) -> pd.Series:
    spd_series = pd.to_numeric(spd_series, errors="coerce")
    med = spd_series.median(skipna=True)
    idx_raw = 100.0 * (spd_series / med)

    # small-field stabilizers (same as your apps)
    x = idx_raw.dropna().values
    N_eff = len(x)
    if N_eff == 0:
        center = 100.0; neff = 0
    else:
        med_race = float(np.median(x))
        alpha = N_eff / (N_eff + 6.0)
        center = alpha * med_race + (1 - alpha) * 100.0
        neff = N_eff

    idx = 100.0 * (spd_series / (center / 100.0 * med))

    def dispersion_equalizer(delta_series, N_eff, N_ref=10, beta=0.20, cap=1.20):
        gamma = 1.0 + beta * max(0, N_ref - N_eff) / N_ref
        return delta_series * min(gamma, cap)

    idx = 100.0 + dispersion_equalizer(idx - 100.0, neff)

    def variance_floor(idx_series, floor=1.5, cap=1.25):
        deltas = idx_series - 100.0
        sigma = mad_std(deltas)
        if not np.isfinite(sigma) or sigma <= 0:
            return idx_series
        if sigma < floor:
            factor = min(cap, floor / sigma)
            return 100.0 + deltas * factor
        return idx_series

    idx = variance_floor(idx)
    return idx

# -------------------- Cached metric builder --------------------
@st.cache_data(show_spinner=True)
def build_metrics_cached(_raw_hash: str, df: pd.DataFrame, D: int, grid: str, seg_plan: list, finish_len_expected: float):
    w = df.copy()

    # numeric finish pos if present
    if "Finish_Pos" in w.columns:
        w["Finish_Pos"] = as_num(w["Finish_Pos"])

    # stage columns + lengths
    cols, lens = stage_columns_and_lengths(grid, D)

    # compute stage speeds
    w["_F200_spd"] = w.apply(lambda r: stage_speed_row(r, cols["F200"],  lens["F200"]),  axis=1)
    w["_MID_spd"]  = w.apply(lambda r: stage_speed_row(r, cols["tsSPI"], lens["tsSPI"]), axis=1)
    w["_ACC_spd"]  = w.apply(lambda r: stage_speed_row(r, cols["Accel"], lens["Accel"]), axis=1)
    w["_GR_spd"]   = w.apply(lambda r: stage_speed_row(r, cols["Grind"], lens["Grind"]), axis=1)

    # speeds → indices
    w["F200_idx"] = speed_to_index(w["_F200_spd"])
    w["tsSPI"]    = speed_to_index(w["_MID_spd"])
    w["Accel"]    = speed_to_index(w["_ACC_spd"])
    w["Grind"]    = speed_to_index(w["_GR_spd"])

    # race time (seconds)
    w["RaceTime_s"] = compute_race_time(w, grid, D)

    # ----- PI v3.1 -----
    acc_med = w["Accel"].median(skipna=True)
    grd_med = w["Grind"].median(skipna=True)
    PI_W = pi_weights_distance_and_context(float(D), acc_med, grd_med)

    def pi_pts_row(row):
        parts, weights = [], []
        for k, wgt in PI_W.items():
            v = pd.to_numeric(row.get(k), errors="coerce")
            if pd.notna(v):
                parts.append(wgt * (float(v) - 100.0))
                weights.append(wgt)
        return np.nan if not weights else sum(parts) / sum(weights)

    w["PI_pts"] = w.apply(pi_pts_row, axis=1)

    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered = pts - med
    sigma = mad_std(centered)
    if not np.isfinite(sigma) or sigma < 0.75:
        sigma = 0.75
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)

    # ----- GCI (0–10) -----
    acc_med_g = w["Accel"].median(skipna=True)
    grd_med_g = w["Grind"].median(skipna=True)
    Wg = pi_weights_distance_and_context(float(D), acc_med_g, grd_med_g)

    wT   = 0.25
    wPACE= Wg["Accel"] + Wg["Grind"]
    wSS  = Wg["tsSPI"]
    wEFF = max(0.0, 1.0 - (wT + wPACE + wSS))

    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try:
            winner_time = float(w["RaceTime_s"].min())
        except Exception:
            winner_time = None

    def map_pct(x, lo=98.0, hi=104.0):
        x = pd.to_numeric(x, errors="coerce")
        if pd.isna(x): return 0.0
        return clamp((float(x) - lo) / (hi - lo), 0.0, 1.0)

    gci_vals = []
    for _, r in w.iterrows():
        T = 0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d = float(r["RaceTime_s"]) - winner_time
            if d <= 0.30:   T = 1.0
            elif d <= 0.60: T = 0.7
            elif d <= 1.00: T = 0.4
            else:           T = 0.2

        LQ = 0.6 * map_pct(r.get("Accel")) + 0.4 * map_pct(r.get("Grind"))
        SS = map_pct(r.get("tsSPI"))

        acc, grd = r.get("Accel"), r.get("Grind")
        if pd.isna(acc) or pd.isna(grd):
            EFF = 0.0
        else:
            dev = (abs(float(acc) - 100.0) + abs(float(grd) - 100.0)) / 2.0
            EFF = clamp(1.0 - dev / 8.0, 0.0, 1.0)

        score01 = (wT * T) + (wPACE * LQ) + (wSS * SS) + (wEFF * EFF)
        gci_vals.append(round(10.0 * score01, 3))
    w["GCI"] = gci_vals

    # ----- tidy rounding -----
    for c in ["F200_idx","tsSPI","Accel","Grind","PI","GCI","RaceTime_s"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce").round(3)

    return w

metrics = build_metrics_cached(RAW_HASH, work, D, grid, seg_plan, finish_len_expected)

# ======================= Hidden Horses v2 — DROP-IN =======================
import numpy as np
import pandas as pd
import streamlit as st

# ---- Inputs expected in session_state ----
metrics = st.session_state["metrics"].copy()
work_df = st.session_state["work_df"].copy()
grid    = st.session_state["grid"]              # "100m" | "200m"
D       = int(st.session_state["race_distance_m"])

# ---- small helpers ----
def mad_std(x):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x); mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def winsorize(s, p_lo=0.10, p_hi=0.90):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(p_lo), s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)

# ---- 1) SOS (strength-of-sectionals) ----
hh = metrics.copy()
if {"tsSPI","Accel","Grind"}.issubset(hh.columns) and len(hh) > 0:
    ts_w = winsorize(hh["tsSPI"]); ac_w = winsorize(hh["Accel"]); gr_w = winsorize(hh["Grind"])
    def rz(s: pd.Series) -> pd.Series:
        mu = np.nanmedian(s); sd = mad_std(s)
        if not np.isfinite(sd) or sd == 0: return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd
    z_ts = rz(ts_w).clip(-2.5, 3.5)
    z_ac = rz(ac_w).clip(-2.5, 3.5)
    z_gr = rz(gr_w).clip(-2.5, 3.5)
    hh["SOS_raw"] = 0.45*z_ts + 0.35*z_ac + 0.20*z_gr
    q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom = (q95 - q5) if (pd.notna(q95) and pd.notna(q5) and (q95 > q5)) else 1.0
    hh["SOS"] = (2.0*(hh["SOS_raw"] - q5)/denom).clip(0.0, 2.0)
else:
    hh["SOS"] = 0.0

# ---- 2) ASI² (anti-shape index squared-lite) ----
acc_med = pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
grd_med = pd.to_numeric(hh.get("Grind"), errors="coerce").median(skipna=True)
bias = (acc_med - 100.0) - (grd_med - 100.0)           # >0 means race rewarded Accel more than Grind
B = min(1.0, abs(bias)/4.0)                             # bias strength 0..1
S = pd.to_numeric(hh.get("Accel"), errors="coerce") - pd.to_numeric(hh.get("Grind"), errors="coerce")
hh["ASI2"] = (B * (np.where(bias >= 0, -S, S)).clip(min=0) / 5.0)
hh["ASI2"] = pd.to_numeric(hh["ASI2"], errors="coerce").fillna(0.0)

# ---- 3) TFS (late variability vs mid) — grid-aware, using work_df raw splits ----
# We'll build a map from Horse -> that row's raw late split times
work_by_horse = work_df.set_index("Horse", drop=False) if "Horse" in work_df.columns else None

def _late_cols_for_grid(grid: str):
    if grid == "100m":
        # three pre-finish 100m blocks
        return ["300_Time","200_Time","100_Time"], 100.0
    # 200m grid
    return ["600_Time","400_Time","200_Time"], 200.0

late_cols, unit_len = _late_cols_for_grid(grid)

def tfs_for_horse(row_metrics):
    # Try to pull raw times from work_df by Horse name (best) else from metrics row if present
    spds = []
    def from_series(sr, col):
        t = pd.to_numeric(sr.get(col), errors="coerce")
        return (unit_len / t) if (pd.notna(t) and t > 0) else np.nan

    src = None
    if work_by_horse is not None:
        hname = str(row_metrics.get("Horse",""))
        if hname in work_by_horse.index:
            src = work_by_horse.loc[hname]
    if src is None:
        src = row_metrics

    for c in late_cols:
        spds.append(from_series(src, c))
    spds = [s for s in spds if pd.notna(s)]
    if len(spds) < 2:
        return np.nan

    sigma = float(np.std(spds, ddof=0))
    mid = float(pd.to_numeric(row_metrics.get("_MID_spd"), errors="coerce"))
    if not np.isfinite(mid) or mid <= 0:
        return np.nan
    return 100.0 * (sigma / mid)

hh["TFS"] = metrics.apply(tfs_for_horse, axis=1)

# Distance-aware TFS gate (keeps only meaningful variability)
D_rounded = int(np.ceil(float(D) / 200.0) * 200)
gate = 4.0 if D_rounded <= 1200 else (3.5 if D_rounded < 1800 else 3.0)
def tfs_plus(x):
    if pd.isna(x) or x < gate: return 0.0
    return min(0.6, (x - gate) / 3.0)
hh["TFS_plus"] = hh["TFS"].apply(tfs_plus)

# ---- 4) UEI (upside eligibility if shape flips) ----
def uei_row(r):
    ts = pd.to_numeric(r.get("tsSPI"), errors="coerce")
    ac = pd.to_numeric(r.get("Accel"), errors="coerce")
    gr = pd.to_numeric(r.get("Grind"), errors="coerce")
    if pd.isna(ts) or pd.isna(ac) or pd.isna(gr): return 0.0
    val = 0.0
    if ts >= 102 and ac <= 98 and gr <= 98:
        gap = min((ts - 102) / 3.0, 1.0); val = max(val, 0.3 + 0.3*gap)
    if ts >= 102 and gr >= 102 and ac <= 100:
        gap = min(((ts - 102) + (gr - 102)) / 6.0, 1.0); val = max(val, 0.3 + 0.3*gap)
    return round(val, 3)
hh["UEI"] = hh.apply(uei_row, axis=1)

# ---- 5) HiddenScore (0..3) + Tier + Note ----
hidden = (
    0.55 * pd.to_numeric(hh["SOS"], errors="coerce").fillna(0.0) +
    0.30 * pd.to_numeric(hh["ASI2"], errors="coerce").fillna(0.0) +
    0.10 * pd.to_numeric(hh["TFS_plus"], errors="coerce").fillna(0.0) +
    0.05 * pd.to_numeric(hh["UEI"], errors="coerce").fillna(0.0)
)
if int(hh.shape[0]) <= 6:
    hidden = hidden * 0.90

h_med = float(np.nanmedian(hidden))
h_mad = float(np.nanmedian(np.abs(hidden - h_med)))
h_sigma = max(1e-6, 1.4826*h_mad)
hh["HiddenScore"] = (1.2 + (hidden - h_med) / (2.5*h_sigma)).clip(0.0, 3.0)

def hh_tier(s):
    if pd.isna(s): return ""
    if s >= 1.8: return "🔥 Top Hidden"
    if s >= 1.2: return "🟡 Notable Hidden"
    return ""
hh["Tier"] = hh["HiddenScore"].apply(hh_tier)

def hh_note(r):
    bits=[]
    if r.get("Tier","")!="":
        if pd.to_numeric(r.get("SOS"), errors="coerce") >= 1.2: bits.append("sectionals superior")
        asi2 = pd.to_numeric(r.get("ASI2"), errors="coerce")
        if asi2 >= 0.8:   bits.append("ran against strong bias")
        elif asi2 >= 0.4: bits.append("ran against bias")
        if pd.to_numeric(r.get("TFS_plus"), errors="coerce") > 0: bits.append("trip friction late")
        if pd.to_numeric(r.get("UEI"), errors="coerce") >= 0.5: bits.append("latent potential if shape flips")
    return ("; ".join(bits).capitalize() + ".") if bits else ""
hh["Note"] = hh.apply(hh_note, axis=1)

# ---- Show & persist ----
cols_hh = ["Horse","Finish_Pos","PI","GCI","tsSPI","Accel","Grind","SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
for c in cols_hh:
    if c not in hh.columns: hh[c] = np.nan

hh_view = hh.sort_values(["Tier","HiddenScore","PI"], ascending=[True, False, False])[cols_hh]
st.markdown("## Hidden Horses v2")
st.dataframe(hh_view, use_container_width=True)

# make available to other batches
st.session_state["hh"] = hh


# -------------------- Persist Batch-2 outputs for later batches --------------------
st.session_state["metrics"] = metrics
st.session_state["hh"] = hh

# -------------------- Minimal on-screen check (small, no plots yet) --------------------
st.markdown("### ✅ Metrics ready")
preview_cols = ["Horse","Finish_Pos","RaceTime_s","F200_idx","tsSPI","Accel","Grind","PI","GCI"]
for c in preview_cols:
    if c not in metrics.columns:
        metrics[c] = np.nan
st.dataframe(metrics[preview_cols].sort_values(["PI","Finish_Pos"], ascending=[False, True]).head(12),
             use_container_width=True)

st.caption("Batch 2 computed stage indices → PI v3.1 → GCI and Hidden Horses v2 (grid-aware). Ready for visualizations & DB in the next batches.")

# ======================= Batch 3 — Visualization Suite =======================
# Requires Batch 1 + 2 to have run. Consumes session_state objects and plots three visuals.

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ---- Guards
need_keys = ["metrics", "work_df", "race_distance_m", "grid", "seg_plan"]
for k in need_keys:
    if k not in st.session_state:
        st.error(f"Batch 2 outputs missing: `{k}`. Please run Batches 1 & 2.")
        st.stop()

metrics = st.session_state["metrics"].copy()
work    = st.session_state["work_df"].copy()
D       = int(st.session_state["race_distance_m"])
grid    = st.session_state["grid"]
seg_plan= list(st.session_state["seg_plan"])   # [(end_marker, length_m), ...]
DEBUG   = st.session_state.get("DEBUG", False)

# ---------------- Helpers (local to Batch 3) ----------------
def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color',
            ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out=[]; i=0
    while len(out)<n: out.append(base[i%len(base)]); i+=1
    return out

def _repel_labels_builtin(ax, x, y, labels, *,
                          init_shift=0.18, k_repel=0.012, max_iter=250):
    trans=ax.transData; renderer=ax.figure.canvas.get_renderer()
    xy=np.column_stack([x,y]).astype(float); offs=np.zeros_like(xy)
    for i,(xi,yi) in enumerate(xy):
        offs[i]=[init_shift if xi>=0 else -init_shift,
                 init_shift if yi>=0 else -init_shift]
    texts,lines=[],[]
    for (xi,yi),(dx,dy),lab in zip(xy,offs,labels):
        t=ax.text(xi+dx, yi+dy, lab, fontsize=8.6, va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))
        texts.append(t)
        ln=Line2D([xi,xi+dx],[yi,yi+dy], lw=0.75, color="black", alpha=0.9)
        ax.add_line(ln); lines.append(ln)
    inv=ax.transData.inverted()
    for _ in range(max_iter):
        moved=False
        bbs=[t.get_window_extent(renderer=renderer).expanded(1.02,1.15) for t in texts]
        for i in range(len(texts)):
            for j in range(i+1,len(texts)):
                if not bbs[i].overlaps(bbs[j]): continue
                ci=((bbs[i].x0+bbs[i].x1)/2,(bbs[i].y0+bbs[i].y1)/2)
                cj=((bbs[j].x0+bbs[j].x1)/2,(bbs[j].y0+bbs[j].y1)/2)
                vx,vy=ci[0]-cj[0],ci[1]-cj[1]
                if vx==0 and vy==0: vx=1.0
                n=(vx**2+vy**2)**0.5; dx,dy=(vx/n)*k_repel*72,(vy/n)*k_repel*72
                for t,s in ((texts[i],+1),(texts[j],-1)):
                    tx,ty=t.get_position()
                    px=trans.transform((tx,ty))+s*np.array([dx,dy])
                    t.set_position(inv.transform(px)); moved=True
        if not moved: break
    for t,ln,(xi,yi) in zip(texts,lines,xy):
        tx,ty=t.get_position(); ln.set_data([xi,tx],[yi,ty])

def label_points_neatly(ax, x, y, names):
    try:
        from adjustText import adjust_text
        texts=[ax.text(xi,yi,nm,fontsize=8.6,
                       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))
               for xi,yi,nm in zip(x,y,names)]
        adjust_text(texts, x=x, y=y, ax=ax,
                    only_move={'points':'y','text':'xy'},
                    force_points=0.6, force_text=0.7,
                    expand_text=(1.05,1.15), expand_points=(1.05,1.15),
                    arrowprops=dict(arrowstyle="->", lw=0.75, color="black", alpha=0.9,
                                    shrinkA=0, shrinkB=3))
    except Exception:
        _repel_labels_builtin(ax, x, y, names)

# ======================= Visual 1: Sectional Shape Map ====================
st.markdown("## Sectional Shape Map — Accel vs Grind (colour = tsSPIΔ, size = PI)")

need_cols = {"Horse","Accel","Grind","tsSPI","PI"}
if not need_cols.issubset(metrics.columns):
    st.warning("Shape Map: required columns missing: " + ", ".join(sorted(need_cols - set(metrics.columns))))
else:
    dfm = metrics.loc[:, ["Horse","Accel","Grind","tsSPI","PI"]].copy()
    for c in ["Accel","Grind","tsSPI","PI"]:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
    dfm = dfm.dropna(subset=["Accel","Grind","tsSPI"])
    if dfm.empty:
        st.info("Not enough data to draw the shape map.")
    else:
        dfm["AccelΔ"] = dfm["Accel"] - 100.0
        dfm["GrindΔ"] = dfm["Grind"] - 100.0
        dfm["tsSPIΔ"] = dfm["tsSPI"] - 100.0

        names = dfm["Horse"].astype(str).to_list()
        xv = dfm["AccelΔ"].to_numpy()
        yv = dfm["GrindΔ"].to_numpy()
        cv = dfm["tsSPIΔ"].to_numpy()
        piv = dfm["PI"].fillna(0).to_numpy()

        try:
            span = float(np.nanmax([np.nanmax(np.abs(xv)), np.nanmax(np.abs(yv))]))
        except Exception:
            span = 1.0
        if not np.isfinite(span) or span <= 0: span = 1.0
        lim = max(4.5, float(np.ceil(span/1.5)*1.5))

        DOT_MIN, DOT_MAX = 40.0, 140.0
        pmin, pmax = float(np.nanmin(piv)), float(np.nanmax(piv))
        sizes = np.full_like(xv, DOT_MIN) if (not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax-pmin)<1e-9) \
                else DOT_MIN + (piv - pmin) / (pmax - pmin) * (DOT_MAX - DOT_MIN)

        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        TINT = 0.06
        ax.add_patch(Rectangle((0,0),  lim,  lim, facecolor="#4daf4a", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((-lim,0), lim,  lim, facecolor="#377eb8", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((0,-lim), lim, lim, facecolor="#ff7f00", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((-lim,-lim),lim, lim, facecolor="#984ea3", alpha=TINT, edgecolor="none"))
        ax.axvline(0, color="gray", lw=1.2, ls=(0,(3,3)))
        ax.axhline(0, color="gray", lw=1.2, ls=(0,(3,3)))

        vmin = float(np.nanmin(cv)) if np.isfinite(cv).any() else -1.0
        vmax = float(np.nanmax(cv)) if np.isfinite(cv).any() else  1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax: vmin, vmax = -1.0, 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

        sc = ax.scatter(xv, yv, s=sizes, c=cv, cmap="coolwarm", norm=norm,
                        edgecolor="black", linewidth=0.6, alpha=0.95)

        label_points_neatly(ax, xv, yv, names)

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Acceleration vs field (points)  →")
        ax.set_ylabel("Grind vs field (points)  ↑")
        ax.set_title("Shape Map — +X: Accel advantage, +Y: Grind advantage")

        s_ex = [DOT_MIN, 0.5*(DOT_MIN+DOT_MAX), DOT_MAX]
        h_ex = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                       markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
        ax.legend(h_ex, ["PI: low", "PI: mid", "PI: high"],
                  loc="upper left", frameon=False, fontsize=8)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("tsSPI − 100")
        ax.grid(True, linestyle=":", alpha=0.25)
        st.pyplot(fig)

        # Download
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        st.download_button("Download shape map (PNG)", buf.getvalue(), file_name="shape_map.png", mime="image/png")

        st.caption("Each bubble is a runner. Size = PI. X: Accel vs field; Y: Grind vs field. Colour shows cruise strength (tsSPI vs field).")

# ======================= Visual 2: Pace Curve =============================
st.markdown("## Pace Curve — field average (black) + Top 8 [stub-aware]")

# Build ordered segments list for plotting labels and speeds
segs = []  # (label, length_m, colname)
if grid == "200m":
    for i, (end_m, L) in enumerate(seg_plan):
        col = f"{end_m}_Time"
        if col in work.columns:
            left = D if i == 0 else int(end_m + L)
            segs.append((f"{left}→{end_m}", float(L), col))
    if "Finish_Time" in work.columns:
        segs.append(("200→0 (Finish)", 200.0, "Finish_Time"))
else:
    # 100m: straight 100s plus Finish
    for m in range(D-100, 99, -100):
        c = f"{m}_Time"
        if c in work.columns:
            segs.append((f"{m+100}→{m}", 100.0, c))
    if "Finish_Time" in work.columns:
        segs.append(("100→0 (Finish)", 100.0, "Finish_Time"))

if len(segs) == 0:
    st.info("Not enough *_Time columns to draw the pace curve.")
else:
    # Build field average speeds
    times_df = work[[c for (_,_,c) in segs]].apply(pd.to_numeric, errors="coerce")
    speed_df = pd.DataFrame(index=work.index)
    for (_, L, c) in segs:
        speed_df[c] = L / times_df[c]
    field_avg = speed_df.mean(axis=0).to_numpy()

    # choose top 8: finish pos if present, else PI
    if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
        top8 = metrics.sort_values("Finish_Pos").head(8)
        top8_rule = "Top-8 by Finish_Pos"
    else:
        top8 = metrics.sort_values("PI", ascending=False).head(8)
        top8_rule = "Top-8 by PI"

    x_idx = list(range(len(segs)))
    x_labels = [lab for (lab,_,_) in segs]

    fig2, ax2 = plt.subplots(figsize=(8.8, 5.2))
    ax2.plot(x_idx, field_avg, linewidth=2.2, color="black", label="Field average", marker=None)

    palette = color_cycle(len(top8))
    for i, (_, r) in enumerate(top8.iterrows()):
        # try to match original row by Horse for individual splits
        if "Horse" in work.columns and "Horse" in metrics.columns:
            row0 = work[work["Horse"] == r.get("Horse")]
            row_times = row0.iloc[0] if not row0.empty else r
        else:
            row_times = r

        y_vals=[]
        for (_, L, c) in segs:
            t = pd.to_numeric(row_times.get(c, np.nan), errors="coerce")
            y_vals.append(L/t if pd.notna(t) and t>0 else np.nan)
        ax2.plot(x_idx, y_vals, linewidth=1.1, marker="o", markersize=2.5,
                 label=str(r.get("Horse", "")), color=palette[i])

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Pace over segments (left = early; handles stub if distance % grid ≠ 0)")
    ax2.grid(True, linestyle="--", alpha=0.30)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
    st.pyplot(fig2)

    # Download
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    st.download_button("Download pace curve (PNG)", buf2.getvalue(), file_name="pace_curve.png", mime="image/png")

    st.caption(f"Top-8 plotted: {top8_rule}. Finish segment included explicitly.")

# ======================= Visual 3: Top-8 PI — stacked parts ===============
st.markdown("## Top-8 PI — stacked contributions")

# build weights again (already reflected in Batch 2 metrics; here we just need the numbers)
def pi_weights_distance_and_context(distance_m: float,
                                    acc_median: float | None,
                                    grd_median: float | None) -> dict:
    import math
    def _lerp(a,b,t): return a+(b-a)*float(t)
    def _interpolate(dm, a_dm, a_w, b_dm, b_w):
        span=float(b_dm-a_dm); t=0.0 if span<=0 else (float(dm)-a_dm)/span
        return {"F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
                "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
                "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
                "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t)}
    dm=float(distance_m or 1200)
    if dm<=1000: base={"F200_idx":0.12,"tsSPI":0.35,"Accel":0.36,"Grind":0.17}
    elif dm<1100: base=_interpolate(dm,1000,{"F200_idx":0.12,"tsSPI":0.35,"Accel":0.36,"Grind":0.17},
                                       1100,{"F200_idx":0.10,"tsSPI":0.36,"Accel":0.34,"Grind":0.20})
    elif dm<1200: base=_interpolate(dm,1100,{"F200_idx":0.10,"tsSPI":0.36,"Accel":0.34,"Grind":0.20},
                                       1200,{"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25})
    elif dm==1200: base={"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25}
    else:
        shift_units=max(0.0,(dm-1200.0)/100.0)*0.01
        grind=min(0.25+shift_units,0.40); F200,ACC=0.08,0.30
        ts=max(0.0,1.0-F200-ACC-grind); base={"F200_idx":F200,"tsSPI":ts,"Accel":ACC,"Grind":grind}
    acc_med = float(acc_median) if acc_median is not None and np.isfinite(acc_median) else None
    grd_med = float(grd_median) if grd_median is not None and np.isfinite(grd_median) else None
    if acc_med is not None and grd_med is not None:
        bias=acc_med-grd_med; scale=math.tanh(abs(bias)/6.0); max_shift=0.02*scale
        F200,ts,ACC,GR=base["F200_idx"],base["tsSPI"],base["Accel"],base["Grind"]
        if bias>0:
            delta=min(max_shift, ACC-0.26); ACC-=delta; GR+=delta
        elif bias<0:
            delta=min(max_shift, GR-0.18); GR-=delta; ACC+=delta
        GR=min(GR,0.40); ts=max(0.0, 1.0-F200-ACC-GR)
        base={"F200_idx":F200,"tsSPI":ts,"Accel":ACC,"Grind":GR}
    s=sum(base.values())
    if abs(s-1.0)>1e-6: base={k:v/s for k,v in base.items()}
    return base

acc_med_for_bars = metrics["Accel"].median(skipna=True)
grd_med_for_bars = metrics["Grind"].median(skipna=True)
PI_W_BARS = pi_weights_distance_and_context(float(D), acc_med_for_bars, grd_med_for_bars)

def parts_scaled_to_total(row, total_pi, weights, zero_floor=True):
    raw = {
        "F200_idx": weights["F200_idx"] * (float(row.get("F200_idx", 100.0)) - 100.0),
        "tsSPI":    weights["tsSPI"]    * (float(row.get("tsSPI",    100.0)) - 100.0),
        "Accel":    weights["Accel"]    * (float(row.get("Accel",    100.0)) - 100.0),
        "Grind":    weights["Grind"]    * (float(row.get("Grind",    100.0)) - 100.0),
    }
    if zero_floor:
        raw = {k: max(0.0, v) for k, v in raw.items()}
    s = sum(raw.values())
    if not np.isfinite(total_pi) or total_pi <= 0 or not np.isfinite(s) or s <= 0:
        return {"F200_idx": 0.0, "tsSPI": 0.0, "Accel": 0.0, "Grind": 0.0}
    scale = float(total_pi) / float(s)
    return {k: v * scale for k, v in raw.items()}

top8_pi = metrics.sort_values(["PI","Finish_Pos"], ascending=[False, True]).head(8).copy()
if top8_pi.empty:
    st.info("No PI values available to plot the stacked contributions.")
else:
    horses, totals, is_winner = [], [], []
    stacks = {"F200_idx": [], "tsSPI": [], "Accel": [], "Grind": []}
    for _, r in top8_pi.iterrows():
        total_pi = float(r.get("PI", 0.0))
        parts = parts_scaled_to_total(r, total_pi, PI_W_BARS, zero_floor=True)
        for k in stacks: stacks[k].append(parts[k])
        totals.append(total_pi)
        horses.append(str(r.get("Horse", "")))
        fp = pd.to_numeric(r.get("Finish_Pos", np.nan), errors="coerce")
        is_winner.append(False if pd.isna(fp) else int(fp) == 1)

    fig3, ax3 = plt.subplots(figsize=(max(7.5, 0.95*len(horses)), 4.8))
    x = np.arange(len(horses))
    palette = {"F200_idx": "#6baed6", "tsSPI": "#9e9ac8", "Accel": "#74c476", "Grind": "#fd8d3c"}

    bottoms = np.zeros(len(horses))
    for key, label in [("F200_idx","F200"), ("tsSPI","tsSPI"), ("Accel","Accel"), ("Grind","Grind")]:
        vals = np.array(stacks[key], dtype=float)
        ax3.bar(x, vals, bottom=bottoms, label=label, color=palette[key], edgecolor="black", linewidth=0.4)
        bottoms += vals

    ymax = max(0.1, max(totals)*1.20)
    for i, tot in enumerate(totals):
        if is_winner[i]:
            ax3.add_patch(plt.Rectangle((i-0.5, 0), 1.0, max(tot, bottoms[i]),
                                        fill=False, lw=2.0, ec="#d4af37"))
            horses[i] = f"★ {horses[i]}"
        ax3.text(i, tot + ymax*0.03, f"{tot:.2f}", ha="center", va="bottom", fontsize=9)

    ax3.set_xticks(x); ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylim(0, ymax)
    ax3.set_ylabel("PI (stacked contributions)")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    st.pyplot(fig3)

    buf3 = io.BytesIO(); fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    st.download_button("Download PI stacks (PNG)", buf3.getvalue(), file_name="pi_stacks.png", mime="image/png")
    st.caption("Slices are rescaled to sum exactly to each horse’s PI. ★ = race winner.")

# ======================= Hidden Horses (v2) — Display =======================
st.markdown("## 🕵️ Hidden Horses (v2)")

hh = st.session_state.get("hh", None)
if hh is None or hh.empty:
    st.info("Hidden Horses not available yet — run Batches 1 & 2 with data.")
else:
    # columns to show
    cols_hh = ["Horse","Finish_Pos","PI","GCI","tsSPI","Accel","Grind",
               "SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
    for c in cols_hh:
        if c not in hh.columns:
            hh[c] = np.nan

    # order: flagged tiers first
    tier_order = {"🔥 Top Hidden": 0, "🟡 Notable Hidden": 1, "": 2}
    hh["_tier_rank"] = hh["Tier"].map(tier_order).fillna(2).astype(int)

    show_flagged_only = st.toggle("Show only flagged (Top/Notable)", value=True)
    view = hh.copy()
    if show_flagged_only:
        view = view[view["Tier"] != ""]

    # tidy formatting
    for c in ["PI","GCI","tsSPI","Accel","Grind","SOS","ASI2","TFS","UEI","HiddenScore"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )

    # stable sort: tier → HiddenScore desc → PI desc → Finish_Pos asc
    view = view.sort_values(
        by=["_tier_rank","HiddenScore","PI","Finish_Pos"],
        ascending=[True, False, False, True],
        kind="mergesort"
    ).drop(columns=["_tier_rank"])

    st.dataframe(view[cols_hh], use_container_width=True)

    # small legend + sanity nudge if TFS is NaN due to grid/columns
    if view.empty:
        st.caption("No runners currently flagged by Hidden Horses v2.")
    else:
        st.caption("Flag rules: 🔥 Top Hidden ≥ 1.8 • 🟡 Notable Hidden ≥ 1.2. "
                   "Scores blend SOS, ASI² (bias adversity), late variability (TFS) and UEI.")
        
# ======================= Batch 4 — Database Integration (SQLite) =======================
# Requires Batches 1–3 to have run. Uses st.session_state["metrics"], ["work_df"], ["race_distance_m"],
# ["grid"], ["seg_plan"], and optionally ["race_meta"] (you can set it in Batch 1).
#
# What you get:
# - SQLite schema (races, horses, runs, splits, shape_points)
# - “Save this race to DB” action (idempotent upserts on race & horses; new runs/splits each save)
# - Quick browser: pick a horse, see their historical runs, metrics, and a simple per-race sectional summary

import os
import math
import time
import json
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Guards ----------------
need_keys = ["metrics", "work_df", "race_distance_m", "grid", "seg_plan"]
for k in need_keys:
    if k not in st.session_state:
        st.error(f"Batch 4 needs `{k}` from earlier batches. Please run Batches 1–3.")
        st.stop()

metrics   = st.session_state["metrics"].copy()
work_df   = st.session_state["work_df"].copy()
D         = int(st.session_state["race_distance_m"])
grid      = st.session_state["grid"]         # "100m" or "200m"
seg_plan  = list(st.session_state["seg_plan"])
race_meta = st.session_state.get("race_meta", {})  # optional dict you can set upstream

# ---- Recommend a default DB path in the app folder (configurable in the UI) ----
with st.sidebar:
    st.markdown("### Database")
    db_path = st.text_input("SQLite DB file", value="race_edge.sqlite")
    if not db_path.strip():
        st.stop()

# ---------------- Schema ----------------
DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS races (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  venue         TEXT,
  meeting_date  TEXT,          -- ISO (YYYY-MM-DD)
  race_num      INTEGER,
  distance_m    INTEGER NOT NULL,
  surface       TEXT,
  going         TEXT,
  class        TEXT,
  grid          TEXT NOT NULL, -- '100m' | '200m'
  meta_json     TEXT,
  created_ts    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_races_key ON races(meeting_date, venue, race_num, distance_m, grid);

CREATE TABLE IF NOT EXISTS horses (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  country     TEXT,
  yob         INTEGER,
  canonical   TEXT,            -- normalized key for de-dup
  created_ts  REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_horses_canonical ON horses(canonical);

CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  race_id       INTEGER NOT NULL REFERENCES races(id) ON DELETE CASCADE,
  horse_id      INTEGER NOT NULL REFERENCES horses(id) ON DELETE CASCADE,
  draw          TEXT,
  jockey        TEXT,
  trainer       TEXT,
  weight_carried REAL,
  finish_pos    INTEGER,
  race_time_s   REAL,          -- total race time (if available)
  f200_idx      REAL,
  tsspi         REAL,
  accel         REAL,
  grind         REAL,
  pi            REAL,
  gci           REAL,
  created_ts    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_race ON runs(race_id);
CREATE INDEX IF NOT EXISTS idx_runs_horse ON runs(horse_id);

CREATE TABLE IF NOT EXISTS splits (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  marker_end_m INTEGER,        -- e.g., 1400 for 1600→1400 block (end marker)
  length_m     REAL NOT NULL,  -- 100 / 200 / stub (e.g. 160 for 1160 distances)
  time_s       REAL,           -- split time in seconds
  col_name     TEXT,           -- original column (e.g., '1400_Time' or 'Finish_Time')
  created_ts   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_splits_run ON splits(run_id);

CREATE TABLE IF NOT EXISTS shape_points (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  accel_delta REAL,            -- Accel - 100
  grind_delta REAL,            -- Grind - 100
  tsspi_delta REAL,            -- tsSPI - 100
  created_ts  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shape_run ON shape_points(run_id);
"""

# ---------------- Utilities ----------------
def _now_ts() -> float:
    return time.time()

def _canonical_name(name: str) -> str:
    if pd.isna(name) or not str(name).strip():
        return ""
    s = str(name).lower().strip()
    # keep only letters/numbers/space and collapse spaces
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    s = " ".join(s.split())
    return s

def _connect(dbfile: str) -> sqlite3.Connection:
    conn = sqlite3.connect(dbfile)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(dbfile: str):
    conn = _connect(dbfile)
    try:
        conn.executescript(DDL)
        conn.commit()
    finally:
        conn.close()

def upsert_horse(conn: sqlite3.Connection, name: str, country: str|None=None, yob: int|None=None) -> int:
    can = _canonical_name(name)
    if not can:
        # unnamed / missing row — store placeholder to keep referential integrity
        can = f"unknown-{int(_now_ts()*1000)}"
        name = name if str(name).strip() else "Unknown"
    cur = conn.cursor()
    cur.execute("SELECT id FROM horses WHERE canonical = ?", (can,))
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute(
        "INSERT INTO horses(name, country, yob, canonical, created_ts) VALUES (?,?,?,?,?)",
        (name, country, yob, can, _now_ts())
    )
    conn.commit()
    return int(cur.lastrowid)

def upsert_race(conn: sqlite3.Connection, *, meeting_date: str|None, venue: str|None,
                race_num: int|None, distance_m: int, grid: str, meta: dict|None) -> int:
    # We insert a new race each time (history preserved), but we try to reuse if exact same key exists.
    cur = conn.cursor()
    cur.execute("""SELECT id FROM races
                   WHERE meeting_date IS ? AND venue IS ? AND race_num IS ?
                         AND distance_m = ? AND grid = ?""",
                (meeting_date, venue, race_num, distance_m, grid))
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute("""INSERT INTO races(venue, meeting_date, race_num, distance_m, surface, going, class, grid, meta_json, created_ts)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    venue, meeting_date, race_num, int(distance_m),
                    meta.get("surface") if meta else None,
                    meta.get("going") if meta else None,
                    meta.get("class") if meta else None,
                    grid,
                    json.dumps(meta or {}, ensure_ascii=False),
                    _now_ts()
                ))
    conn.commit()
    return int(cur.lastrowid)

def insert_run(conn: sqlite3.Connection, *, race_id: int, horse_id: int, row: pd.Series) -> int:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO runs(race_id, horse_id, draw, jockey, trainer, weight_carried, finish_pos,
                            race_time_s, f200_idx, tsspi, accel, grind, pi, gci, created_ts)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            race_id, horse_id,
            row.get("Draw"),
            row.get("Jockey"),
            row.get("Trainer"),
            _safe_float(row.get("WeightCarried")),
            _safe_int(row.get("Finish_Pos")),
            _safe_float(row.get("RaceTime_s")),
            _safe_float(row.get("F200_idx")),
            _safe_float(row.get("tsSPI")),
            _safe_float(row.get("Accel")),
            _safe_float(row.get("Grind")),
            _safe_float(row.get("PI")),
            _safe_float(row.get("GCI")),
            _now_ts()
        )
    )
    conn.commit()
    return int(cur.lastrowid)

def _safe_float(x):
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None

def _safe_int(x):
    try:
        i = int(float(x))
        return i
    except Exception:
        return None

def insert_splits_for_run(conn: sqlite3.Connection, *, run_id: int, grid: str,
                          seg_plan: list[tuple[int, float]], row_times: pd.Series):
    """
    For 200m grid: seg_plan = [(end_marker, length_m), ...], Finish_Time is 200→0.
    For 100m grid: we infer 100m ends from D like in Batch 3 (we rely on row_times having those cols).
    """
    cur = conn.cursor()
    created = _now_ts()
    if grid == "200m":
        for end_m, L in seg_plan:
            col = f"{end_m}_Time"
            t = _safe_float(row_times.get(col))
            cur.execute(
                "INSERT INTO splits(run_id, marker_end_m, length_m, time_s, col_name, created_ts) VALUES (?,?,?,?,?,?)",
                (run_id, int(end_m), float(L), t, col, created)
            )
        # Finish
        tfin = _safe_float(row_times.get("Finish_Time"))
        cur.execute(
            "INSERT INTO splits(run_id, marker_end_m, length_m, time_s, col_name, created_ts) VALUES (?,?,?,?,?,?)",
            (run_id, 0, 200.0, tfin, "Finish_Time", created)
        )
    else:
        # 100m: build expected ends from D
        for m in range(int(D) - 100, 99, -100):
            col = f"{m}_Time"
            t = _safe_float(row_times.get(col))
            cur.execute(
                "INSERT INTO splits(run_id, marker_end_m, length_m, time_s, col_name, created_ts) VALUES (?,?,?,?,?,?)",
                (run_id, int(m), 100.0, t, col, created)
            )
        tfin = _safe_float(row_times.get("Finish_Time"))
        cur.execute(
            "INSERT INTO splits(run_id, marker_end_m, length_m, time_s, col_name, created_ts) VALUES (?,?,?,?,?,?)",
            (run_id, 0, 100.0 if grid=="100m" else 200.0, tfin, "Finish_Time", created)
        )
    conn.commit()

def insert_shape_point(conn: sqlite3.Connection, *, run_id: int, accel: float|None, grind: float|None, tsspi: float|None):
    a = _safe_float(accel); g = _safe_float(grind); t = _safe_float(tsspi)
    if a is None or g is None or t is None:
        return
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO shape_points(run_id, accel_delta, grind_delta, tsspi_delta, created_ts) VALUES (?,?,?,?,?)",
        (run_id, a-100.0, g-100.0, t-100.0, _now_ts())
    )
    conn.commit()

# ---------------- Save this race ----------------
st.markdown("### 💾 Persist this race to the database")
colA, colB = st.columns([1,1])
with colA:
    venue   = st.text_input("Venue", value=str(race_meta.get("venue","")).strip())
    mdate   = st.text_input("Meeting date (YYYY-MM-DD)", value=str(race_meta.get("meeting_date","")).strip())
with colB:
    race_no = st.number_input("Race number", min_value=1, max_value=20, value=int(race_meta.get("race_num", 1)))
    surface = st.text_input("Surface (opt.)", value=str(race_meta.get("surface","")).strip())

going   = st.text_input("Going (opt.)", value=str(race_meta.get("going","")).strip())
rclass  = st.text_input("Class (opt.)", value=str(race_meta.get("class","")).strip())

meta_preview = {
    "surface": surface or None,
    "going": going or None,
    "class": rclass or None
}
save_clicked = st.button("Save race, runners, sectionals & shape points")

if save_clicked:
    try:
        init_db(db_path)
        conn = _connect(db_path)
        # Upsert race
        race_id = upsert_race(
            conn,
            meeting_date=mdate or None,
            venue=venue or None,
            race_num=int(race_no) if race_no else None,
            distance_m=int(D),
            grid=grid,
            meta=meta_preview
        )

        # Choose an indexer to map back row splits: prefer work_df by Horse name
        # (we assume Batch 1 normalized Horse & Finish_Pos where available)
        work_by_horse = None
        if "Horse" in work_df.columns:
            work_by_horse = work_df.set_index("Horse", drop=False)

        # Save each runner
        inserted = 0
        for _, r in metrics.iterrows():
            horse_name = str(r.get("Horse","")).strip() or "Unknown"
            horse_id = upsert_horse(conn, horse_name)

            run_id = insert_run(conn, race_id=race_id, horse_id=horse_id, row=r)

            # Find this runner's raw split row to capture per-split times
            if work_by_horse is not None and horse_name in work_by_horse.index:
                row_times = work_by_horse.loc[horse_name]
            else:
                row_times = r  # fallback

            insert_splits_for_run(conn, run_id=run_id, grid=grid, seg_plan=seg_plan, row_times=row_times)
            insert_shape_point(conn, run_id=run_id,
                               accel=r.get("Accel"), grind=r.get("Grind"), tsspi=r.get("tsSPI"))
            inserted += 1

        conn.close()
        st.success(f"Saved race (id #{race_id}) and {inserted} runner(s) to `{db_path}`.")
    except Exception as e:
        st.error("Failed to save to database.")
        st.exception(e)

# ---------------- Quick browser: a horse’s history ----------------
st.markdown("---")
st.markdown("### 🔎 Browse horse history")

browse_ok = False
try:
    init_db(db_path)
    conn = _connect(db_path)
    horses_df = pd.read_sql_query("SELECT id, name FROM horses ORDER BY name COLLATE NOCASE", conn)
    conn.close()
    browse_ok = True
except Exception as e:
    st.info("Database not initialized yet. Save a race first.")

if browse_ok and not horses_df.empty:
    name_to_id = dict(zip(horses_df["name"], horses_df["id"]))
    pick = st.selectbox("Select horse", options=["— pick —"] + horses_df["name"].tolist())
    if pick and pick != "— pick —":
        hid = name_to_id[pick]
        try:
            conn = _connect(db_path)
            q = """
            SELECT
              r.meeting_date, r.venue, r.race_num, r.distance_m, r.grid,
              ru.finish_pos, ru.race_time_s, ru.f200_idx, ru.tsspi, ru.accel, ru.grind, ru.pi, ru.gci,
              ru.id as run_id
            FROM runs ru
            JOIN races r ON r.id = ru.race_id
            WHERE ru.horse_id = ?
            ORDER BY COALESCE(r.meeting_date,'9999-12-31') DESC, r.race_num DESC
            """
            hist = pd.read_sql_query(q, conn, params=(hid,))
            st.markdown(f"#### History for **{pick}**")
            if hist.empty:
                st.info("No runs found yet.")
            else:
                show = hist.copy()
                # Nice formatting
                for c in ["race_time_s","f200_idx","tsspi","accel","grind","pi","gci"]:
                    if c in show.columns:
                        show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if c!="race_time_s" else f"{x:.2f}"))
                st.dataframe(show, use_container_width=True)

                # Optional: simple sectional preview for the most recent run
                st.markdown("**Most recent run — sectional times**")
                last_run_id = int(hist.iloc[0]["run_id"])
                segs = pd.read_sql_query(
                    "SELECT marker_end_m, length_m, time_s, col_name FROM splits WHERE run_id = ? ORDER BY (marker_end_m IS 0), marker_end_m DESC",
                    conn, params=(last_run_id,)
                )
                if segs.empty:
                    st.info("No splits stored for this run.")
                else:
                    segs["speed_mps"] = segs.apply(lambda r: (float(r["length_m"])/float(r["time_s"])) if (pd.notna(r["time_s"]) and r["time_s"]>0) else np.nan, axis=1)
                    segs_view = segs.copy()
                    for c in ["time_s","speed_mps","length_m"]:
                        segs_view[c] = pd.to_numeric(segs_view[c], errors="coerce").map(lambda x: "" if pd.isna(x) else (f"{x:.2f}" if c!="length_m" else f"{x:.0f}"))
                    st.dataframe(segs_view, use_container_width=True)
            conn.close()
        except Exception as e:
            st.error("Failed to read history.")
            st.exception(e)
else:
    if browse_ok:
        st.info("No horses yet — save a race first.")

# ---------------- Notes ----------------
st.caption(
    "DB schema: races (1) — runs (many) — splits (many per run) — shape_points (1 per run). "
    "We upsert horses by a normalized canonical name; races are keyed by (date, venue, number, distance, grid)."
)

# ======================= Batch 5 — DB Reports + Horse Profile =======================
# Needs the SQLite created by Batch 4. You can run Batch 5 independently if you only want to browse/export.
# Provides:
# 1) Race Report (from DB) -> PDF download
# 2) Horse Profile (history table + historical shape map) -> PDF download

import io
import math
import json
import time
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ---------------- Config / DB path ----------------
with st.sidebar:
    st.markdown("### Reports & Profiles")
    db_path = st.text_input("SQLite DB file (same as Batch 4)", value=st.session_state.get("db_path", "race_edge.sqlite"))
    st.session_state["db_path"] = db_path

def _connect(dbfile: str) -> sqlite3.Connection:
    conn = sqlite3.connect(dbfile)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ---------------- Appearance helpers ----------------
def _color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color',
            ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out=[]; i=0
    while len(out)<n: out.append(base[i%len(base)]); i+=1
    return out

def _shape_scatter(img_width=(7.6, 6.0), *, df_points: pd.DataFrame, title: str):
    """
    df_points columns expected:
      accel_delta, grind_delta, tsspi_delta, pi(optional), label(optional), when(optional)
    Returns (fig, bytes_png)
    """
    d = df_points.copy()
    d["accel_delta"] = pd.to_numeric(d["accel_delta"], errors="coerce")
    d["grind_delta"] = pd.to_numeric(d["grind_delta"], errors="coerce")
    d["tsspi_delta"] = pd.to_numeric(d["tsspi_delta"], errors="coerce")
    d = d.dropna(subset=["accel_delta","grind_delta","tsspi_delta"])
    if d.empty:
        return None, None

    xv = d["accel_delta"].to_numpy()
    yv = d["grind_delta"].to_numpy()
    cv = d["tsspi_delta"].to_numpy()
    labels = d.get("label", pd.Series([None]*len(d))).astype(str).tolist()
    sizes_pi = d.get("pi", pd.Series([np.nan]*len(d))).astype(float).to_numpy()

    try:
        span = float(np.nanmax([np.nanmax(np.abs(xv)), np.nanmax(np.abs(yv))]))
    except Exception:
        span = 1.0
    if not np.isfinite(span) or span <= 0: span = 1.0
    lim = max(4.5, float(np.ceil(span/1.5)*1.5))

    # size by PI if available
    DOT_MIN, DOT_MAX = 40.0, 140.0
    if np.isfinite(sizes_pi).any():
        pmin, pmax = float(np.nanmin(sizes_pi)), float(np.nanmax(sizes_pi))
        sizes = np.full_like(xv, DOT_MIN) if (not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax-pmin)<1e-9) \
                else DOT_MIN + (sizes_pi - pmin) / (pmax - pmin) * (DOT_MAX - DOT_MIN)
    else:
        sizes = np.full_like(xv, DOT_MIN)

    vmin = float(np.nanmin(cv)) if np.isfinite(cv).any() else -1.0
    vmax = float(np.nanmax(cv)) if np.isfinite(cv).any() else  1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax: vmin, vmax = -1.0, 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=img_width)
    TINT = 0.06
    ax.add_patch(Rectangle((0,0),  lim,  lim, facecolor="#4daf4a", alpha=TINT, edgecolor="none"))
    ax.add_patch(Rectangle((-lim,0), lim,  lim, facecolor="#377eb8", alpha=TINT, edgecolor="none"))
    ax.add_patch(Rectangle((0,-lim), lim, lim, facecolor="#ff7f00", alpha=TINT, edgecolor="none"))
    ax.add_patch(Rectangle((-lim,-lim),lim, lim, facecolor="#984ea3", alpha=TINT, edgecolor="none"))
    ax.axvline(0, color="gray", lw=1.2, ls=(0,(3,3)))
    ax.axhline(0, color="gray", lw=1.2, ls=(0,(3,3)))

    sc = ax.scatter(xv, yv, s=sizes, c=cv, cmap="coolwarm", norm=norm,
                    edgecolor="black", linewidth=0.6, alpha=0.95)

    # labels (only if few points to avoid clutter)
    if len(labels) <= 18:
        for xi, yi, lab in zip(xv, yv, labels):
            if lab and lab != "None":
                ax.text(xi, yi, lab, fontsize=8.2,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Acceleration vs field (points)  →")
    ax.set_ylabel("Grind vs field (points)  ↑")
    ax.set_title(title)

    # legend for PI size
    s_ex = [DOT_MIN, 0.5*(DOT_MIN+DOT_MAX), DOT_MAX]
    h_ex = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                   markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
    ax.legend(h_ex, ["PI: low","PI: mid","PI: high"], loc="upper left", frameon=False, fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("tsSPI − 100")
    ax.grid(True, linestyle=":", alpha=0.25)

    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    return fig, buf.getvalue()

# ---------------- PDF helpers ----------------
def _pdf_safe_number(x, ndigits=3):
    try:
        v = float(x)
        if not math.isfinite(v): return ""
        fmt = f"{{:.{ndigits}f}}"
        return fmt.format(v)
    except Exception:
        return ""

def make_race_pdf(*, dbfile: str, race_id: int, shape_png: bytes|None=None):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import cm
    except Exception:
        st.error("`reportlab` is required. Install with: `pip install reportlab>=4.2.0`")
        return None

    # fetch race + runs
    conn = _connect(dbfile)
    race = pd.read_sql_query("SELECT * FROM races WHERE id = ?", conn, params=(race_id,))
    runs = pd.read_sql_query("""
        SELECT h.name as horse, ru.finish_pos, ru.race_time_s, ru.f200_idx, ru.tsspi, ru.accel, ru.grind, ru.pi, ru.gci
        FROM runs ru
        JOIN horses h ON h.id = ru.horse_id
        WHERE ru.race_id = ?
        ORDER BY COALESCE(ru.finish_pos, 1e9), ru.pi DESC
    """, conn, params=(race_id,))
    conn.close()

    if race.empty:
        st.error("Race not found.")
        return None

    R = race.iloc[0]
    hdr_bits = [
        f"Venue: <b>{R.get('venue','') or ''}</b>",
        f"Date: <b>{R.get('meeting_date','') or ''}</b>",
        f"Race: <b>{R.get('race_num','') or ''}</b>",
        f"Distance: <b>{int(R.get('distance_m',0))}m</b>",
        f"Grid: <b>{R.get('grid','')}</b>",
    ]
    meta = {}
    try:
        meta = json.loads(R.get("meta_json") or "{}")
    except Exception:
        meta = {}
    if meta.get("surface"): hdr_bits.append(f"Surface: <b>{meta['surface']}</b>")
    if meta.get("going"):   hdr_bits.append(f"Going: <b>{meta['going']}</b>")
    if meta.get("class"):   hdr_bits.append(f"Class: <b>{meta['class']}</b>")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()
    H = styles["Heading1"]; H.fontSize = 18; H.leading = 22; H.spaceAfter = 6
    H3 = styles["Heading3"]; P = styles["BodyText"]; P.fontSize = 9; P.leading = 12
    story = []

    story.append(Paragraph("Race Report", H))
    story.append(Paragraph(" &nbsp; • ".join(hdr_bits), P))
    story.append(Spacer(0, 6))

    # Table of runners
    if not runs.empty:
        tab = runs.copy()
        for c in ["race_time_s","f200_idx","tsspi","accel","grind","pi","gci"]:
            tab[c] = tab[c].map(lambda x: "" if pd.isna(x) else (_pdf_safe_number(x, 2) if c!="race_time_s" else _pdf_safe_number(x, 2)))
        data = [list(["Horse","Finish_Pos","RaceTime_s","F200_idx","tsSPI","Accel","Grind","PI","GCI"])] + \
               tab.fillna("").astype(str).values.tolist()
        from reportlab.platypus import Table, TableStyle
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('ALIGN', (2,1), (-1,-1), 'RIGHT'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.whitesmoke),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
        ]))
        story.append(t)
        story.append(Spacer(0, 10))

    # Shape map if provided
    if shape_png:
        from reportlab.platypus import Image
        story.append(Paragraph("Shape Map — Accel vs Grind (colour = tsSPIΔ)", H3))
        story.append(Image(io.BytesIO(shape_png), width=24*cm, height=18*cm, kind="proportional"))
        story.append(Spacer(0, 8))

    # Footnote
    story.append(Paragraph(
        "Notes: Accel/Grind/tsSPI are indices vs field (100=par). PI v3.1 uses distance+context weights; "
        "GCI is a 0–10 composite of time merit, late quality, mid-race strength, and efficiency.",
        P
    ))

    doc.build(story)
    buf.seek(0)
    return buf

def make_horse_pdf(*, horse_name: str, hist_table: pd.DataFrame, shape_png: bytes|None):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import cm
    except Exception:
        st.error("`reportlab` is required. Install with: `pip install reportlab>=4.2.0`")
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()
    H = styles["Heading1"]; H.fontSize = 18; H.leading = 22; H.spaceAfter = 6
    H3 = styles["Heading3"]; P = styles["BodyText"]; P.fontSize = 9; P.leading = 12
    story = []

    story.append(Paragraph(f"Horse Profile: <b>{horse_name}</b>", H))
    story.append(Spacer(0, 6))

    # History table
    if not hist_table.empty:
        tab = hist_table.copy()
        for c in ["race_time_s","f200_idx","tsspi","accel","grind","pi","gci"]:
            if c in tab.columns:
                tab[c] = tab[c].map(lambda x: "" if pd.isna(x) else (_pdf_safe_number(x, 2) if c!="race_time_s" else _pdf_safe_number(x, 2)))
        cols = [c for c in ["meeting_date","venue","race_num","distance_m","grid","finish_pos",
                            "race_time_s","f200_idx","tsspi","accel","grind","pi","gci"] if c in tab.columns]
        data = [cols] + tab[cols].fillna("").astype(str).values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('ALIGN', (6,1), (-1,-1), 'RIGHT'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.whitesmoke),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
        ]))
        story.append(t)
        story.append(Spacer(0, 10))

    # Historical shape map
    if shape_png:
        story.append(Paragraph("Historical Shape Map — Accel vs Grind (colour = tsSPIΔ)", H3))
        story.append(Image(io.BytesIO(shape_png), width=24*cm, height=18*cm, kind="proportional"))
        story.append(Spacer(0, 8))

    # Footnote
    story.append(Paragraph(
        "Each point represents a run (size = PI where available). Axes show deviation vs field in points "
        "(100=par baseline). Colour is tsSPIΔ.", P
    ))

    doc.build(story)
    buf.seek(0)
    return buf

# ======================= 1) Race Report from DB ============================
st.markdown("## 📄 Race Report (from DB)")

race_picker_ok = False
try:
    conn = _connect(db_path)
    races = pd.read_sql_query("""
        SELECT id,
               COALESCE(meeting_date,'????-??-??') as meeting_date,
               COALESCE(venue,'?') as venue,
               COALESCE(race_num, '?') as race_num,
               distance_m, grid
        FROM races
        ORDER BY COALESCE(meeting_date,'9999-12-31') DESC, venue COLLATE NOCASE ASC, race_num ASC
    """, conn)
    conn.close()
    race_picker_ok = True
except Exception as e:
    st.info("Could not open DB. Check the path above.")

if race_picker_ok and not races.empty:
    races["label"] = races.apply(lambda r: f"{r['meeting_date']} | {r['venue']} R{r['race_num']} | {r['distance_m']}m [{r['grid']}]", axis=1)
    label_to_id = dict(zip(races["label"], races["id"]))
    choice = st.selectbox("Pick a stored race", options=["— pick —"] + races["label"].tolist())

    if choice and choice != "— pick —":
        rid = label_to_id[choice]

        # Build shape map points directly from DB: use current race's runs (AccelΔ, GrindΔ, tsSPIΔ + PI)
        conn = _connect(db_path)
        pts = pd.read_sql_query("""
            SELECT h.name as label,
                   (ru.accel-100.0) as accel_delta,
                   (ru.grind-100.0) as grind_delta,
                   (ru.tsspi-100.0) as tsspi_delta,
                   ru.pi
            FROM runs ru
            JOIN horses h ON h.id = ru.horse_id
            WHERE ru.race_id = ?
        """, conn, params=(rid,))
        conn.close()

        fig, png = _shape_scatter(df_points=pts, title="This Race — Accel vs Grind (colour = tsSPIΔ, size = PI)")

        if fig is not None:
            st.pyplot(fig)
            st.download_button("Download race shape map (PNG)", png, file_name="race_shape_map.png", mime="image/png")

        # Compose and download PDF
        pdf_buf = make_race_pdf(dbfile=db_path, race_id=rid, shape_png=png)
        if pdf_buf is not None:
            st.download_button("📥 Download Race PDF", data=pdf_buf.getvalue(),
                               file_name=f"Race_Report_{rid}.pdf", mime="application/pdf")

# ======================= 2) Horse Profile (history + shape) ================
st.markdown("---")
st.markdown("## 🐎 Horse Profile (historical)")

horse_picker_ok = False
try:
    conn = _connect(db_path)
    horses_df = pd.read_sql_query("SELECT id, name FROM horses ORDER BY name COLLATE NOCASE", conn)
    conn.close()
    horse_picker_ok = True
except Exception as e:
    st.info("Could not open DB. Check the path above.")

if horse_picker_ok and not horses_df.empty:
    name_to_id = dict(zip(horses_df["name"], horses_df["id"]))
    pick = st.selectbox("Select horse", options=["— pick —"] + horses_df["name"].tolist(), key="horse_profile_pick")

    if pick and pick != "— pick —":
        hid = name_to_id[pick]
        conn = _connect(db_path)
        hist = pd.read_sql_query("""
            SELECT
              r.id as race_id, r.meeting_date, r.venue, r.race_num, r.distance_m, r.grid,
              ru.finish_pos, ru.race_time_s, ru.f200_idx, ru.tsspi, ru.accel, ru.grind, ru.pi, ru.gci,
              ru.id as run_id
            FROM runs ru
            JOIN races r ON r.id = ru.race_id
            WHERE ru.horse_id = ?
            ORDER BY COALESCE(r.meeting_date,'9999-12-31') ASC, r.race_num ASC
        """, conn, params=(hid,))

        # shape points (if explicit historical shape_points exist, prefer those; else derive from runs table)
        sp = pd.read_sql_query("""
            SELECT sp.run_id, sp.accel_delta, sp.grind_delta, sp.tsspi_delta
            FROM shape_points sp
            JOIN runs ru ON ru.id = sp.run_id
            WHERE ru.horse_id = ?
        """, conn, params=(hid,))
        conn.close()

        st.markdown(f"### History for **{pick}**")
        if hist.empty:
            st.info("No runs found.")
        else:
            show = hist.copy()
            for c in ["race_time_s","f200_idx","tsspi","accel","grind","pi","gci"]:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if c!="race_time_s" else f"{x:.2f}"))
            st.dataframe(show, use_container_width=True)

            # Build points df for plotting
            if not sp.empty:
                pts = hist.merge(sp, how="left", left_on="run_id", right_on="run_id")
                pts["accel_delta"] = pts["accel_delta"].fillna(pts["accel"] - 100.0)
                pts["grind_delta"] = pts["grind_delta"].fillna(pts["grind"] - 100.0)
                pts["tsspi_delta"] = pts["tsspi_delta"].fillna(pts["tsspi"] - 100.0)
            else:
                pts = hist.copy()
                pts["accel_delta"] = pts["accel"] - 100.0
                pts["grind_delta"] = pts["grind"] - 100.0
                pts["tsspi_delta"] = pts["tsspi"] - 100.0

            # label by meeting date + distance
            pts["label"] = pts.apply(lambda r: f"{r['meeting_date'] or ''} | {int(r['distance_m'])}m", axis=1)
            figH, pngH = _shape_scatter(df_points=pts[["accel_delta","grind_delta","tsspi_delta","pi","label"]],
                                        title=f"{pick} — Historical Shape Map")

            if figH is not None:
                st.pyplot(figH)
                st.download_button("Download horse shape map (PNG)", pngH,
                                   file_name=f"{pick.replace(' ','_')}_shape_history.png", mime="image/png")

            # PDF download for horse
            pdf_buf = make_horse_pdf(horse_name=pick, hist_table=hist, shape_png=pngH)
            if pdf_buf is not None:
                st.download_button("📥 Download Horse PDF", data=pdf_buf.getvalue(),
                                   file_name=f"Horse_Profile_{pick.replace(' ','_')}.pdf",
                                   mime="application/pdf")

# ======================= Batch 6 — Stability, State & Polish =======================
# Append AFTER Batch 5

import streamlit as st
import datetime
import sqlite3

# ---------------- Version Tag & Header ----------------
st.markdown("---")
st.markdown("### 🧭 Race Edge — v3.1 Unified Stable Build")
st.caption(
    "Handles both 100 m and 200 m splits • Distance + Context PI v3.1 • Hidden Horses v2 • "
    "Database & Reports • Optimized caching and stability layer."
)

# ---------------- Session State Defaults ----------------
if "last_upload" not in st.session_state:
    st.session_state["last_upload"] = None
if "db_path" not in st.session_state:
    st.session_state["db_path"] = "race_edge.sqlite"
if "runtime_stats" not in st.session_state:
    st.session_state["runtime_stats"] = []

def log_runtime_event(label: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state["runtime_stats"].append(f"[{ts}] {label}")
    # keep the last 12 events to avoid UI bloat
    if len(st.session_state["runtime_stats"]) > 12:
        st.session_state["runtime_stats"] = st.session_state["runtime_stats"][-12:]

# ---------------- Safe DB Check ----------------
def verify_db_access(db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        st.sidebar.warning(f"⚠ DB not reachable: {e}")
        return False

# ---------------- Cache helpers ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def cached_read_csv(file):
    import pandas as pd
    return pd.read_csv(file)

@st.cache_resource
def cached_db_connection(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ---------------- Error Guard ----------------
def safe_exec(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"❌ Operation failed: {e}")
        return None

# ---------------- Diagnostics Panel ----------------
with st.sidebar.expander("🩺 Diagnostics"):
    st.markdown("Recent runtime events:")
    if st.session_state["runtime_stats"]:
        for line in reversed(st.session_state["runtime_stats"]):
            st.text(line)
    else:
        st.caption("No events logged yet.")
    if st.button("Clear logs"):
        st.session_state["runtime_stats"] = []
        st.success("Runtime logs cleared.")

# ---------------- Automatic DB Verification ----------------
if verify_db_access(st.session_state["db_path"]):
    log_runtime_event("Database verified OK")
else:
    log_runtime_event("Database access failed")

# ---------------- Friendly Footer ----------------
st.markdown("---")
st.markdown(
    "© 2025 Race Edge Analytics — Built for sectional-based performance insight. "
    "All data locally cached. Version <b>3.1 Unified Stable</b>.",
    unsafe_allow_html=True,
)

