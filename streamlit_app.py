import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import io, math, re

# ======================= Page config =======================
st.set_page_config(
    page_title="Race Edge — 200m splits (+stub) | PI v3.1 + GCI + Hidden Horses v2 + Report",
    layout="wide"
)

# ======================= Small helpers =====================
def as_num(x): return pd.to_numeric(x, errors="coerce")

def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color',
            ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out=[]; i=0
    while len(out)<n: out.append(base[i%len(base)]); i+=1
    return out

def clamp(v, lo, hi): return max(lo, min(hi, float(v)))

def mad_std(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size==0: return np.nan
    med=np.median(x); mad=np.median(np.abs(x-med)); return 1.4826*mad

def winsorize(s, p_lo=0.10, p_hi=0.90):
    lo=s.quantile(p_lo); hi=s.quantile(p_hi); return s.clip(lower=lo, upper=hi)

def _lerp(a,b,t): return a+(b-a)*float(t)

def _interpolate_weights(dm, a_dm, a_w, b_dm, b_w):
    span=float(b_dm-a_dm); t=0.0 if span<=0 else (float(dm)-a_dm)/span
    return {"F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
            "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
            "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
            "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t)}

def _dbg(enabled, label, obj=None):
    if enabled:
        st.write(f"🔧 {label}")
        if obj is not None: st.write(obj)

# ======================= Sidebar ===========================
with st.sidebar:
    st.markdown("### Upload (200 m grid + optional total)")
    up = st.file_uploader(
        "CSV/XLSX with splits like 1800_Time, 1600_Time, …, 200_Time, and Finish_Time (or a race total).",
        type=["csv","xlsx","xls"]
    )
    race_distance_input = st.number_input(
        "Race Distance (m)",
        min_value=800, max_value=4000, step=50, value=1600,
        help="Any distance (e.g., 1160, 1250, 1450, 1750, 1900). If not a multiple of 200, the first split is a stub."
    )
    SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
    DEBUG = st.toggle("Debug info", value=False)

if not up: st.stop()

# ======================= Load ==============================
try:
    work = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    st.success("File loaded.")
except Exception:
    st.error("Failed to read file."); st.stop()

# ======================= Time parsing ======================
def parse_time_to_seconds(v):
    """Accepts 12.34 / 12,34 / m:ss(.xx) / h:mm:ss(.xx) / strings; returns seconds float or NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
    s = str(v).strip()
    if s == "": return np.nan
    s = s.replace(",", ".")
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except Exception:
            return np.nan
        if len(parts) == 3:   # h:mm:ss(.xx)
            h, m, sec = parts
            return h*3600 + m*60 + sec
        elif len(parts) == 2: # m:ss(.xx)
            m, sec = parts
            return m*60 + sec
        else:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

# ======================= Header normalizer =================
def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

FINISH_SPLIT_ALIASES = {
    "finishsplit","finish200","last200","final200","home200","tofinish","200mfinish","fin200"
}
FINISH_TOTAL_ALIASES = {
    "finishtime","racetime","totaltime","overalltime","finaltime","time","finish"
}

def normalize_headers(df: pd.DataFrame, D_m: int) -> pd.DataFrame:
    rename = {}
    for col in list(df.columns):
        raw = str(col).strip()
        cx = _canon(raw)

        # Finish position
        if re.fullmatch(r'(?i)(finish\s*pos(ition)?|pos(ition)?|fin_pos|finishpos)', raw):
            rename[col] = "Finish_Pos"; continue

        # Horse
        if re.fullmatch(r'(?i)(horse|runner|name)', raw):
            rename[col] = "Horse"; continue

        # Finish split vs total
        if (cx in FINISH_SPLIT_ALIASES) or re.fullmatch(r'(?i)(finish\s*split|finish200|last200|final200|home200)', raw):
            rename[col] = "Finish_Time"; continue
        if (cx in FINISH_TOTAL_ALIASES) or re.fullmatch(r'(?i)(finish\s*time|race\s*time|total\s*time)', raw):
            rename[col] = "RaceTotal_raw"; continue

        # Generic "{metres}[m][ _-]?(time|split)?" → "{metres}_Time"
        m = re.match(r'(?i).*?(\d{2,4})\s*m?\s*[_\s-]?\s*(time|split)?$', raw)
        if m:
            metres = int(m.group(1))
            if 1 <= metres <= int(D_m):
                rename[col] = f"{metres}_Time"; continue

        # Camel case "1600mTime"
        m2 = re.match(r'(?i)^(\d{2,4})m(Time|Split)$', raw)
        if m2:
            metres = int(m2.group(1))
            if 1 <= metres <= int(D_m):
                rename[col] = f"{metres}_Time"; continue

    out = df.rename(columns=rename)
    out = out.loc[:, ~out.columns.duplicated()]  # drop duplicate columns (keep first)

    # Coerce Finish_Pos
    if "Finish_Pos" in out.columns:
        out["Finish_Pos"] = out["Finish_Pos"].astype(str).str.extract(r'(\d+)')[0].astype(float)

    return out

work = normalize_headers(work, int(race_distance_input))

# ======================= Coerce & derive times =================
def coerce_and_derives(df: pd.DataFrame, D_m: int) -> pd.DataFrame:
    g = df.copy()

    # Parse *_Time and raw total
    for c in list(g.columns):
        if c.endswith("_Time") or c == "RaceTotal_raw":
            g[c] = g[c].apply(parse_time_to_seconds)

    # If Finish_Time looks like a total (>40s), reclassify as total
    if "Finish_Time" in g.columns:
        mask_totalish = g["Finish_Time"] > 40.0
        if mask_totalish.any():
            g.loc[mask_totalish, "RaceTotal_s"] = g.loc[mask_totalish, "Finish_Time"]
            g.loc[mask_totalish, "Finish_Time"] = np.nan

    # Convert RaceTotal_raw → RaceTotal_s
    if "RaceTotal_raw" in g.columns:
        g["RaceTotal_s"] = g["RaceTotal_raw"].apply(parse_time_to_seconds)
        g.drop(columns=["RaceTotal_raw"], inplace=True)

    # If Finish_Time missing, derive from total minus sum(other splits)
    need_finish = ("Finish_Time" not in g.columns) or g["Finish_Time"].isna().all()
    if need_finish and "RaceTotal_s" in g.columns:
        D = int(D_m)
        rem = D % 200
        first_end = D - rem if rem > 0 else D - 200
        split_ends = []
        if first_end >= 200:
            split_ends.append(first_end)
        for m in range(first_end - 200, 199, -200):
            split_ends.append(m)
        split_cols = [f"{m}_Time" for m in split_ends if f"{m}_Time" in g.columns]
        if split_cols:
            sums = g[split_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            g["Finish_Time"] = g["RaceTotal_s"] - sums
            # sanity clamp
            g.loc[(g["Finish_Time"] < 6.0) | (g["Finish_Time"] > 30.0), "Finish_Time"] = np.nan

    return g

work = coerce_and_derives(work, int(race_distance_input))

st.markdown("### Raw Table")
st.dataframe(work.head(12), use_container_width=True)
_dbg(DEBUG, "Columns (normalized, deduped)", list(work.columns))

# ======================= 200 m mechanics (+stub) =========================
def collect_numeric_markers(df):
    """Return all integer prefixes for columns named like '<int>_Time' (excluding Finish_Time)."""
    marks=[]
    for c in df.columns:
        if c.endswith("_Time") and c!="Finish_Time":
            try:
                m=int(c.split("_")[0]); marks.append(m)
            except: pass
    return sorted(set(marks), reverse=True)

def sum_times(row, cols):
    vals=[as_num(row.get(c)).item() if hasattr(as_num(row.get(c)), "item") else as_num(row.get(c)) for c in cols]
    vals=[v for v in vals if pd.notna(v)]
    return np.sum(vals) if len(vals) else np.nan

def stage_speed(row, cols, meters_each):
    """meters_each can be a float or a list aligned to cols (to support stub length)."""
    if not cols: return np.nan
    ts = sum_times(row, cols)
    if pd.isna(ts) or ts <= 0: return np.nan
    if isinstance(meters_each, (list, tuple, np.ndarray)):
        dist = 0.0
        for c, m in zip(cols, meters_each):
            if pd.notna(row.get(c)): dist += float(m)
    else:
        # count only available splits
        dist = float(meters_each) * len([c for c in cols if pd.notna(row.get(c))])
    return np.nan if dist <= 0 else dist / ts

def grind_speed(row):
    tfin=as_num(row.get("Finish_Time"))
    return (200.0/float(tfin)) if pd.notna(tfin) and float(tfin)>0 else np.nan

def build_segment_plan(D: int):
    """Return ordered segments: list of (end_marker, length_m), excluding Finish (200→0)."""
    rem = D % 200
    segs = []
    if rem > 0:
        first_end = D - rem
        if first_end >= 200:
            segs.append((first_end, float(rem)))  # D → (D-rem)
    else:
        first_end = D - 200
        if first_end >= 200:
            segs.append((first_end, 200.0))      # D → (D-200)

    # then regular 200s down to 200
    m = first_end - 200
    while m >= 200:
        segs.append((m, 200.0))
        m -= 200
    return segs  # e.g. [(1000,160.0),(800,200.0),(600,200.0),...,(200,200.0)]

# ======================= Distance + Context PI weights =====================
def pi_weights_distance_and_context(distance_m: float,
                                    acc_median: float | None,
                                    grd_median: float | None) -> dict:
    dm=float(distance_m or 1200)
    if dm<=1000:
        base={"F200_idx":0.12,"tsSPI":0.35,"Accel":0.36,"Grind":0.17}
    elif dm<1100:
        base=_interpolate_weights(dm,1000,{"F200_idx":0.12,"tsSPI":0.35,"Accel":0.36,"Grind":0.17},
                                     1100,{"F200_idx":0.10,"tsSPI":0.36,"Accel":0.34,"Grind":0.20})
    elif dm<1200:
        base=_interpolate_weights(dm,1100,{"F200_idx":0.10,"tsSPI":0.36,"Accel":0.34,"Grind":0.20},
                                     1200,{"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25})
    elif dm==1200:
        base={"F200_idx":0.08,"tsSPI":0.37,"Accel":0.30,"Grind":0.25}
    else:
        shift_units=max(0.0,(dm-1200.0)/100.0)*0.01
        grind=min(0.25+shift_units,0.40); F200,ACC=0.08,0.30
        ts=max(0.0,1.0-F200-ACC-grind)
        base={"F200_idx":F200,"tsSPI":ts,"Accel":ACC,"Grind":grind}

    acc_med = float(acc_median) if acc_median is not None else None
    grd_med = float(grd_median) if grd_median is not None else None
    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        bias=acc_med - grd_med
        scale=math.tanh(abs(bias)/6.0); max_shift=0.02*scale
        F200=base["F200_idx"]; ts=base["tsSPI"]; ACC=base["Accel"]; GR=base["Grind"]
        if bias>0:   # favour Grind slightly
            delta=min(max_shift, ACC-0.26); ACC-=delta; GR+=delta
        elif bias<0: # favour Accel slightly
            delta=min(max_shift, GR-0.18); GR-=delta; ACC+=delta
        GR=min(GR,0.40); ts=max(0.0,1.0-F200-ACC-GR)
        base={"F200_idx":F200,"tsSPI":ts,"Accel":ACC,"Grind":GR}

    s=sum(base.values())
    if abs(s-1.0)>1e-6: base={k:v/s for k,v in base.items()}
    return base

# ======================= Core metric build (stub-aware, UNCHANGED MATH) ===
def build_metrics_stub(df_in: pd.DataFrame, D_actual_m: float):
    w = df_in.copy()
    D = int(D_actual_m)

    # Finish pos numeric
    if "Finish_Pos" in w.columns: w["Finish_Pos"] = as_num(w["Finish_Pos"])

    # Segment plan (end markers + lengths), excluding Finish
    seg_plan = build_segment_plan(D)  # [(end_marker, length_m), ...]
    cols_present = set(w.columns)

    # Per-segment speeds
    for end_m, L in seg_plan:
        col = f"{end_m}_Time"
        if col in cols_present:
            w[f"spd_{end_m}"] = float(L) / as_num(w.get(col))
        else:
            w[f"spd_{end_m}"] = np.nan

    # Finish speed (always 200 m)
    w["spd_Finish"] = 200.0 / as_num(w.get("Finish_Time")) if "Finish_Time" in w.columns else np.nan

    # RaceTime_s: prefer explicit total if provided
    if "RaceTotal_s" in w.columns and w["RaceTotal_s"].notna().any():
        w["RaceTime_s"] = pd.to_numeric(w["RaceTotal_s"], errors="coerce")
    else:
        cols = [f"{end}_Time" for (end, _) in seg_plan if f"{end}_Time" in cols_present]
        if "Finish_Time" in cols_present: cols += ["Finish_Time"]
        w["RaceTime_s"] = w[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # ---------- Stages ----------
    # F200 = the FIRST split (stub length if any, else first 200 m)
    if len(seg_plan) > 0:
        first_end, first_len = seg_plan[0]
        f200_cols = [f"{first_end}_Time"] if f"{first_end}_Time" in cols_present else []
        f200_len  = float(first_len)
    else:
        f200_cols = []; f200_len = np.nan

    # tsSPI = from (D-400) down to 600 inclusive, stepping 200 and clipping to available ends
    tssp_ends = [m for m in range(D-400, 599, -200) if m >= 600]
    tssp_cols = [f"{m}_Time" for m in tssp_ends if f"{m}_Time" in cols_present]
    tssp_lens = [200.0]*len(tssp_cols)  # always 200 m blocks for mid-race

    # Accel = 400 + 200 (before finish)
    accel_cols = [c for c in ["400_Time","200_Time"] if c in cols_present]
    accel_lens = [200.0 for _ in accel_cols]

    # Grind = Finish only (200 m) — handled by grind_speed

    # ---------- Convert to indices vs field (100 = par) ----------
    def speed_to_index(spd_series):
        med = spd_series.median(skipna=True)
        idx_raw = 100.0 * (spd_series / med)
        # stabilizers
        def shrink_center(idx_series):
            x = idx_series.dropna().values; N_eff = len(x)
            if N_eff == 0: return 100.0, 0
            med_race = float(np.median(x)); alpha = N_eff / (N_eff + 6.0)
            return alpha*med_race + (1-alpha)*100.0, N_eff
        def dispersion_equalizer(delta_series, N_eff, N_ref=10, beta=0.20, cap=1.20):
            gamma = 1.0 + beta * max(0, N_ref - N_eff) / N_ref
            return delta_series * min(gamma, cap)
        def variance_floor(idx_series, floor=1.5, cap=1.25):
            deltas = idx_series - 100.0; sigma = mad_std(deltas)
            if not np.isfinite(sigma) or sigma <= 0: return idx_series
            if sigma < floor:
                factor = min(cap, floor / sigma); return 100.0 + deltas * factor
            return idx_series

        center, n_eff = shrink_center(idx_raw)
        idx = 100.0 * (spd_series / (center/100.0 * med))
        idx = 100.0 + dispersion_equalizer(idx - 100.0, n_eff)
        idx = variance_floor(idx)
        return idx

    w["_F200_spd"] = w.apply(
        lambda r: (f200_len / as_num(r.get(f200_cols[0]))) if (f200_cols and pd.notna(as_num(r.get(f200_cols[0]))) and float(as_num(r.get(f200_cols[0])))>0) else np.nan,
        axis=1
    )
    w["_MID_spd"]  = w.apply(lambda r: stage_speed(r, tssp_cols, tssp_lens), axis=1)
    w["_ACC_spd"]  = w.apply(lambda r: stage_speed(r, accel_cols, accel_lens), axis=1)
    w["_GR_spd"]   = w.apply(grind_speed, axis=1)

    w["F200_idx"] = speed_to_index(pd.to_numeric(w["_F200_spd"], errors="coerce"))
    w["tsSPI"]    = speed_to_index(pd.to_numeric(w["_MID_spd"],  errors="coerce"))
    w["Accel"]    = speed_to_index(pd.to_numeric(w["_ACC_spd"],  errors="coerce"))
    w["Grind"]    = speed_to_index(pd.to_numeric(w["_GR_spd"],   errors="coerce"))

    # ---------- PI v3.1 ----------
    acc_med = w["Accel"].median(skipna=True)
    grd_med = w["Grind"].median(skipna=True)
    PI_W = pi_weights_distance_and_context(float(D), acc_med, grd_med)

    def pi_pts_row(row):
        parts, weights = [], []
        for k, wgt in PI_W.items():
            v = row.get(k, np.nan)
            if pd.notna(v): parts.append(wgt*(v-100.0)); weights.append(wgt)
        return np.nan if not weights else sum(parts)/sum(weights)

    w["PI_pts"] = w.apply(pi_pts_row, axis=1)
    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered = pts - med; sigma = mad_std(centered)
    if not np.isfinite(sigma) or sigma < 0.75: sigma = 0.75
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)

    # ---------- GCI ----------
    acc_med_g = w["Accel"].median(skipna=True)
    grd_med_g = w["Grind"].median(skipna=True)
    Wg = pi_weights_distance_and_context(float(D), acc_med_g, grd_med_g)
    wT = 0.25
    wPACE = Wg["Accel"] + Wg["Grind"]
    wSS = Wg["tsSPI"]
    wEFF = max(0.0, 1.0 - (wT + wPACE + wSS))

    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try: winner_time = float(w["RaceTime_s"].min())
        except Exception: winner_time = None

    def map_pct(x, lo=98.0, hi=104.0):
        if pd.isna(x): return 0.0
        return clamp((float(x)-lo)/(hi-lo), 0.0, 1.0)

    gci=[]
    for _, r in w.iterrows():
        T=0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d=float(r["RaceTime_s"]) - winner_time
            T = 1.0 if d<=0.30 else (0.7 if d<=0.60 else (0.4 if d<=1.00 else 0.2))
        LQ = 0.6 * map_pct(r.get("Accel")) + 0.4 * map_pct(r.get("Grind"))
        SS = map_pct(r.get("tsSPI"))
        acc, grd = r.get("Accel"), r.get("Grind")
        EFF = 0.0 if (pd.isna(acc) or pd.isna(grd)) else clamp(1.0 - (abs(acc-100.0)+abs(grd-100.0))/2.0/8.0, 0.0, 1.0)
        gci.append(round(10.0 * (wT*T + wPACE*LQ + wSS*SS + wEFF*EFF), 3))
    w["GCI"] = gci

    # tidy
    for c in ["F200_idx","tsSPI","Accel","Grind","PI","GCI","RaceTime_s"]:
        if c in w.columns: w[c] = w[c].round(3)

    return w, seg_plan

# ---- compute metrics
try:
    metrics, seg_plan = build_metrics_stub(work, float(race_distance_input))
except Exception as e:
    st.error("Metric computation failed."); st.exception(e); st.stop()

# ======================= Data Integrity (expected vs present) ==============
D = int(race_distance_input)

def expected_cols_for_200m(D_m: int, seg_plan_local):
    """Build expected *_Time columns (excluding Finish) from seg plan + Finish_Time."""
    cols = [f"{end}_Time" for (end, _) in seg_plan_local]
    cols.append("Finish_Time")
    return cols

exp_cols = expected_cols_for_200m(D, seg_plan)
missing_cols = [c for c in exp_cols if c not in work.columns]
invalid_counts = {}
for c in exp_cols:
    if c in work.columns:
        s = pd.to_numeric(work[c], errors="coerce")
        invalid_counts[c] = int(((s <= 0) | s.isna()).sum())

def integrity_line():
    msgs = []
    if missing_cols:
        msgs.append("Missing: " + ", ".join(missing_cols))
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
    if bads:
        msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    return " • ".join(msgs)

# ======================= Minimal Header (distance only) ====================
st.markdown(f"## Race Distance: **{D}m**")
if SHOW_WARNINGS and (missing_cols or any(v>0 for v in invalid_counts.values())):
    st.markdown(f"*(⚠ {integrity_line()})*")

# ======================= Metrics table =====================
st.markdown("## Sectional Metrics — stub-aware (PI v3.1 & GCI)")
show_cols = ["Horse","Finish_Pos","RaceTime_s","F200_idx","tsSPI","Accel","Grind","PI","GCI"]
for c in show_cols:
    if c not in metrics.columns: metrics[c]=np.nan

# stable sort: NaN Finish_Pos last
display_df = metrics[show_cols].copy()
_finish_sort = display_df["Finish_Pos"].fillna(1e9)
display_df = display_df.assign(_FinishSort=_finish_sort)
display_df = display_df.sort_values(["PI","_FinishSort"], ascending=[False, True]).drop(columns=["_FinishSort"])
st.dataframe(display_df, use_container_width=True)

# ===================== Sectional Shape Map — Accel vs Grind =====================
def _repel_labels_builtin(ax, x, y, labels, *,
                          init_shift=0.18, k_attract=0.006, k_repel=0.012,
                          max_iter=250):
    trans=ax.transData; renderer=ax.figure.canvas.get_renderer()
    xy=np.column_stack([x,y]).astype(float); offs=np.zeros_like(xy)
    for i,(xi,yi) in enumerate(xy):
        offs[i]=[init_shift if xi>=0 else -init_shift, init_shift if yi>=0 else -init_shift]
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
        for t,(xi,yi) in zip(texts,xy):
            tx,ty=t.get_position(); pt=trans.transform((tx,ty)); pp=trans.transform((xi,yi))
            d=((pt[0]-pp[0])**2+(pt[1]-pp[1])**2)**0.5; tgt=25.0
            if abs(d-tgt)>1.0:
                v=(pt-pp)/(d+1e-9); pt2=pt+v*(0.6*(tgt-d)); t.set_position(inv.transform(pt2)); moved=True
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

st.markdown("## Sectional Shape Map — Accel (400+200) vs Grind (Finish)")
shape_map_png = None
need_cols={"Horse","Accel","Grind","tsSPI","PI"}
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
        dfm["AccelΔ"]=dfm["Accel"]-100.0; dfm["GrindΔ"]=dfm["Grind"]-100.0; dfm["tsSPIΔ"]=dfm["tsSPI"]-100.0
        names=dfm["Horse"].astype(str).to_list()
        xv=dfm["AccelΔ"].to_numpy(); yv=dfm["GrindΔ"].to_numpy(); cv=dfm["tsSPIΔ"].to_numpy(); piv=dfm["PI"].fillna(0).to_numpy()

        span = np.nanmax([np.nanmax(np.abs(xv)), np.nanmax(np.abs(yv))]) if np.isfinite(xv).any() and np.isfinite(yv).any() else 1.0
        if not np.isfinite(span) or span <= 0: span = 1.0
        lim = max(4.5, float(np.ceil(span/1.5)*1.5))

        DOT_MIN, DOT_MAX = 40.0, 140.0
        pmin, pmax = float(np.nanmin(piv)), float(np.nanmax(piv))
        sizes = np.full_like(xv, DOT_MIN) if (not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax-pmin)<1e-9) \
                else DOT_MIN + (piv - pmin) / (pmax - pmin) * (DOT_MAX - DOT_MIN)

        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        TINT = 0.06
        ax.add_patch(Rectangle((0,0),  lim,  lim, facecolor="#4daf4a", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((-lim,0), lim,  lim, facecolor="#377eb8", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((0,-lim), lim, lim, facecolor="#ff7f00", alpha=TINT, edgecolor="none"))
        ax.add_patch(Rectangle((-lim,-lim),lim, lim, facecolor="#984ea3", alpha=TINT, edgecolor="none"))
        ax.axvline(0, color="gray", lw=1.3, ls=(0,(3,3)))
        ax.axhline(0, color="gray", lw=1.3, ls=(0,(3,3)))

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
        ax.set_title("Quadrants: +X = Accel (400+200); +Y = Grind (Finish). Colour = tsSPI deviation")

        s_ex = [DOT_MIN, 0.5*(DOT_MIN+DOT_MAX), DOT_MAX]
        h_ex = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                       markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
        ax.legend(h_ex, ["PI: low","PI: mid","PI: high"], loc="upper left", frameon=False, fontsize=8)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("tsSPI − 100")
        ax.grid(True, linestyle=":", alpha=0.25)
        st.pyplot(fig)

        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        shape_map_png = buf.getvalue()
        st.download_button("Download shape map (PNG)", shape_map_png, file_name="shape_map.png", mime="image/png")
        st.caption("Size = PI. X: Accel (400+200). Y: Finish (200→0). Colour = tsSPIΔ.")

# ======================= Pace Curve (stub-aware) ==========================
st.markdown("## Pace Curve — field average (black) + Top 8 finishers [stub-aware]")
pace_png = None

segs = []  # list of (label, length_m, colname)
for i, (end_m, L) in enumerate(seg_plan):
    col = f"{end_m}_Time"
    if col in work.columns:
        left = D if i == 0 else int(end_m + L)
        segs.append((f"{left}→{end_m}", float(L), col))
# Finish
if "Finish_Time" in work.columns:
    segs.append(("200→0 (Finish)", 200.0, "Finish_Time"))

if len(segs) == 0:
    st.info("Not enough *_Time columns to draw the pace curve.")
else:
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
    ax2.set_title("Pace over segments (left = early; handles stub if distance % 200 ≠ 0)")
    ax2.grid(True, linestyle="--", alpha=0.30)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
    st.pyplot(fig2)

    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
    pace_png = buf2.getvalue()
    st.download_button("Download pace curve (PNG)", pace_png, file_name="pace_curve.png", mime="image/png")
    st.caption(f"Top-8 plotted: {top8_rule}. Finish segment included explicitly.")

# ======================= Top-8 PI — stacked contributions =========
st.markdown("## Top-8 PI — stacked contributions")
acc_med_for_bars = metrics["Accel"].median(skipna=True)
grd_med_for_bars = metrics["Grind"].median(skipna=True)
PI_W_BARS = pi_weights_distance_and_context(float(D), acc_med_for_bars, grd_med_for_bars)
bars_png = None

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
    if not np.isfinite(total_pi) or total_pi<=0 or not np.isfinite(s) or s<=0:
        return {"F200_idx":0.0, "tsSPI":0.0, "Accel":0.0, "Grind":0.0}
    scale = float(total_pi)/float(s)
    return {k: v*scale for k,v in raw.items()}

top8_pi = metrics.sort_values(["PI","Finish_Pos"], ascending=[False, True]).head(8).copy()
if not top8_pi.empty:
    horses, totals, is_winner = [], [], []
    stacks = {"F200_idx": [], "tsSPI": [], "Accel": [], "Grind": []}
    for _, r in top8_pi.iterrows():
        total_pi = float(r.get("PI", 0.0))
        parts = parts_scaled_to_total(r, total_pi, PI_W_BARS, zero_floor=True)
        for k in stacks: stacks[k].append(parts[k])
        totals.append(total_pi)
        horses.append(str(r.get("Horse", "")))
        fp = pd.to_numeric(r.get("Finish_Pos", np.nan), errors="coerce")
        is_winner.append(False if pd.isna(fp) else int(fp)==1)

    fig3, ax3 = plt.subplots(figsize=(max(7.5, 0.95*len(horses)), 4.8))
    x = np.arange(len(horses))
    palette = {"F200_idx":"#6baed6", "tsSPI":"#9e9ac8", "Accel":"#74c476", "Grind":"#fd8d3c"}

    bottoms = np.zeros(len(horses))
    for key, label in [("F200_idx","F200"), ("tsSPI","tsSPI"), ("Accel","Accel"), ("Grind","Grind")]:
        vals = np.array(stacks[key], dtype=float)
        ax3.bar(x, vals, bottom=bottoms, label=label, color=palette[key], edgecolor="black", linewidth=0.4)
        bottoms += vals

    ymax = max(0.1, max(totals)*1.20)
    for i, tot in enumerate(totals):
        if is_winner[i]:
            ax3.add_patch(plt.Rectangle((i-0.5, 0), 1.0, max(tot, bottoms[i]), fill=False, lw=2.0, ec="#d4af37"))
            horses[i] = f"★ {horses[i]}"
        ax3.text(i, tot + ymax*0.03, f"{tot:.2f}", ha="center", va="bottom", fontsize=9)

    ax3.set_xticks(x); ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylim(0, ymax)
    ax3.set_ylabel("PI (stacked contributions)")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    st.pyplot(fig3)

    buf3 = io.BytesIO(); fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight")
    bars_png = buf3.getvalue()
    st.download_button("Download PI stacks (PNG)", bars_png, file_name="pi_stacks.png", mime="image/png")
    st.caption("Slices are rescaled to sum to each horse’s PI. ★ = race winner.")
else:
    st.info("No PI values available to plot the stacked contributions.")

# ======================= Hidden Horses (v2) =================
st.markdown("## Hidden Horses (v2)")
hh = metrics.copy()

# 1) SOS
need_cols = {"tsSPI","Accel","Grind"}
if need_cols.issubset(hh.columns) and len(hh)>0:
    ts_w = winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
    ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
    gr_w = winsorize(pd.to_numeric(hh["Grind"], errors="coerce"))

    def rz(s: pd.Series)->pd.Series:
        mu=np.nanmedian(s); sd=mad_std(s)
        if not np.isfinite(sd) or sd==0: return pd.Series(np.zeros(len(s)), index=s.index)
        return (s-mu)/sd

    z_ts=rz(ts_w).clip(-2.5,3.5)
    z_ac=rz(ac_w).clip(-2.5,3.5)
    z_gr=rz(gr_w).clip(-2.5,3.5)

    hh["SOS_raw"]=0.45*z_ts + 0.35*z_ac + 0.20*z_gr
    q5,q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom = (q95-q5) if (pd.notna(q95) and pd.notna(q5) and (q95>q5)) else 1.0
    hh["SOS"] = (2.0*(hh["SOS_raw"]-q5)/denom).clip(lower=0.0, upper=2.0)
else:
    hh["SOS"] = 0.0

# 2) ASI²
acc_med = pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
grd_med = pd.to_numeric(hh.get("Grind"), errors="coerce").median(skipna=True)
bias = (acc_med-100.0) - (grd_med-100.0)
B = min(1.0, abs(bias)/4.0)
S = pd.to_numeric(hh.get("Accel"), errors="coerce") - pd.to_numeric(hh.get("Grind"), errors="coerce")
if bias >= 0:
    hh["ASI2"] = (B * (-S).clip(lower=0.0) / 5.0).fillna(0.0)
else:
    hh["ASI2"] = (B * (S).clip(lower=0.0) / 5.0).fillna(0.0)

# 3) TFS (late variability vs mid pace) — last three pre-finish blocks: 600, 400, 200
def tfs_row(r):
    last3_cols = [c for c in ["600_Time","400_Time","200_Time"] if c in r.index]
    spds=[]
    for c in last3_cols:
        t=pd.to_numeric(r.get(c), errors="coerce")
        spds.append(200.0/t if pd.notna(t) and t>0 else np.nan)
    spds=[s for s in spds if pd.notna(s)]
    if len(spds)<2: return np.nan
    sigma=float(np.std(spds, ddof=0))
    mid=float(r.get("_MID_spd", np.nan))
    if not np.isfinite(mid) or mid<=0: return np.nan
    return 100.0*(sigma/mid)
hh["TFS"] = hh.apply(tfs_row, axis=1)

# 4) Distance-aware TFS gate
D_rounded = int(np.ceil(float(D) / 200.0) * 200)
gate = 4.0 if D_rounded <= 1200 else (3.5 if D_rounded < 1800 else 3.0)
def tfs_plus(x): 
    if pd.isna(x) or x < gate: return 0.0
    return min(0.6, (x - gate) / 3.0)
hh["TFS_plus"] = hh["TFS"].apply(tfs_plus)

# 5) UEI
def uei_row(r):
    ts = pd.to_numeric(r.get("tsSPI"), errors="coerce")
    ac = pd.to_numeric(r.get("Accel"), errors="coerce")
    gr = pd.to_numeric(r.get("Grind"), errors="coerce")
    if pd.isna(ts) or pd.isna(ac) or pd.isna(gr): return 0.0
    val=0.0
    if ts >= 102 and ac <= 98 and gr <= 98:
        gap = min((ts - 102) / 3.0, 1.0); val = max(val, 0.3 + 0.3*gap)
    if ts >= 102 and gr >= 102 and ac <= 100:
        gap = min(((ts - 102) + (gr - 102)) / 6.0, 1.0); val = max(val, 0.3 + 0.3*gap)
    return round(val, 3)
hh["UEI"] = hh.apply(uei_row, axis=1)

# 6) HiddenScore v2 (0..3)
hidden = (0.55*pd.to_numeric(hh["SOS"], errors="coerce").fillna(0.0) +
          0.30*pd.to_numeric(hh["ASI2"], errors="coerce").fillna(0.0) +
          0.10*pd.to_numeric(hh["TFS_plus"], errors="coerce").fillna(0.0) +
          0.05*pd.to_numeric(hh["UEI"], errors="coerce").fillna(0.0))
if int(hh.shape[0]) <= 6: hidden *= 0.90

h_med = float(np.nanmedian(hidden))
h_mad = float(np.nanmedian(np.abs(hidden - h_med)))
h_sigma = max(1e-6, 1.4826*h_mad)
hh["HiddenScore"] = (1.2 + (hidden - h_med) / (2.5*h_sigma)).clip(lower=0.0, upper=3.0)

# 7) Tiering & Notes
def hh_tier(s):
    if pd.isna(s): return ""
    if s >= 1.8:   return "Top Hidden"   # text for PDF safety (emoji can break in ReportLab)
    if s >= 1.2:   return "Notable Hidden"
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

cols_hh = ["Horse","Finish_Pos","PI","GCI","tsSPI","Accel","Grind","SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
for c in cols_hh:
    if c not in hh.columns: hh[c]=np.nan
hh_view = hh.sort_values(["Tier","HiddenScore","PI"], ascending=[True,False,False])[cols_hh]
st.dataframe(hh_view, use_container_width=True)

st.caption(
    "Conventions — grid uses 200 m blocks with an optional initial stub if distance % 200 ≠ 0. "
    "`X_Time` = time from (X+Δ)→X, where Δ = stub length for the first split, else 200. "
    "`Finish_Time` is 200→0. If only a race total is uploaded, the app derives the last-200 as "
    "`Finish_Time = total − sum(other splits)`."
)

# ======================= PDF Report Builder ===============================
st.markdown("---")
st.markdown("### 📥 Download Comprehensive Report (PDF)")

def make_pdf_report(distance_m: int, metrics_df: pd.DataFrame,
                    shape_png: bytes|None, pace_png: bytes|None, bars_png_: bytes|None,
                    hh_flagged_df: pd.DataFrame, integrity_text: str|None):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import cm
    except Exception as e:
        st.error("`reportlab` is required to create the PDF. Install with: `pip install reportlab>=4.2.0`")
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()
    H = styles["Heading1"]; H.fontSize = 18; H.leading = 22; H.spaceAfter = 6
    H3 = styles["Heading3"]
    P = styles["BodyText"]; P.fontSize = 9; P.leading = 12

    # Header: Distance (bold & large)
    story.append(Paragraph(f"Race Distance: <b>{int(distance_m)}m</b>", H))
    if integrity_text:
        story.append(Paragraph(f"<font color='#b36b00'>⚠ {integrity_text}</font>", P))
    story.append(Spacer(0, 6))

    # 1) Sectional Metrics table
    story.append(Paragraph("Sectional Metrics (PI v3.1 & GCI)", H3))
    table_df = metrics_df.copy()
    for col in ["RaceTime_s","F200_idx","tsSPI","Accel","Grind","PI","GCI"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    data = [list(table_df.columns)] + table_df.fillna("").astype(str).values.tolist()
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

    # 2) Shape Map image
    if shape_png is not None:
        story.append(Paragraph("Sectional Shape Map — Accel vs Grind (colour = tsSPIΔ)", H3))
        story.append(Image(io.BytesIO(shape_png), width=24*cm, height=18*cm, kind="proportional"))
        story.append(Spacer(0, 8))

    # 3) Pace Curve image
    if pace_png is not None:
        story.append(Paragraph("Pace Curve — field average + Top-8", H3))
        story.append(Image(io.BytesIO(pace_png), width=24*cm, height=15*cm, kind="proportional"))
        story.append(Spacer(0, 8))

    # 4) Top-8 PI stacks
    if bars_png_ is not None:
        story.append(Paragraph("Top-8 PI — stacked contributions", H3))
        story.append(Image(io.BytesIO(bars_png_), width=24*cm, height=12*cm, kind="proportional"))
        story.append(Spacer(0, 8))

    # 5) Hidden Horses v2 table (only flagged)
    flagged = hh_flagged_df.copy()
    if not flagged.empty:
        story.append(Paragraph("Hidden Horses v2 (flagged)", H3))
        fh = flagged.copy()
        for col in ["PI","GCI","tsSPI","Accel","Grind","SOS","ASI2","TFS","UEI","HiddenScore"]:
            if col in fh.columns:
                fh[col] = pd.to_numeric(fh[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        data_hh = [list(fh.columns)] + fh.fillna("").astype(str).values.tolist()
        t2 = Table(data_hh, repeatRows=1)
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('ALIGN', (2,1), (-1,-1), 'RIGHT'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.whitesmoke),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
        ]))
        story.append(t2)
        story.append(Spacer(0, 10))

    # 6) Footnotes / conventions
    story.append(Paragraph("<b>Conventions</b>", H3))
    story.append(Paragraph(
        "Grid uses 200 m blocks with an optional initial stub if distance % 200 ≠ 0. "
        "X_Time = time from (X+Δ)→X, where Δ = stub length for the first split, else 200. "
        "Finish_Time is 200→0. If only a race total is uploaded, the app derives the last-200 as "
        "Finish_Time = total − sum(other splits). Indices are vs-field (100=par) with small-field stabilizers. "
        "PI v3.1 uses distance+context weights; GCI aligns to the same worldview.",
        P
    ))

    doc.build(story)
    buf.seek(0)
    return buf

# Build flagged HH view (text tier already safe for PDF fonts)
hh_flagged = hh_view[hh_view["Tier"] != ""].copy()

# Compose PDF metrics table exactly like UI table
pdf_table_df = display_df.copy()

# Integrity line (only if warnings enabled & something wrong)
integrity_text = integrity_line() if (SHOW_WARNINGS and (missing_cols or any(v>0 for v in invalid_counts.values()))) else None

pdf_buf = make_pdf_report(
    distance_m=D,
    metrics_df=pdf_table_df,
    shape_png=shape_map_png,
    pace_png=pace_png,
    bars_png_=bars_png,
    hh_flagged_df=hh_flagged,
    integrity_text=integrity_text
)
if pdf_buf is not None:
    st.download_button(
        "📥 Download PDF report",
        data=pdf_buf.getvalue(),
        file_name=f"RaceEdge_Report_{D}m.pdf",
        mime="application/pdf"
    )
