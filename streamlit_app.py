import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import io, math, re

# ============== Page config ==============
st.set_page_config(page_title="Race Edge — 200m splits (+stub) | PI v3.1 + GCI + Hidden Horses v2",
                   layout="wide")

# ============== Small helpers ==============
def as_num(x): return pd.to_numeric(x, errors="coerce")
def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color',
            ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out, i = [], 0
    while len(out) < n: out.append(base[i % len(base)]); i += 1
    return out
def clamp(v, lo, hi): return max(lo, min(hi, float(v)))
def mad_std(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x); mad = np.median(np.abs(x - med))
    return 1.4826 * mad
def winsorize(s, p_lo=0.10, p_hi=0.90):
    lo = s.quantile(p_lo); hi = s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)
def _lerp(a,b,t): return a + (b - a) * float(t)
def _interpolate_weights(dm, a_dm, a_w, b_dm, b_w):
    span = float(b_dm - a_dm)
    t = 0.0 if span <= 0 else (float(dm) - a_dm) / span
    return {"F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
            "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
            "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
            "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t)}
def _dbg(on, label, obj=None):
    if on:
        st.write(f"🔧 {label}")
        if obj is not None: st.write(obj)

# ============== Sidebar ==============
with st.sidebar:
    st.markdown("### Upload (200 m grid + stub)")
    up = st.file_uploader(
        "CSV/XLSX with splits like 1600_Time, 1400_Time, …, 200_Time, optional stub (e.g. 50_Time/160_Time), and Finish.",
        type=["csv","xlsx","xls"]
    )
    # allow ANY distance (10m steps to make it easy to type)
    race_distance_input = st.number_input("Race Distance (m)", min_value=800, max_value=4000,
                                          step=10, value=1600)
    DEBUG = st.toggle("Debug info", value=False)
if not up: st.stop()

# ============== Load file ==============
try:
    work = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    st.success("File loaded.")
except Exception:
    st.error("Failed to read file."); st.stop()

# ============== Time parsing ==============
def parse_time_to_seconds(v):
    """Accepts 12.34 / 12,34 / m:ss.xx / h:mm:ss.xx / strings; returns seconds (float) or NaN."""
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
        if len(parts) == 3:
            h, m, sec = parts; return h*3600 + m*60 + sec
        if len(parts) == 2:
            m, sec = parts; return m*60 + sec
        return np.nan
    try: return float(s)
    except Exception: return np.nan

# ============== Header normalizer ==============
def _canon(s: str) -> str: return re.sub(r'[^a-z0-9]', '', s.lower())

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

        # Horse
        if re.fullmatch(r'(?i)(horse|runner|name)', raw):
            rename[col] = "Horse"; continue

        # Position
        if re.fullmatch(r'(?i)(finish\s*pos(ition)?|pos(ition)?|fin_pos|finishpos)', raw):
            rename[col] = "Finish_Pos"; continue

        # Explicit finish split vs total
        if (cx in FINISH_SPLIT_ALIASES) or re.fullmatch(r'(?i)(finish\s*split|finish200|last200|final200|home200)', raw):
            rename[col] = "Finish_Time"; continue
        if (cx in FINISH_TOTAL_ALIASES) or re.fullmatch(r'(?i)(finish\s*time|race\s*time|total\s*time)', raw):
            rename[col] = "RaceTotal_raw"; continue

        # Any numeric split like "1600mTime", "160_Time", "1450_Time", etc. (≤ D)
        m = re.match(r'(?i).*?(\d{2,4})\s*m?\s*[_\s-]?\s*(time|split)?$', raw)
        if m:
            metres = int(m.group(1))
            if 0 < metres <= int(D_m):  # accept ANY metres up to D (so stub like 50/100/160 works)
                rename[col] = f"{metres}_Time"; continue

        m2 = re.match(r'(?i)^(\d{2,4})m(Time|Split)$', raw)
        if m2:
            metres = int(m2.group(1))
            if 0 < metres <= int(D_m):
                rename[col] = f"{metres}_Time"; continue

    out = df.rename(columns=rename)
    out = out.loc[:, ~out.columns.duplicated()]  # drop dupes, keep first

    if "Finish_Pos" in out.columns:
        out["Finish_Pos"] = out["Finish_Pos"].astype(str).str.extract(r'(\d+)')[0].astype(float)

    return out

work = normalize_headers(work, int(race_distance_input))

# ============== Coerce times & derive Finish if needed ==============
def coerce_and_derives(df: pd.DataFrame, D_m: int) -> pd.DataFrame:
    g = df.copy()

    # parse all *_Time + any raw total
    for c in list(g.columns):
        if c.endswith("_Time") or c == "RaceTotal_raw":
            g[c] = g[c].apply(parse_time_to_seconds)

    # If Finish_Time looks like a total (> 40s), park it as RaceTotal_s
    if "Finish_Time" in g.columns:
        mask_totalish = g["Finish_Time"] > 40.0
        if mask_totalish.any():
            g.loc[mask_totalish, "RaceTotal_s"] = g.loc[mask_totalish, "Finish_Time"]
            g.loc[mask_totalish, "Finish_Time"] = np.nan

    # If we have a raw total, convert to seconds
    if "RaceTotal_raw" in g.columns:
        g["RaceTotal_s"] = g["RaceTotal_raw"].apply(parse_time_to_seconds)
        g.drop(columns=["RaceTotal_raw"], inplace=True)

    # If no Finish_Time, derive as: total − sum(other splits up to D (including stub))
    need_finish = ("Finish_Time" not in g.columns) or g["Finish_Time"].isna().all()
    if need_finish and "RaceTotal_s" in g.columns:
        # collect all numeric split columns within (0..D], EXCEPT Finish_Time
        split_cols = [c for c in g.columns if c.endswith("_Time") and c != "Finish_Time"
                      and 0 < int(c.split("_")[0]) <= int(D_m)]
        if split_cols:
            sums = g[split_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            g["Finish_Time"] = g["RaceTotal_s"] - sums
            # sanity bounds for last-200 (usually ~9–20s)
            g.loc[(g["Finish_Time"] < 6.0) | (g["Finish_Time"] > 30.0), "Finish_Time"] = np.nan

    return g

work = coerce_and_derives(work, int(race_distance_input))

st.markdown("### Raw Table")
st.dataframe(work.head(12), use_container_width=True)
_dbg(DEBUG, "Columns (normalized)", list(work.columns))

# ============== 200 m grid + STUB support ==============
D = float(race_distance_input)
stub = int(D % 200)  # 0 if clean; else e.g. 50, 100, 160
first_marker = int(D - (stub if stub > 0 else 200))  # e.g. 1160→1000, 1450→1400, 1600→1400

def collect_markers(df):
    """All *_Time (numeric) <= D, excluding Finish_Time; sorted desc."""
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                m = int(c.split("_")[0])
                if 0 < m <= int(D): marks.append(m)
            except: pass
    return sorted(set(marks), reverse=True)

def sum_times(row, cols):
    vals = [as_num(row.get(c)).item() if hasattr(as_num(row.get(c)), "item")
            else as_num(row.get(c)) for c in cols]
    vals = [v for v in vals if pd.notna(v)]
    return np.sum(vals) if len(vals) else np.nan

def stage_block_cols(start_m, end_m_inclusive, step, df_cols_set):
    """Return existing *_Time from start_m down to end_m_inclusive by 'step'."""
    if start_m < end_m_inclusive: return []
    want = list(range(int(start_m), int(end_m_inclusive) - 1, -int(step)))
    return [f"{m}_Time" for m in want if f"{m}_Time" in df_cols_set]

def stage_speed(row, cols, meters_per_split):
    if not cols: return np.nan
    tsum = sum_times(row, cols)
    if pd.isna(tsum) or tsum <= 0: return np.nan
    # distance = sum of actual segment lengths represented by those column names
    total_m = 0.0
    for c in cols:
        m = int(c.split("_")[0])
        prev = m + meters_per_split  # meters_per_split is 200 for regular blocks; stub handled separately
        # We'll compute exact length using names: for regular blocks it's 200; for stub it's 'stub'
        # But easier: when we assemble cols, pass meters_per_split=200 and handle stub separately where needed.
    # For our usage below we either pass all-200 blocks, OR a single stub with its true length via meters_per_split.
    return (meters_per_split * len(cols)) / tsum  # used only for pure-200 blocks OR single stub

def speed_to_index(spd_series):
    med = spd_series.median(skipna=True)
    idx_raw = 100.0 * (spd_series / med)
    # stabilizers
    def shrink_center(idx_series):
        x = idx_series.dropna().values; N = len(x)
        if N == 0: return 100.0, 0
        med_race = float(np.median(x)); alpha = N / (N + 6.0)
        return alpha*med_race + (1-alpha)*100.0, N
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

# ============== Build metrics (F200 uses FIRST SPLIT; may be stub) ==============
def build_metrics(df_in: pd.DataFrame):
    w = df_in.copy()
    if "Finish_Pos" in w.columns: w["Finish_Pos"] = as_num(w["Finish_Pos"])

    seg_markers = collect_markers(w)  # includes stub marker if present
    _dbg(DEBUG, "Markers (≤D)", seg_markers)
    cols_set = set(w.columns)

    # per-segment speeds for 200 m blocks
    for m in seg_markers:
        w[f"spd_{m}"] = (200.0 / as_num(w.get(f"{m}_Time"))) if (m % 200 == 0) else np.nan
    w["spd_Finish"] = 200.0 / as_num(w.get("Finish_Time")) if "Finish_Time" in w.columns else np.nan

    # --- Race time ---
    if "RaceTotal_s" in w.columns and w["RaceTotal_s"].notna().any():
        w["RaceTime_s"] = pd.to_numeric(w["RaceTotal_s"], errors="coerce")
    else:
        # sum all split times up to D plus Finish
        split_cols = [f"{m}_Time" for m in seg_markers if f"{m}_Time" in cols_set]
        if "Finish_Time" in cols_set: split_cols += ["Finish_Time"]
        w["RaceTime_s"] = w[split_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # --- Stage definitions (with stub handling) ---
    # F200 = FIRST split (stub if D%200>0; otherwise the D→D−200 block)
    f200_cols = []
    f200_len_m = 200.0
    if f"{first_marker}_Time" in cols_set:
        f200_cols = [f"{first_marker}_Time"]
        f200_len_m = float(200 if (stub == 0) else stub)  # if stub, it's stub metres

    # tsSPI = (D-400) down to 600, step 200, regular 200 m blocks only
    tssp_cols = stage_block_cols(int(D-400), 600, 200, cols_set)

    # Accel = 400 + 200 (regular blocks)
    accel_cols = [c for c in ["400_Time", "200_Time"] if c in cols_set]

    # Grind = Finish only (200→0)
    # (handled via grind_speed)

    # --- Compute composite speeds ---
    def stage_spd_f200(row):
        if not f200_cols: return np.nan
        t = as_num(row.get(f200_cols[0]))
        return (f200_len_m / float(t)) if pd.notna(t) and float(t) > 0 else np.nan

    def stage_speed_generic(row, cols, meters_per_split):
        return stage_speed(row, cols, meters_per_split=meters_per_split)

    def grind_speed(row):
        tfin = as_num(row.get("Finish_Time"))
        return (200.0 / float(tfin)) if pd.notna(tfin) and float(tfin) > 0 else np.nan

    w["_F200_spd"] = w.apply(stage_spd_f200, axis=1)
    w["_MID_spd"]  = w.apply(lambda r: stage_speed_generic(r, tssp_cols, 200.0), axis=1)
    w["_ACC_spd"]  = w.apply(lambda r: stage_speed_generic(r, accel_cols, 200.0), axis=1)
    w["_GR_spd"]   = w.apply(grind_speed, axis=1)

    # --- Indices vs field (100 = par) ---
    w["F200_idx"] = speed_to_index(pd.to_numeric(w["_F200_spd"], errors="coerce"))
    w["tsSPI"]    = speed_to_index(pd.to_numeric(w["_MID_spd"],  errors="coerce"))
    w["Accel"]    = speed_to_index(pd.to_numeric(w["_ACC_spd"],  errors="coerce"))
    w["Grind"]    = speed_to_index(pd.to_numeric(w["_GR_spd"],   errors="coerce"))

    # --- PI v3.1 ---
    acc_med=w["Accel"].median(skipna=True); grd_med=w["Grind"].median(skipna=True)
    PI_W=pi_weights_distance_and_context(float(D), acc_med, grd_med)

    def pi_pts_row(row):
        parts=[]; weights=[]
        for k,wgt in PI_W.items():
            v=row.get(k, np.nan)
            if pd.notna(v): parts.append(wgt*(v-100.0)); weights.append(wgt)
        return np.nan if not weights else sum(parts)/sum(weights)

    w["PI_pts"]=w.apply(pi_pts_row, axis=1)
    pts=pd.to_numeric(w["PI_pts"], errors="coerce")
    med=float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered=pts-med; sigma=mad_std(centered);  sigma=0.75 if (not np.isfinite(sigma) or sigma<0.75) else sigma
    w["PI"]=(5.0+2.2*(centered/sigma)).clip(0.0,10.0).round(2)

    # --- GCI (0–10) ---
    acc_med_g=w["Accel"].median(skipna=True); grd_med_g=w["Grind"].median(skipna=True)
    Wg=pi_weights_distance_and_context(float(D), acc_med_g, grd_med_g)
    wT=0.25; wPACE=Wg["Accel"]+Wg["Grind"]; wSS=Wg["tsSPI"]; wEFF=max(0.0, 1.0-(wT+wPACE+wSS))

    winner_time=None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try: winner_time=float(w["RaceTime_s"].min())
        except Exception: winner_time=None

    def map_pct(x, lo=98.0, hi=104.0):
        if pd.isna(x): return 0.0
        return clamp((float(x)-lo)/(hi-lo), 0.0, 1.0)

    gci=[]
    for _, r in w.iterrows():
        T=0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d=float(r["RaceTime_s"])-winner_time
            T = 1.0 if d<=0.30 else (0.7 if d<=0.60 else (0.4 if d<=1.00 else 0.2))
        LQ=0.6*map_pct(r.get("Accel")) + 0.4*map_pct(r.get("Grind"))
        SS=map_pct(r.get("tsSPI"))
        acc,grd=r.get("Accel"), r.get("Grind")
        EFF=0.0 if (pd.isna(acc) or pd.isna(grd)) else clamp(1.0-(abs(acc-100.0)+abs(grd-100.0))/2.0/8.0,0.0,1.0)
        gci.append(round(10.0*(wT*T + wPACE*LQ + wSS*SS + wEFF*EFF),3))
    w["GCI"]=gci

    for c in ["F200_idx","tsSPI","Accel","Grind","PI","GCI","RaceTime_s"]:
        if c in w.columns: w[c]=w[c].round(3)
    return w, seg_markers

# compute
try:
    metrics, seg_markers = build_metrics_200(work, float(race_distance_input))
except Exception as e:
    st.error("Metric computation failed."); st.exception(e); st.stop()

# ---------------- tables/plots ----------------
st.markdown("## Sectional Metrics (200m) — PI v3.1 & GCI")
show_cols=["Horse","Finish_Pos","RaceTime_s","F200_idx","tsSPI","Accel","Grind","PI","GCI"]
for c in show_cols:
    if c not in metrics.columns: metrics[c]=np.nan
st.dataframe(metrics[show_cols].sort_values(["PI","Finish_Pos"], ascending=[False, True]),
             use_container_width=True)

# (plots unchanged from previous version; they use Finish_Time for Grind and RaceTime_s for totals)
# ---------------- Shape Map ----------------
def _repel_labels_builtin(ax, x, y, labels, *, init_shift=0.18, k_attract=0.006, k_repel=0.012, max_iter=250):
    trans=ax.transData; renderer=ax.figure.canvas.get_renderer()
    xy=np.column_stack([x,y]).astype(float); offs=np.zeros_like(xy)
    for i,(xi,yi) in enumerate(xy):
        offs[i]=[init_shift if xi>=0 else -init_shift, init_shift if yi>=0 else -init_shift]
    texts,lines=[],[]
    for (xi,yi),(dx,dy),lab in zip(xy,offs,labels):
        t=ax.text(xi+dx, yi+dy, lab, fontsize=8.6, va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))
        texts.append(t); ln=Line2D([xi,xi+dx],[yi,yi+dy], lw=0.75, color="black", alpha=0.9)
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
                    tx,ty=t.get_position(); px=trans.transform((tx,ty))+s*np.array([dx,dy])
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
        adjust_text(texts, x=x, y=y, ax=ax, only_move={'points':'y','text':'xy'},
                    force_points=0.6, force_text=0.7,
                    expand_text=(1.05,1.15), expand_points=(1.05,1.15),
                    arrowprops=dict(arrowstyle="->", lw=0.75, color="black", alpha=0.9,
                                    shrinkA=0, shrinkB=3))
    except Exception:
        _repel_labels_builtin(ax, x, y, names)

st.markdown("## Sectional Shape Map — Accel (400+200) vs Grind (Finish split)")
need_cols={"Horse","Accel","Grind","tsSPI","PI"}
if not need_cols.issubset(metrics.columns):
    st.warning("Shape Map: required columns missing: "+", ".join(sorted(need_cols-set(metrics.columns))))
else:
    dfm=metrics.loc[:, ["Horse","Accel","Grind","tsSPI","PI"]].copy()
    for c in ["Accel","Grind","tsSPI","PI"]: dfm[c]=pd.to_numeric(dfm[c], errors="coerce")
    dfm=dfm.dropna(subset=["Accel","Grind","tsSPI"])
    if dfm.empty:
        st.info("Not enough data to draw the shape map.")
    else:
        dfm["AccelΔ"]=dfm["Accel"]-100.0; dfm["GrindΔ"]=dfm["Grind"]-100.0; dfm["tsSPIΔ"]=dfm["tsSPI"]-100.0
        names=dfm["Horse"].astype(str).to_list()
        xv=dfm["AccelΔ"].to_numpy(); yv=dfm["GrindΔ"].to_numpy(); cv=dfm["tsSPIΔ"].to_numpy(); piv=dfm["PI"].fillna(0).to_numpy()
        span=float(np.nanmax([np.nanmax(np.abs(xv)), np.nanmax(np.abs(yv))])) if np.isfinite(xv).any() and np.isfinite(yv).any() else 1.0
        if not np.isfinite(span) or span<=0: span=1.0
        lim=max(4.5, float(np.ceil(span/1.5)*1.5))
        DOT_MIN, DOT_MAX = 40.0, 140.0
        pmin,pmax=float(np.nanmin(piv)), float(np.nanmax(piv))
        sizes=np.full_like(xv, DOT_MIN) if (not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax-pmin)<1e-9) \
              else DOT_MIN + (piv-pmin)/(pmax-pmin)*(DOT_MAX-DOT_MIN)
        fig, ax = plt.subplots(figsize=(7.6,6.4))
        TINT=0.06
        ax.add_patch(Rectangle((0,0),lim,lim,facecolor="#4daf4a",alpha=TINT,edgecolor="none"))
        ax.add_patch(Rectangle((-lim,0),lim,lim,facecolor="#377eb8",alpha=TINT,edgecolor="none"))
        ax.add_patch(Rectangle((0,-lim),lim,lim,facecolor="#ff7f00",alpha=TINT,edgecolor="none"))
        ax.add_patch(Rectangle((-lim,-lim),lim,lim,facecolor="#984ea3",alpha=TINT,edgecolor="none"))
        ax.axvline(0,color="gray",lw=1.3,ls=(0,(3,3))); ax.axhline(0,color="gray",lw=1.3,ls=(0,(3,3)))
        vmin=float(np.nanmin(cv)) if np.isfinite(cv).any() else -1.0
        vmax=float(np.nanmax(cv)) if np.isfinite(cv).any() else  1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax: vmin,vmax=-1.0,1.0
        norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        sc=ax.scatter(xv,yv,s=sizes,c=cv,cmap="coolwarm",norm=norm,edgecolor="black",linewidth=0.6,alpha=0.95)
        label_points_neatly(ax, xv, yv, names)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.set_xlabel("Acceleration vs field (points)  →"); ax.set_ylabel("Grind (Finish split) vs field (points)  ↑")
        ax.set_title("X = Accel (400+200); Y = Grind (200→0). Colour = tsSPI deviation")
        s_ex=[DOT_MIN,0.5*(DOT_MIN+DOT_MAX),DOT_MAX]
        h_ex=[Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                     markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
        ax.legend(h_ex, ["PI: low","PI: mid","PI: high"], loc="upper left", frameon=False, fontsize=8)
        cbar=fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("tsSPI − 100")
        ax.grid(True, linestyle=":", alpha=0.25); st.pyplot(fig)
        buf=io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Download shape map (PNG)", buf.getvalue(), file_name="shape_map.png", mime="image/png")

# -------- Top-8 PI stacks --------
st.markdown("## Top-8 PI — stacked contributions (200m definitions)")
acc_med_for_bars=metrics["Accel"].median(skipna=True)
grd_med_for_bars=metrics["Grind"].median(skipna=True)
PI_W_BARS=pi_weights_distance_and_context(float(race_distance_input), acc_med_for_bars, grd_med_for_bars)
def parts_scaled_to_total(row, total_pi, weights, zero_floor=True):
    raw={"F200_idx":weights["F200_idx"]*(float(row.get("F200_idx",100.0))-100.0),
         "tsSPI":   weights["tsSPI"]   *(float(row.get("tsSPI",   100.0))-100.0),
         "Accel":   weights["Accel"]   *(float(row.get("Accel",   100.0))-100.0),
         "Grind":   weights["Grind"]   *(float(row.get("Grind",   100.0))-100.0)}
    if zero_floor: raw={k:max(0.0,v) for k,v in raw.items()}
    s=sum(raw.values())
    if not np.isfinite(total_pi) or total_pi<=0 or not np.isfinite(s) or s<=0: return {k:0.0 for k in raw}
    scale=float(total_pi)/float(s); return {k:v*scale for k,v in raw.items()}
top8_pi=metrics.sort_values(["PI","Finish_Pos"], ascending=[False, True]).head(8).copy()
if not top8_pi.empty:
    horses, totals, is_winner=[], [], []
    stacks={"F200_idx": [], "tsSPI": [], "Accel": [], "Grind": []}
    for _, r in top8_pi.iterrows():
        total_pi=float(r.get("PI",0.0)); parts=parts_scaled_to_total(r, total_pi, PI_W_BARS, zero_floor=True)
        for k in stacks: stacks[k].append(parts[k])
        totals.append(total_pi); horses.append(str(r.get("Horse","")))
        fp=pd.to_numeric(r.get("Finish_Pos", np.nan), errors="coerce"); is_winner.append(False if pd.isna(fp) else int(fp)==1)
    fig3, ax3 = plt.subplots(figsize=(max(7.5, 0.95*len(horses)), 4.8))
    x=np.arange(len(horses)); palette={"F200_idx":"#6baed6","tsSPI":"#9e9ac8","Accel":"#74c476","Grind":"#fd8d3c"}
    bottoms=np.zeros(len(horses))
    for key,label in [("F200_idx","F200"),("tsSPI","tsSPI"),("Accel","Accel"),("Grind","Grind")]:
        vals=np.array(stacks[key], dtype=float)
        ax3.bar(x, vals, bottom=bottoms, label=label, color=palette[key], edgecolor="black", linewidth=0.4)
        bottoms+=vals
    ymax=max(0.1, max(totals)*1.20)
    for i, tot in enumerate(totals):
        if is_winner[i]:
            ax3.add_patch(plt.Rectangle((i-0.5,0),1.0,max(tot, bottoms[i]), fill=False, lw=2.0, ec="#d4af37"))
            horses[i]=f"★ {horses[i]}"
        ax3.text(i, tot + ymax*0.03, f"{tot:.2f}", ha="center", va="bottom", fontsize=9)
    ax3.set_xticks(x); ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylim(0, ymax); ax3.set_ylabel("PI (stacked contributions)")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5,-0.18), ncol=4, frameon=False)
    st.pyplot(fig3)
    st.caption("Slices are rescaled to sum exactly to each horse’s PI. ★ = race winner.")
else:
    st.info("No PI values available to plot the stacked contributions.")

# -------- Hidden Horses v2 --------
st.markdown("## Hidden Horses (v2) — 200 m grid")
hh=metrics.copy()
need_cols={"tsSPI","Accel","Grind"}
if need_cols.issubset(hh.columns) and len(hh)>0:
    ts_w=winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
    ac_w=winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
    gr_w=winsorize(pd.to_numeric(hh["Grind"], errors="coerce"))
    def rz(s: pd.Series)->pd.Series:
        mu=np.nanmedian(s); sd=mad_std(s)
        if not np.isfinite(sd) or sd==0: return pd.Series(np.zeros(len(s)), index=s.index)
        return (s-mu)/sd
    z_ts=rz(ts_w).clip(-2.5,3.5); z_ac=rz(ac_w).clip(-2.5,3.5); z_gr=rz(gr_w).clip(-2.5,3.5)
    hh["SOS_raw"]=0.45*z_ts+0.35*z_ac+0.20*z_gr
    q5,q95=hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom=(q95-q5) if (pd.notna(q95) and pd.notna(q5) and (q95>q5)) else 1.0
    hh["SOS"]=(2.0*(hh["SOS_raw"]-q5)/denom).clip(lower=0.0, upper=2.0)
else:
    hh["SOS"]=0.0

acc_med=pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
grd_med=pd.to_numeric(hh.get("Grind"), errors="coerce").median(skipna=True)
bias=(acc_med-100.0)-(grd_med-100.0); B=min(1.0, abs(bias)/4.0)
S=pd.to_numeric(hh.get("Accel"), errors="coerce")-pd.to_numeric(hh.get("Grind"), errors="coerce")
hh["ASI2"]=((B*((-S).clip(lower=0.0)) if bias>=0 else B*(S.clip(lower=0.0)))/5.0).fillna(0.0)

def tfs_row(r):
    last3_cols=[c for c in ["600_Time","400_Time","200_Time"] if c in r.index]
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

hh["TFS"]=hh.apply(tfs_row, axis=1)
D_rounded=int(np.ceil(float(race_distance_input)/200.0)*200)
gate=4.0 if D_rounded<=1200 else (3.5 if D_rounded<1800 else 3.0)
def tfs_plus(x): 
    if pd.isna(x) or x<gate: return 0.0
    return min(0.6, (x-gate)/3.0)
hh["TFS_plus"]=hh["TFS"].apply(tfs_plus)

def uei_row(r):
    ts=pd.to_numeric(r.get("tsSPI"), errors="coerce")
    ac=pd.to_numeric(r.get("Accel"), errors="coerce")
    gr=pd.to_numeric(r.get("Grind"), errors="coerce")
    if pd.isna(ts) or pd.isna(ac) or pd.isna(gr): return 0.0
    val=0.0
    if ts>=102 and ac<=98 and gr<=98:
        gap=min((ts-102)/3.0, 1.0); val=max(val, 0.3+0.3*gap)
    if ts>=102 and gr>=102 and ac<=100:
        gap=min(((ts-102)+(gr-102))/6.0, 1.0); val=max(val, 0.3+0.3*gap)
    return round(val, 3)
hh["UEI"]=hh.apply(uei_row, axis=1)

hidden=(0.55*pd.to_numeric(hh["SOS"], errors="coerce").fillna(0.0) +
        0.30*pd.to_numeric(hh["ASI2"], errors="coerce").fillna(0.0) +
        0.10*pd.to_numeric(hh["TFS_plus"], errors="coerce").fillna(0.0) +
        0.05*pd.to_numeric(hh["UEI"], errors="coerce").fillna(0.0))
if int(hh.shape[0])<=6: hidden=hidden*0.90
h_med=float(np.nanmedian(hidden)); h_mad=float(np.nanmedian(np.abs(hidden-h_med))); h_sigma=max(1e-6, 1.4826*h_mad)
hh["HiddenScore"]=(1.2+(hidden-h_med)/(2.5*h_sigma)).clip(lower=0.0, upper=3.0)

def hh_tier(s):
    if pd.isna(s): return ""
    if s>=1.8: return "🔥 Top Hidden"
    if s>=1.2: return "🟡 Notable Hidden"
    return ""
hh["Tier"]=hh["HiddenScore"].apply(hh_tier)
def hh_note(r):
    bits=[]
    if r.get("Tier","")!="":
        if pd.to_numeric(r.get("SOS"), errors="coerce")>=1.2: bits.append("sectionals superior")
        asi2=pd.to_numeric(r.get("ASI2"), errors="coerce")
        if asi2>=0.8: bits.append("ran against strong bias")
        elif asi2>=0.4: bits.append("ran against bias")
        if pd.to_numeric(r.get("TFS_plus"), errors="coerce")>0: bits.append("trip friction late")
        if pd.to_numeric(r.get("UEI"), errors="coerce")>=0.5: bits.append("latent potential if shape flips")
    return ("; ".join(bits).capitalize()+".") if bits else ""
hh["Note"]=hh.apply(hh_note, axis=1)

cols_hh=["Horse","Finish_Pos","PI","GCI","tsSPI","Accel","Grind","SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
for c in cols_hh:
    if c not in hh.columns: hh[c]=np.nan
st.dataframe(hh.sort_values(["Tier","HiddenScore","PI"], ascending=[True,False,False])[cols_hh], use_container_width=True)

st.caption("Conventions — 200m grid. `X_Time` = time from (X+200)→X. `Finish_Time` = 200→0. "
           "Stages: F200=(D−200); tsSPI=(D−400)…600; Accel=400+200; Grind=Finish. "
           "Indices are vs-field (100=par) with small-field stabilizers. "
           "PI v3.1 uses distance+context weights; GCI aligns to the same worldview.")
