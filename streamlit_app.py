import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="Sectional CSV Builder — Race Edge format", layout="wide")

st.title("Sectional CSV Builder")
st.caption("Generates CSVs in Kiran's standard Race Edge format (100m/200m splits)")

# -------------------- Helpers --------------------

def parse_time_to_seconds(x):
    """Accepts 'm:ss.xx', 'ss.xx', 'm:ss', 'ss', or already-decimal. Returns float seconds or NaN."""
    if pd.isna(x) or x == "":
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = s.replace(",", ".")
    if ":" in s:
        parts = s.split(":")
        try:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        except:
            return np.nan
    else:
        try:
            return float(s)
        except:
            return np.nan

def seconds_to_str(x):
    if pd.isna(x):
        return ""
    m, s = divmod(float(x), 60)
    if m >= 1:
        return f"{int(m)}:{s:05.2f}"
    return f"{s:.2f}"

def build_markers(distance_m:int, step:int):
    """
    Build descending distance markers from (distance-step) down to 0 (finish).
    Example: distance=1600, step=200 -> [1400,1200,1000,800,600,400,200,0]
    Example: distance=1000, step=100 -> [900,800,...,0]
    """
    start = distance_m - step
    markers = list(range(start, -1, -step))
    if markers[-1] != 0:
        markers.append(0)
    return markers

def ordered_columns(distance_m:int, step:int):
    """
    Standard Race Edge CSV order:
    Draw, Horse, Jockey, Trainer, Horse Weight, Weight Allocated,
    then markers descending with _Time and _Pos pairs,
    then Finish_Time, Finish_Pos, Race Time, 800-400, 400-Finish
    """
    cols = ["Draw", "Horse", "Jockey", "Trainer", "Horse Weight", "Weight Allocated"]
    markers = build_markers(distance_m, step)
    for m in markers[:-1]:
        cols.append(f"{m}_Time")
        cols.append(f"{m}_Pos")
    cols += ["Finish_Time", "Finish_Pos", "Race Time", "800-400", "400-Finish"]
    return cols, markers

def detect_and_fix_halved_times(df, time_cols):
    """
    If (median of non-first split times) < 0.6 * first_split_time, assume halved and double them (except first split).
    Mirrors your frequent issue where all times are halved besides the first.
    """
    if not time_cols:
        return df
    first = time_cols[0]
    sec = []
    for c in time_cols:
        sec.append(df[c].apply(parse_time_to_seconds))
    sec_arr = np.vstack([s.values for s in sec])
    first_med = np.nanmedian(sec_arr[0, :]) if sec_arr.shape[0] > 0 else np.nan
    others = sec_arr[1:, :].flatten()
    others_med = np.nanmedian(others) if others.size else np.nan
    if np.isfinite(first_med) and np.isfinite(others_med) and others_med < 0.6 * first_med:
        for c in time_cols[1:]:
            df[c] = df[c].apply(lambda v: parse_time_to_seconds(v) * 2 if pd.notna(v) and str(v) != "" else v)
    return df

def normalize_time_formats(df, time_cols):
    """Normalize all time columns to decimal seconds (string with 2dp)."""
    for c in time_cols:
        df[c] = df[c].apply(lambda v: round(parse_time_to_seconds(v), 2) if pd.notna(v) and str(v) != "" else v)
    if "Finish_Time" in df.columns:
        df["Finish_Time"] = df["Finish_Time"].apply(lambda v: round(parse_time_to_seconds(v), 2) if pd.notna(v) and str(v) != "" else v)
    return df

def compute_derived_segments(df, distance_m, step, markers):
    """
    Compute 'Race Time', '800-400', and '400-Finish' when possible.
    - 'Race Time' = sum of all available segment times including Finish_Time if present.
    - '800-400' sums segment times between 800 and 400 (handles 100m or 200m steps).
    - '400-Finish' sums segment times from 400 to Finish (handles 100m or 200m steps).
    """
    time_cols = [f"{m}_Time" for m in markers[:-1]]

    def sum_times(row):
        total = 0.0
        have_any = False
        for c in time_cols:
            if c in row and pd.notna(row[c]) and str(row[c]) != "":
                total += parse_time_to_seconds(row[c])
                have_any = True
        if "Finish_Time" in row and pd.notna(row["Finish_Time"]) and str(row["Finish_Time"]) != "":
            total += parse_time_to_seconds(row["Finish_Time"])
            have_any = True
        return round(total, 2) if have_any else np.nan

    df["Race Time"] = df.apply(sum_times, axis=1)

    # 800-400
    if "800_Time" in df.columns and "400_Time" in df.columns:
        if step == 200:
            for i, row in df.iterrows():
                segs = []
                for m in [800, 600]:
                    c = f"{m}_Time"
                    if c in df.columns and pd.notna(row[c]) and str(row[c]) != "":
                        segs.append(parse_time_to_seconds(row[c]))
                df.at[i, "800-400"] = round(sum(segs), 2) if segs else np.nan
        else:
            for i, row in df.iterrows():
                segs = []
                for m in [800,700,600,500,400]:
                    c = f"{m}_Time"
                    if c in df.columns and pd.notna(row[c]) and str(row[c]) != "":
                        segs.append(parse_time_to_seconds(row[c]))
                df.at[i, "800-400"] = round(sum(segs), 2) if segs else np.nan

    # 400-Finish
    if "400_Time" in df.columns:
        for i, row in df.iterrows():
            segs = []
            if step == 200:
                for m in [400, 200]:
                    c = f"{m}_Time"
                    if c in df.columns and pd.notna(row[c]) and str(row[c]) != "":
                        segs.append(parse_time_to_seconds(row[c]))
                if "Finish_Time" in df.columns and pd.notna(row["Finish_Time"]) and str(row["Finish_Time"]) != "":
                    segs.append(parse_time_to_seconds(row["Finish_Time"]))
            else:
                for m in [400,300,200,100]:
                    c = f"{m}_Time"
                    if c in df.columns and pd.notna(row[c]) and str(row[c]) != "":
                        segs.append(parse_time_to_seconds(row[c]))
                if "Finish_Time" in df.columns and pd.notna(row["Finish_Time"]) and str(row["Finish_Time"]) != "":
                    segs.append(parse_time_to_seconds(row["Finish_Time"]))
            df.at[i, "400-Finish"] = round(sum(segs), 2) if segs else np.nan

    return df

def make_blank_df(distance_m:int, step:int, n:int):
    cols, markers = ordered_columns(distance_m, step)
    df = pd.DataFrame([{c: "" for c in cols} for _ in range(n)])
    df["Draw"] = ""
    df["Horse Weight"] = ""
    df["Weight Allocated"] = ""
    return df, markers

def enforce_finish_pos_by_row(df):
    if "Finish_Pos" in df.columns:
        df["Finish_Pos"] = [i+1 for i in range(len(df))]
    return df

def reorder_columns(df, distance_m, step):
    desired, markers = ordered_columns(distance_m, step)
    extra = [c for c in df.columns if c not in desired]
    final = desired + extra
    return df.reindex(columns=final)

# -------------------- UI --------------------

left, right = st.columns([1,1])

with left:
    distance_m = st.number_input("Race distance (m)", min_value=800, max_value=3600, step=50, value=1600)
    split_step = st.radio("Split step", options=[200, 100], index=0, horizontal=True)
    n_runners = st.number_input("Number of runners (rows)", min_value=1, max_value=30, step=1, value=12)

    auto_finish_pos = st.checkbox("Auto-set Finish_Pos by row order (Race Edge rule)", value=True)
    normalize_times = st.checkbox("Normalize time formats to decimal seconds", value=True)
    fix_halved = st.checkbox("Auto-fix halved times (all except first split)", value=True)

with right:
    st.markdown("**Quick notes**")
    st.markdown("""
    - Columns follow your standard: **Draw, Horse, Jockey, Trainer, Horse Weight, Weight Allocated**, then split pairs (descending), then **Finish_Time, Finish_Pos, Race Time, 800-400, 400-Finish**.
    - Paste segment times as **per-split segment times** (e.g. `200_Time` is the 200m segment, not cumulative).
    - You can type times as `m:ss.xx`, `ss.xx`, or plain decimals; we'll normalize.
    - If your data suffers the usual error (all segments halved except the first), check **Auto-fix halved times**.
    """)

st.divider()

# Create or upload
st.subheader("1) Start with a blank template or upload a partial CSV")
template_df, markers = make_blank_df(distance_m, split_step, n_runners)

colA, colB = st.columns([1,1])
with colA:
    st.markdown("**Blank template**")
    st.dataframe(template_df.head(5))
    csv_template = template_df.to_csv(index=False)
    st.download_button("⬇️ Download blank template CSV", csv_template, file_name=f"race_{distance_m}m_{split_step}splits_template.csv", mime="text/csv")

with colB:
    st.markdown("**Upload to edit/finish**")
    uploaded = st.file_uploader("Upload an existing/partial CSV (optional)", type=["csv"])
    if uploaded:
        user_df = pd.read_csv(uploaded, dtype=str).fillna("")
    else:
        user_df = template_df.copy()

st.subheader("2) Edit your sheet")
st.caption("Tip: sort the rows by finishing order first if you're using auto Finish_Pos.")
edited_df = st.data_editor(user_df, num_rows="dynamic", use_container_width=True)

# Process & export
st.subheader("3) Process & export (Race Edge format)")
desired_cols, markers = ordered_columns(distance_m, split_step)
time_cols = [f"{m}_Time" for m in markers[:-1]]

df_out = edited_df.copy()

if normalize_times:
    df_out = normalize_time_formats(df_out, time_cols + (["Finish_Time"] if "Finish_Time" in df_out.columns else []))

if fix_halved:
    df_out = detect_and_fix_halved_times(df_out, time_cols + (["Finish_Time"] if "Finish_Time" in df_out.columns else []))

if auto_finish_pos:
    df_out = enforce_finish_pos_by_row(df_out)

df_out = compute_derived_segments(df_out, distance_m, split_step, markers)
df_out = reorder_columns(df_out, distance_m, split_step)

st.dataframe(df_out.head(len(df_out)), use_container_width=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (Race Edge format)", csv_bytes, file_name=f"race_{distance_m}m_{split_step}splits.csv", mime="text/csv")

st.success("Ready! Your CSV follows the standard spec and includes derived fields when possible.")
