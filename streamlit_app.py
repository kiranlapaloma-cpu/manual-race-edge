
import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any

try:
    import requests
except Exception:
    requests = None

st.set_page_config(page_title="Sectional CSV Builder — Race Edge format", layout="wide")

st.title("Sectional CSV Builder")
st.caption("Generates CSVs in Kiran's standard Race Edge format (100m/200m splits) + Gallop JSON importer")

# -------------------- Helpers --------------------

def parse_time_to_seconds(x):
    if pd.isna(x) or x == "":
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    if ":" in s:
        parts = s.split(":")
        try:
            minutes = float(parts[0]); seconds = float(parts[1])
            return minutes * 60.0 + seconds
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def build_markers(distance_m:int, step:int):
    start = distance_m - step
    markers = list(range(start, -1, -step))
    if markers[-1] != 0:
        markers.append(0)
    return markers

def ordered_columns(distance_m:int, step:int):
    cols = ["Draw", "Horse", "Jockey", "Trainer", "Horse Weight", "Weight Allocated"]
    markers = build_markers(distance_m, step)
    for m in markers[:-1]:
        cols.append(f"{m}_Time"); cols.append(f"{m}_Pos")
    cols += ["Finish_Time", "Finish_Pos", "Race Time", "800-400", "400-Finish"]
    return cols, markers

def detect_and_fix_halved_times(df, time_cols):
    if not time_cols: return df
    sec = [df[c].apply(parse_time_to_seconds) for c in time_cols]
    sec_arr = np.vstack([s.values for s in sec])
    first_med = np.nanmedian(sec_arr[0, :]) if sec_arr.shape[0] > 0 else np.nan
    others = sec_arr[1:, :].flatten()
    others_med = np.nanmedian(others) if others.size else np.nan
    if np.isfinite(first_med) and np.isfinite(others_med) and others_med < 0.6 * first_med:
        for c in time_cols[1:]:
            df[c] = df[c].apply(lambda v: parse_time_to_seconds(v) * 2 if pd.notna(v) and str(v) != "" else v)
    return df

def normalize_time_formats(df, time_cols):
    for c in time_cols:
        df[c] = df[c].apply(lambda v: round(parse_time_to_seconds(v), 2) if pd.notna(v) and str(v) != "" else v)
    if "Finish_Time" in df.columns:
        df["Finish_Time"] = df["Finish_Time"].apply(lambda v: round(parse_time_to_seconds(v), 2) if pd.notna(v) and str(v) != "" else v)
    return df

def compute_derived_segments(df, distance_m, step, markers):
    time_cols = [f"{m}_Time" for m in markers[:-1]]

    def sum_times(row):
        total = 0.0; have_any = False
        for c in time_cols:
            if c in row and pd.notna(row[c]) and str(row[c]) != "":
                total += parse_time_to_seconds(row[c]); have_any = True
        if "Finish_Time" in row and pd.notna(row["Finish_Time"]) and str(row["Finish_Time"]) != "":
            total += parse_time_to_seconds(row["Finish_Time"]); have_any = True
        return round(total, 2) if have_any else np.nan

    df["Race Time"] = df.apply(sum_times, axis=1)

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
    df["Draw"] = ""; df["Horse Weight"] = ""; df["Weight Allocated"] = ""
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

# -------- Gallop JSON importer --------

def fetch_json_from_url(url:str):
    if requests is None:
        return None, "The 'requests' package is not available."
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def normalize_gallop_payload(payload:Any)->List[Dict[str,Any]]:
    runners = []
    if payload is None:
        return runners

    if isinstance(payload, dict) and "sectionals" in payload and isinstance(payload["sectionals"], list):
        for obj in payload["sectionals"]:
            horse = obj.get("horse") or obj.get("runner") or obj.get("name") or obj.get("Horse") or ""
            secs = obj.get("sections", [])
            clean_secs = []
            for s in secs:
                try:
                    end = int(float(str(s.get("end","0")).replace("m","").strip()))
                except:
                    continue
                time_sec = s.get("timeSec")
                try:
                    time_sec = float(time_sec) if time_sec is not None else None
                except:
                    time_sec = None
                try:
                    rank_sec = int(str(s.get("rankSec","")).strip()) if s.get("rankSec") not in (None,"") else None
                except:
                    rank_sec = None
                clean_secs.append({"end": end, "timeSec": time_sec, "rankSec": rank_sec})
            runners.append({"horse": horse, "sections": clean_secs})
        return runners

    if isinstance(payload, list):
        for obj in payload:
            horse = obj.get("horse") or obj.get("runner") or obj.get("name") or ""
            secs = obj.get("sections", [])
            clean_secs = []
            for s in secs:
                try:
                    end = int(float(str(s.get("end","0")).replace("m","").strip()))
                except:
                    continue
                time_sec = s.get("timeSec")
                try:
                    time_sec = float(time_sec) if time_sec is not None else None
                except:
                    time_sec = None
                try:
                    rank_sec = int(str(s.get("rankSec","")).strip()) if s.get("rankSec") not in (None,"") else None
                except:
                    rank_sec = None
                clean_secs.append({"end": end, "timeSec": time_sec, "rankSec": rank_sec})
            runners.append({"horse": horse, "sections": clean_secs})
        return runners

    return runners

def build_df_from_gallop(runners:List[Dict[str,Any]], distance_m:int, step:int)->pd.DataFrame:
    cols, markers = ordered_columns(distance_m, step)
    df = pd.DataFrame([{c:"" for c in cols} for _ in range(len(runners))])
    for i, r in enumerate(runners):
        df.at[i, "Horse"] = (r.get("horse","") or "").strip()
        by_end = {s["end"]: s for s in r.get("sections",[]) if isinstance(s.get("end"), int)}
        for m in markers[:-1]:
            if m in by_end:
                seg = by_end[m]
                if seg.get("timeSec") is not None:
                    df.at[i, f"{m}_Time"] = round(float(seg["timeSec"]), 2)
                if seg.get("rankSec") is not None:
                    df.at[i, f"{m}_Pos"] = seg["rankSec"]
        if 0 in by_end and by_end[0].get("timeSec") is not None:
            df.at[i, "Finish_Time"] = round(float(by_end[0]["timeSec"]), 2)
    return df

# -------------------- UI --------------------

tab_manual, tab_gallop = st.tabs(["✍️ Manual / CSV workflow", "🛰️ Import from Gallop JSON"])

with tab_manual:
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
        - Paste segment times as **per-split segment times**.
        - You can type times as `m:ss.xx`, `ss.xx`, or decimals; we'll normalize.
        - If your data suffers the usual error (all segments halved except the first), check **Auto-fix halved times**.
        """)

    st.divider()

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

with tab_gallop:
    st.subheader("Import directly from Gallop JSON")
    st.caption("Paste a Gallop JSON URL or raw JSON payload for a race (e.g., from gallop.php?feed=sectional&date=YYYYMMDD&...).")

    gcol1, gcol2 = st.columns([2,1])
    with gcol1:
        gallop_url = st.text_input("Gallop JSON URL (optional)")
    with gcol2:
        distance_m_g = st.number_input("Race distance (m)", min_value=800, max_value=3600, step=50, value=1600, key="dist_g")
        split_step_g = st.radio("Split step", options=[200, 100], index=0, horizontal=True, key="step_g")

    raw_json_text = st.text_area("...or paste the raw JSON here (we'll auto-detect shape)", height=200)

    if st.button("Fetch & Build CSV"):
        payload = None; err = None
        if gallop_url.strip():
            payload, err = fetch_json_from_url(gallop_url.strip())
            if err: st.error(f"Fetch error: {err}")
        elif raw_json_text.strip():
            try:
                payload = json.loads(raw_json_text.strip())
            except Exception as e:
                st.error(f"Invalid JSON pasted: {e}")
        else:
            st.error("Please provide a JSON URL or paste JSON.")

        if payload is not None:
            runners = normalize_gallop_payload(payload)
            if not runners:
                st.warning("Parsed JSON but couldn't find 'sectionals'/'sections'. Paste the exact response from the feed.")
            else:
                st.info(f"Parsed {len(runners)} runners from JSON.")
                df_g = build_df_from_gallop(runners, int(distance_m_g), int(split_step_g))

                desired_cols_g, markers_g = ordered_columns(int(distance_m_g), int(split_step_g))
                time_cols_g = [f"{m}_Time" for m in markers_g[:-1]]
                df_g = normalize_time_formats(df_g, time_cols_g + (["Finish_Time"] if "Finish_Time" in df_g.columns else []))
                df_g = enforce_finish_pos_by_row(df_g)
                df_g = compute_derived_segments(df_g, int(distance_m_g), int(split_step_g), markers_g)
                df_g = reorder_columns(df_g, int(distance_m_g), int(split_step_g))

                st.dataframe(df_g, use_container_width=True)
                csv_bytes_g = df_g.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV from Gallop JSON", csv_bytes_g, file_name=f"race_{distance_m_g}m_{split_step_g}splits_from_gallop.csv", mime="text/csv")
