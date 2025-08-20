# heymath_su_builder.py

import streamlit as st
import pandas as pd
import zipfile, io, csv, re, os

# Page config MUST be the first Streamlit call
st.set_page_config(page_title="HeyMath — Report Builder (SU + TU)", layout="wide")


# --------------------------- Mapping loader ---------------------------

def _load_mapping_if_present(folder: str):
    """
    report_mapping_proposal.xlsx / _report_mapping_proposal.xlsx / report_mapping_proposal.csv
    → returns (mapping_df, path) or (None, None)
    """
    candidates = [
        os.path.join(folder, "report_mapping_proposal.xlsx"),
        os.path.join(folder, "_report_mapping_proposal.xlsx"),
        os.path.join(folder, "report_mapping_proposal.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            m = pd.read_excel(p) if p.lower().endswith(".xlsx") else pd.read_csv(p)
            # normalize columns; accept either 'source_column' or 'suggested_source_column'
            m.columns = [str(c).strip().lower().replace(" ", "_") for c in m.columns]
            if "source_column" not in m.columns and "suggested_source_column" in m.columns:
                m = m.rename(columns={"suggested_source_column": "source_column"})
            for col in ["target_sheet", "target_column", "source_csv", "source_column", "aggregate", "transform"]:
                if col not in m.columns:
                    m[col] = ""
            return m, p
    return None, None


# Load mapping once
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_DF, MAPPING_PATH = _load_mapping_if_present(SCRIPT_DIR)
if MAPPING_PATH:
    st.caption(f"Using mapping: `{os.path.basename(MAPPING_PATH)}`")


# --------------------------- Helpers ---------------------------

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def read_csv_flex_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="replace")
    # delimiter sniff
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=[",", ";", "\t", "|"])
        delim = dialect.delimiter
    except Exception:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        delim = "," if not lines else max([",", ";", "\t", "|"], key=lambda d: lines[0].count(d))
    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, engine="python")
    df = df.loc[:, ~df.columns.to_series().astype(str).str.match(r"^Unnamed")]
    return df


# --------------------------- ZIP detection ---------------------------

def detect_files_in_zip(zip_bytes: bytes, mapping_df: pd.DataFrame | None = None):
    """
    Return (assign_df, lessons_df, logins_df, teachers_df, names).
    - Prefer 'School Assignments Usage ...csv' for assignments (fallback: aggregate level files).
    - Prefer 'School Lessons Usage ...csv' for lessons (fallback: aggregate level files).
    - Detect Logins by filename or headers.
    - Detect Teachers Usage (e.g., 'All Teachers Usage Logins') by mapping hint or headers.
    """
    assign_df = lessons_df = logins_df = teachers_df = None
    names = {"assign": None, "lessons": None, "logins": None, "teachers": None}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]

        # Read all CSVs once
        read_cache: dict[str, pd.DataFrame] = {}
        for name in csv_names:
            with zf.open(name) as f:
                data = f.read()
            df = read_csv_flex_from_bytes(data)
            read_cache[name] = df

        # helper: pick first file whose name contains pattern (shortest name wins)
        def pick_by_pattern(pattern: str):
            if not pattern:
                return (None, None)
            pat = pattern.lower()
            cands = [(name, df) for name, df in read_cache.items() if pat in name.lower()]
            if not cands:
                return (None, None)
            cands.sort(key=lambda x: len(x[0]))
            return cands[0]

        # -------- 0) Mapping hints (if provided) --------
        if mapping_df is not None:
            su_map = mapping_df[mapping_df["target_sheet"].astype(str).str.strip().str.lower()
                                .isin(["su", "students usage", "student usage", "students_usage"])]
            tu_map = mapping_df[mapping_df["target_sheet"].astype(str).str.strip().str.lower()
                                .isin(["tu", "teacher usage", "teachers usage"])]
            # SU hints
            for hint in su_map["source_csv"].dropna().astype(str).unique().tolist():
                name, df = pick_by_pattern(hint)
                if not name or df is None:
                    continue
                low = name.lower()
                if ("assign" in low or "assignments usage" in low) and assign_df is None:
                    assign_df, names["assign"] = df, name
                if ("lessons usage" in low) and lessons_df is None:
                    lessons_df, names["lessons"] = df, name
                if ("logins" in low) and logins_df is None:
                    logins_df, names["logins"] = df, name
            # TU hints
            for hint in tu_map["source_csv"].dropna().astype(str).unique().tolist():
                name, df = pick_by_pattern(hint)
                if name and df is not None and teachers_df is None:
                    teachers_df, names["teachers"] = df, name

        # -------- 1) Logins --------
        if logins_df is None:
            for name, df in read_cache.items():
                low = norm(name)
                cols = [norm(c) for c in df.columns]
                if ("logins report" in low) or ("total_students" in cols) or ("total_logins" in cols):
                    logins_df, names["logins"] = df, name
                    break

        # -------- 2) Assignments (prefer 'School Assignments Usage') --------
        if assign_df is None:
            school_assign, level_assign, other_assign = [], [], []
            for name, df in read_cache.items():
                low = norm(name)
                cols = [norm(c) for c in df.columns]
                looks_like = ("assignment" in low) or any(k in " ".join(cols) for k in
                                ["ongoing_aqc", "ongoing_prasso", "ongoing_worksheet", "ongoing_reading", "prasso"])
                if not looks_like:
                    continue
                if "school assignments usage" in low:
                    school_assign.append((name, df))
                elif "assignments usage" in low:
                    level_assign.append((name, df))
                else:
                    other_assign.append((name, df))

            if school_assign:
                name, df = school_assign[0]
                assign_df, names["assign"] = df, name
            elif level_assign:
                # aggregate all level-assignment files by LEVEL
                parts = []
                for name, df in level_assign:
                    level_col = next((c for c in df.columns if "level" in norm(c)), None)
                    metrics = [c for c in df.columns if any(k in norm(c) for k in
                               ["ongoing_aqc", "ongoing_worksheet", "ongoing_prasso", "ongoing_reading"])]
                    if not (level_col and metrics):
                        continue
                    d = df[[level_col] + metrics].copy()
                    for c in metrics:
                        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
                    g = d.groupby(level_col, as_index=False).sum()
                    # standardize column names
                    ren = {level_col: "LEVEL"}
                    for m in metrics:
                        lm = norm(m)
                        if "ongoing_aqc" in lm: ren[m] = "ONGOING_AQC"
                        if "ongoing_worksheet" in lm: ren[m] = "ONGOING_WORKSHEET"
                        if "ongoing_prasso" in lm: ren[m] = "ONGOING_PRASSO"
                        if "ongoing_reading" in lm: ren[m] = "ONGOING_READING"
                    parts.append(g.rename(columns=ren))
                if parts:
                    agg = parts[0]
                    for p in parts[1:]:
                        agg = agg.merge(p, on="LEVEL", how="outer")
                    for col in ["ONGOING_AQC", "ONGOING_WORKSHEET", "ONGOING_PRASSO", "ONGOING_READING"]:
                        if col in agg.columns:
                            agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0)
                    assign_df = agg.groupby("LEVEL", as_index=False).sum(numeric_only=True)
                    names["assign"] = f"Aggregated {len(parts)} Assignments files"
            elif other_assign:
                name, df = other_assign[0]
                assign_df, names["assign"] = df, name

        # -------- 3) Lessons (prefer 'School Lessons Usage') --------
        if lessons_df is None:
            school_lessons, level_lessons, other_lessons = [], [], []
            for name, df in read_cache.items():
                low = norm(name)
                cols = [norm(c) for c in df.columns]
                if not any("lesson" in c for c in cols):
                    continue
                if "school lessons usage" in low:
                    school_lessons.append((name, df))
                elif "level lessons usage" in low:
                    level_lessons.append((name, df))
                elif "lessons usage" in low:
                    other_lessons.append((name, df))

            if school_lessons:
                name, df = school_lessons[0]
                lessons_df, names["lessons"] = df, name
            elif level_lessons:
                parts = []
                for name, df in level_lessons:
                    level_col = next((c for c in df.columns if "level" in norm(c)), None)
                    les_col = next((c for c in df.columns if "lesson" in norm(c)), None)
                    if not (level_col and les_col):
                        continue
                    d = df[[level_col, les_col]].copy()
                    d[les_col] = pd.to_numeric(d[les_col], errors="coerce").fillna(0)
                    d = d.groupby(level_col, as_index=False)[les_col].sum()
                    d.columns = ["LEVEL", "LESSONS_ACCESSED"]
                    parts.append(d)
                if parts:
                    agg = pd.concat(parts, ignore_index=True).groupby("LEVEL", as_index=False)["LESSONS_ACCESSED"].sum()
                    lessons_df, names["lessons"] = agg, f"Aggregated {len(parts)} Level Lessons files"
            elif other_lessons:
                best = max(other_lessons, key=lambda t: (t[1]["LEVEL"].nunique() if "LEVEL" in t[1].columns else 0))
                lessons_df, names["lessons"] = best[1], best[0]

        # -------- 4) Teachers Usage (for TU) --------
        if teachers_df is None:
            # 4a) file-name preference
            for name, df in read_cache.items():
                low = norm(name)
                if "teachers usage" in low or "all teachers usage" in low:
                    teachers_df, names["teachers"] = df, name
                    break
            # 4b) header heuristics
            if teachers_df is None:
                for name, df in read_cache.items():
                    cols = [norm(c) for c in df.columns]
                    if any(("user" in c) or ("teacher" in c) for c in cols) and any("login" in c for c in cols):
                        teachers_df, names["teachers"] = df, name
                        break

    return assign_df, lessons_df, logins_df, teachers_df, names


# --------------------------- Builders ---------------------------

def build_su(assign_df: pd.DataFrame, lessons_df: pd.DataFrame, logins_df: pd.DataFrame,
             mapping_df: pd.DataFrame | None = None) -> pd.DataFrame:
    # --- Optional: apply mapping-driven renames (stabilize headers) ---
    if mapping_df is not None:
        su_map = mapping_df[mapping_df["target_sheet"].astype(str).str.strip().str.lower()
                            .isin(["su", "students usage", "student usage", "students_usage"])]
        def rename_from_map(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
            sub = su_map[su_map["source_csv"].astype(str).str.lower().str.contains(source_label)]
            rename_pairs = {}
            for _, r in sub.iterrows():
                src = str(r.get("source_column", "")).strip()
                tgt = str(r.get("target_column", "")).strip()
                if not (src and tgt and (src in df.columns)):
                    continue
                # Map SU target names to internal canonical column names
                canon = {
                    "Class": "LEVEL_DISPLAY_NAME",
                    "No of Students": "TOTAL_STUDENTS",
                    "No of Logins": "TOTAL_LOGINS",
                    "No of Lessons Accessed": "LESSONS_ACCESSED",
                    "Quiz": "ONGOING_AQC",
                    "Worksheet": "ONGOING_WORKSHEET",
                    "Prasso": "ONGOING_PRASSO",
                    "Reading": "ONGOING_READING",
                }.get(tgt, tgt)
                rename_pairs[src] = canon
            return df.rename(columns=rename_pairs) if rename_pairs else df

        logins_df  = rename_from_map(logins_df,  "login")
        lessons_df = rename_from_map(lessons_df, "lesson")
        assign_df  = rename_from_map(assign_df,  "assign")

    # --- Normalize numerics ---
    for c in ["TOTAL_STUDENTS", "ACTIVE_STUDENTS", "TOTAL_LOGINS"]:
        if c in logins_df.columns:
            logins_df[c] = pd.to_numeric(logins_df[c], errors="coerce").fillna(0).astype(int)
    if "LESSONS_ACCESSED" in lessons_df.columns:
        lessons_df["LESSONS_ACCESSED"] = pd.to_numeric(lessons_df["LESSONS_ACCESSED"], errors="coerce").fillna(0).astype(int)
    for c in ["ONGOING_AQC", "ONGOING_WORKSHEET", "ONGOING_PRASSO", "ONGOING_READING"]:
        if c in assign_df.columns:
            assign_df[c] = pd.to_numeric(assign_df[c], errors="coerce").fillna(0).astype(int)

    # --- Base from LOGINS ---
    needed = ["LEVEL", "LEVEL_DISPLAY_NAME", "SORT_ORDER", "TOTAL_STUDENTS", "TOTAL_LOGINS"]
    missing = [c for c in needed if c not in logins_df.columns]
    if missing:
        raise ValueError(f"Logins CSV missing required columns: {missing}")

    base = logins_df[["LEVEL", "LEVEL_DISPLAY_NAME", "SORT_ORDER", "TOTAL_STUDENTS", "TOTAL_LOGINS"]].copy()
    base = base.rename(columns={
        "LEVEL_DISPLAY_NAME": "Class",
        "TOTAL_STUDENTS": "No of Students",
        "TOTAL_LOGINS": "No of Logins",
    })

    # --- Lessons (aggregate) ---
    less = pd.DataFrame(columns=["LEVEL", "No of Lessons Accessed"])
    if "LEVEL" in lessons_df.columns and "LESSONS_ACCESSED" in lessons_df.columns:
        tmp = lessons_df[["LEVEL", "LESSONS_ACCESSED"]].copy()
        less = (
            tmp.groupby("LEVEL", as_index=False)["LESSONS_ACCESSED"]
               .sum()
               .rename(columns={"LESSONS_ACCESSED": "No of Lessons Accessed"})
        )

    # --- Assignments (map → Quiz/Worksheet/Prasso/Reading; aggregate) ---
    assign = pd.DataFrame(columns=["LEVEL", "Quiz", "Worksheet", "Prasso", "Reading"])
    if "LEVEL" not in assign_df.columns:
        # best-effort: vendor changed the header
        for c in assign_df.columns:
            if "level" in norm(c):
                assign_df = assign_df.rename(columns={c: "LEVEL"})
                break
    if "LEVEL" in assign_df.columns:
        keep = ["LEVEL"]; mapping = {}
        if "ONGOING_AQC" in assign_df.columns:
            mapping["ONGOING_AQC"] = "Quiz"; keep.append("ONGOING_AQC")
        if "ONGOING_WORKSHEET" in assign_df.columns:
            mapping["ONGOING_WORKSHEET"] = "Worksheet"; keep.append("ONGOING_WORKSHEET")
        if "ONGOING_PRASSO" in assign_df.columns:
            mapping["ONGOING_PRASSO"] = "Prasso"; keep.append("ONGOING_PRASSO")
        if "ONGOING_READING" in assign_df.columns:
            mapping["ONGOING_READING"] = "Reading"; keep.append("ONGOING_READING")

        tmp = assign_df[keep].copy()
        for raw_col in ["ONGOING_AQC", "ONGOING_WORKSHEET", "ONGOING_PRASSO", "ONGOING_READING"]:
            if raw_col in tmp.columns:
                tmp[raw_col] = pd.to_numeric(tmp[raw_col], errors="coerce").fillna(0)
        tmp = tmp.groupby("LEVEL", as_index=False).sum(numeric_only=True)
        assign = tmp.rename(columns=mapping)

        for col in ["Quiz", "Worksheet", "Prasso", "Reading"]:
            if col not in assign.columns:
                assign[col] = 0
        assign = assign[["LEVEL", "Quiz", "Worksheet", "Prasso", "Reading"]]

    # --- Merge + finish ---
    su = base.merge(less, on="LEVEL", how="left").merge(assign, on="LEVEL", how="left")
    for c in ["No of Students", "No of Logins", "No of Lessons Accessed", "Quiz", "Worksheet", "Prasso", "Reading"]:
        if c in su.columns:
            su[c] = pd.to_numeric(su[c], errors="coerce").fillna(0).astype(int)

    # Sorting
    if "SORT_ORDER" in su.columns:
        su = su.sort_values(["SORT_ORDER", "Class"], kind="stable")
    else:
        su["_g"] = su["Class"].astype(str).str.extract(r"(\d+)", expand=False).astype(float).fillna(9999)
        su = su.sort_values(["_g", "Class"], kind="stable").drop(columns="_g")

    cols = ["Class", "No of Students", "No of Logins", "No of Lessons Accessed", "Quiz", "Worksheet", "Prasso", "Reading"]
    for c in cols:
        if c not in su.columns:
            su[c] = 0
    return su[cols]


def build_tu(teachers_df: pd.DataFrame | None, mapping_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the TU sheet from a Teachers Usage CSV.
    Mapping-driven: target_sheet == 'TU' rows in the mapping pick source columns.
    Expected targets (example): Name, No of logins, No of Lessons Accessed, No of Assignments Assigned
    """
    if teachers_df is None or teachers_df.empty:
        return pd.DataFrame(columns=["Name", "No of logins", "No of Lessons Accessed", "No of Assignments Assigned"])

    df = teachers_df.copy()

    # Apply mapping-driven renames if available
    if mapping_df is not None:
        tu_map = mapping_df[mapping_df["target_sheet"].astype(str).str.strip().str.lower()
                            .isin(["tu", "teacher usage", "teachers usage"])].copy()
        if not tu_map.empty:
            rename_pairs = {}
            for _, r in tu_map.iterrows():
                src = str(r.get("source_column", "")).strip()
                tgt = str(r.get("target_column", "")).strip()
                if src and tgt and src in df.columns:
                    rename_pairs[src] = tgt
            if rename_pairs:
                df = df.rename(columns=rename_pairs)

    # Canonical pickers (fallbacks if mapping didn't rename some)
    def pick(*alts):
        for a in alts:
            if a in df.columns:
                return a
        # case-insensitive fallback
        low = {c.lower(): c for c in df.columns}
        for a in alts:
            if a.lower() in low:
                return low[a.lower()]
        return None

    name_col   = pick("Name", "userName", "username", "Teacher", "Teacher Name")
    logins_col = pick("No of logins", "logins", "Login Count", "Logins")
    lessons_col = pick("No of Lessons Accessed", "lesson", "Lessons", "Lessons Accessed")
    assigns_col = pick("No of Assignments Assigned", "assessment", "Assignments", "Assignments Assigned")

    cols = [c for c in [name_col, logins_col, lessons_col, assigns_col] if c]
    if not cols or name_col is None:
        return pd.DataFrame(columns=["Name", "No of logins", "No of Lessons Accessed", "No of Assignments Assigned"])

    tmp = df[cols].copy()
    for c in [logins_col, lessons_col, assigns_col]:
        if c and c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

    # group by teacher name
    tmp = tmp.groupby(name_col, as_index=False).sum(numeric_only=True)

    # Final rename to standard TU headings if needed
    rename_out = {}
    if name_col and name_col != "Name": rename_out[name_col] = "Name"
    if logins_col and logins_col != "No of logins": rename_out[logins_col] = "No of logins"
    if lessons_col and lessons_col != "No of Lessons Accessed": rename_out[lessons_col] = "No of Lessons Accessed"
    if assigns_col and assigns_col != "No of Assignments Assigned": rename_out[assigns_col] = "No of Assignments Assigned"
    tmp = tmp.rename(columns=rename_out)

    # Ensure all expected columns exist
    for c in ["Name", "No of logins", "No of Lessons Accessed", "No of Assignments Assigned"]:
        if c not in tmp.columns:
            tmp[c] = 0

    # Sort teachers by name
    tmp = tmp.sort_values(["Name"], kind="stable")
    return tmp[["Name", "No of logins", "No of Lessons Accessed", "No of Assignments Assigned"]]

# ---- Levelwise helpers ----
# ---- Levelwise helpers (select/rename/sort) ----
def _pick_sort_col(df: pd.DataFrame):
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc == "sort_order" or ("sort" in lc and "order" in lc):
            return c
    return None

def _apply_levelwise_mapping(d: pd.DataFrame, mapping_df: pd.DataFrame | None):
    """If mapping has Levelwise rows, rename source -> canonical internal names."""
    if mapping_df is None or d.empty:
        return d
    m = mapping_df.copy()
    m["target_sheet"] = m["target_sheet"].astype(str).str.strip().str.lower()
    lv = m[m["target_sheet"].isin(["levelwise", "level wise", "level-wise"])]
    if lv.empty:
        return d

    # target_column -> canonical internal name
    target2canon = {
        "level": "LEVEL",
        "class": "LEVEL_DISPLAY_NAME",
        "no of lessons accessed": "LESSONS_ACCESSED",
        "sort_order": "SORT_ORDER",
    }
    rename_pairs = {}
    for _, r in lv.iterrows():
        src = str(r.get("source_column", "")).strip()
        tgt = str(r.get("target_column", "")).strip().lower()
        if src and tgt and src in d.columns and tgt in target2canon:
            rename_pairs[src] = target2canon[tgt]
    return d.rename(columns=rename_pairs) if rename_pairs else d

def build_levelwise_from_frames(frames, mapping_df=None, logins_df=None):
    """Read per-grade Level Lessons frames, sort each by SORT_ORDER,
    keep only LEVEL / LEVEL_DISPLAY_NAME / LESSONS_ACCESSED, then stack."""
    if not frames:
        return pd.DataFrame(columns=["Level","Class","No of Lessons Accessed"])

    cleaned = []
    for df in frames:
        d = df.copy()

        # 1) apply mapping if present
        d = _apply_levelwise_mapping(d, mapping_df)

        # 2) standardize column names (fallbacks)
        if "LEVEL" not in d.columns:
            for c in d.columns:
                if "level" in str(c).lower():
                    d = d.rename(columns={c: "LEVEL"})
                    break
        if "LEVEL_DISPLAY_NAME" not in d.columns:
            for c in d.columns:
                lc = str(c).lower()
                if "level_display_name" in lc or "class" in lc:
                    d = d.rename(columns={c: "LEVEL_DISPLAY_NAME"})
                    break
        if "LESSONS_ACCESSED" not in d.columns:
            for c in d.columns:
                lc = str(c).lower()
                if ("lesson" in lc) and ("access" in lc):
                    d = d.rename(columns={c: "LESSONS_ACCESSED"})
                    break

        # 3) sort within this CSV
        sc = _pick_sort_col(d)
        if sc:
            d[sc] = pd.to_numeric(d[sc], errors="coerce").fillna(9999)
            d = d.sort_values([sc], kind="stable")

        # 4) keep only the three columns, coerce numeric
        keep = [c for c in ["LEVEL","LEVEL_DISPLAY_NAME","LESSONS_ACCESSED"] if c in d.columns]
        d = d[keep].copy()
        if "LESSONS_ACCESSED" in d.columns:
            d["LESSONS_ACCESSED"] = pd.to_numeric(d["LESSONS_ACCESSED"], errors="coerce").fillna(0).astype(int)

        cleaned.append(d)

    # 5) stack all grades
    out = pd.concat(cleaned, ignore_index=True, sort=False)

    # if Class missing, try to map from logins
    if "LEVEL_DISPLAY_NAME" not in out.columns and logins_df is not None:
        if {"LEVEL","LEVEL_DISPLAY_NAME"}.issubset(logins_df.columns):
            out = out.merge(logins_df[["LEVEL","LEVEL_DISPLAY_NAME"]].drop_duplicates(), on="LEVEL", how="left")

    # ensure columns exist, then rename to final headers
    for c in ["LEVEL","LEVEL_DISPLAY_NAME","LESSONS_ACCESSED"]:
        if c not in out.columns:
            out[c] = 0 if c == "LESSONS_ACCESSED" else ""
    out = out[["LEVEL","LEVEL_DISPLAY_NAME","LESSONS_ACCESSED"]].rename(
        columns={"LEVEL":"Level","LEVEL_DISPLAY_NAME":"Class","LESSONS_ACCESSED":"No of Lessons Accessed"}
    )

    # Turn any "Grade 1 TPP Group 1", "Grade 1A-ECR", "G1", "01A0" → "Grade 1"
    def _grade_label_from_text(s):
        s = str(s)
        m = re.search(r'grade\s*([0-9]+)', s, flags=re.I)
        if m:
            return f"Grade {int(m.group(1))}"
        m = re.search(r'([0-9]+)', s)   # fallback: first number
        return f"Grade {int(m.group(1))}" if m else s

    out["Level"] = out["Class"].apply(_grade_label_from_text)

    # final sort: ascending by Class (grade-aware), using the numeric part of Level
    out["_g"] = out["Level"].str.extract(r"(\d+)", expand=False).astype(float).fillna(9999)
    out = out.sort_values(["_g","Class"], kind="stable").drop(columns="_g").reset_index(drop=True)
    return out


def collect_level_lessons_from_zip(zip_bytes: bytes,
                                   mapping_df: pd.DataFrame | None = None,
                                   logins_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    frames, names = [], []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv") and "level lessons usage" in name.lower():
                frames.append(read_csv_flex_from_bytes(zf.read(name)))
                names.append(name)
    return build_levelwise_from_frames(frames, mapping_df=mapping_df, logins_df=logins_df), names



def filter_active_grades(su: pd.DataFrame, min_students: int = 1, min_activity: int = 0) -> pd.DataFrame:
    activity_cols = ["No of Logins", "No of Lessons Accessed", "Quiz", "Worksheet", "Prasso", "Reading"]
    su = su.copy()
    su["_sum_activity"] = su[activity_cols].sum(axis=1)
    out = su[(su["No of Students"] >= min_students) | (su["_sum_activity"] > min_activity)]
    return out.drop(columns=["_sum_activity"])


# --------------------------- UI ---------------------------

st.title("HeyMath — Students/Teachers Usage Builder")

tab_zip, tab_csv = st.tabs(["Upload ZIP (recommended)", "Upload CSVs"])

# --- ZIP flow ---
with tab_zip:
    zip_file = st.file_uploader("Drop a HeyMath ZIP here", type=["zip"])
    
    if zip_file is not None:
        zip_bytes = zip_file.read()   # <-- add this
        a_df, l_df, g_df, t_df, names = detect_files_in_zip(zip_bytes, mapping_df=MAPPING_DF)  # use zip_bytes here

        if any(x is None for x in (a_df, l_df, g_df)):
            st.error("Could not auto-detect Assignments/Lessons/Logins from the ZIP. Try the CSV tab.")
        else:
            st.caption(
                f"Detected: Assignments = `{names['assign']}`, Lessons = `{names['lessons']}`, "
                f"Logins = `{names['logins']}`, Teachers = `{names['teachers']}`"
            )

            with st.form("zip_form"):
                st.subheader("Options")
                mode = st.radio("Grade selection", ["Active (default)", "All", "Whitelist"], index=0, horizontal=True)
                min_students = st.number_input("Active: minimum students", min_value=0, value=1, step=1)
                min_activity = st.number_input("Active: minimum total activity", min_value=0, value=0, step=1)
                whitelist = st.text_input('Whitelist grades (comma-separated, e.g. "Grade 1,Grade 2")', value="")
                submitted = st.form_submit_button("Build")

            if submitted:
                # Build SU
                su = build_su(a_df, l_df, g_df, mapping_df=MAPPING_DF)
                if mode.startswith("Active"):
                    su = filter_active_grades(su, min_students=min_students, min_activity=min_activity)
                elif mode == "Whitelist":
                    allow = {g.strip().lower() for g in whitelist.split(",") if g.strip()}
                    if allow:
                        su = su[su["Class"].str.lower().isin(allow)]

                # Build TU
                tu = build_tu(t_df, mapping_df=MAPPING_DF)

                # Build Levelwise from the ZIP
                levelwise_df, level_names = collect_level_lessons_from_zip(zip_bytes, mapping_df=MAPPING_DF, logins_df=g_df)

                # Previews
                tab_su, tab_tu, tab_lv = st.tabs(["SU preview", "TU preview", "Levelwise preview"])
                with tab_su: st.dataframe(su.reset_index(drop=True), use_container_width=True)
                with tab_tu: st.dataframe(tu.reset_index(drop=True), use_container_width=True)
                with tab_lv:
                    if level_names:
                        st.caption("Merged from:\n- " + "\n- ".join(level_names))
                    st.dataframe(levelwise_df.reset_index(drop=True), use_container_width=True)

                # Downloads – one set, with unique keys
                xbuf = io.BytesIO()
                with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                    su.to_excel(w, sheet_name="SU", index=False)
                    tu.to_excel(w, sheet_name="TU", index=False)
                    if not levelwise_df.empty:
                        levelwise_df.to_excel(w, sheet_name="Levelwise", index=False)
                st.download_button("Download XLSX (SU + TU + Levelwise)", data=xbuf.getvalue(),
                                   file_name="SchoolReport_FINAL.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="zip_xlsx")
                st.download_button("Download SU.csv", data=su.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="SchoolReport_SU_FINAL.csv", mime="text/csv", key="zip_su_csv")
                st.download_button("Download TU.csv", data=tu.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="SchoolReport_TU_FINAL.csv", mime="text/csv", key="zip_tu_csv")
                if not levelwise_df.empty:
                    st.download_button("Download Levelwise.csv",
                                       data=levelwise_df.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="SchoolReport_Levelwise_FINAL.csv", mime="text/csv",
                                       key="zip_levelwise_csv")


               
                # # Single workbook: SU + TU (+ Levelwise if present)
                # xbuf = io.BytesIO()
                # with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                    # su.to_excel(w, sheet_name="SU", index=False)
                    # tu.to_excel(w, sheet_name="TU", index=False)
                    # if not levelwise_df.empty:
                        # levelwise_df.to_excel(w, sheet_name="Levelwise", index=False)
                # st.download_button("Download XLSX (SU + TU + Levelwise)", data=xbuf.getvalue(),
                                   # file_name="SchoolReport_FINAL.xlsx",
                                   # mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # # Optional separate CSVs
                # st.download_button("Download SU.csv", data=su.to_csv(index=False).encode("utf-8-sig"),
                                   # file_name="SchoolReport_SU_FINAL.csv", mime="text/csv")
                # st.download_button("Download TU.csv", data=tu.to_csv(index=False).encode("utf-8-sig"),
                                   # file_name="SchoolReport_TU_FINAL.csv", mime="text/csv")



# --- CSV flow ---
with tab_csv:
    a_up = st.file_uploader("Assignments CSV", type=["csv"], key="a")
    l_up = st.file_uploader("Lessons CSV", type=["csv"], key="l")
    g_up = st.file_uploader("Logins CSV", type=["csv"], key="g")
    t_up = st.file_uploader("Teachers Usage CSV (optional, for TU)", type=["csv"], key="t")
    lvl_up = st.file_uploader("Level Lessons Usage CSVs (multiple, optional — for Levelwise)",
                              type=["csv"], accept_multiple_files=True, key="lvl")

    if a_up and l_up and g_up:
        a_df = read_csv_flex_from_bytes(a_up.read())
        l_df = read_csv_flex_from_bytes(l_up.read())
        g_df = read_csv_flex_from_bytes(g_up.read())
        t_df = read_csv_flex_from_bytes(t_up.read()) if t_up else None

        with st.form("csv_form"):
            st.subheader("Options")
            mode = st.radio("Grade selection", ["Active (default)", "All", "Whitelist"], index=0, horizontal=True)
            min_students = st.number_input("Active: minimum students", min_value=0, value=1, step=1, key="ms2")
            min_activity = st.number_input("Active: minimum total activity", min_value=0, value=0, step=1, key="ma2")
            whitelist = st.text_input('Whitelist grades (comma-separated, e.g. "Grade 1,Grade 2")',
                                      value="", key="wl2")
            submitted = st.form_submit_button("Build")

        if submitted:
            # Build SU
            su = build_su(a_df, l_df, g_df, mapping_df=MAPPING_DF)
            if mode.startswith("Active"):
                su = filter_active_grades(su, min_students=min_students, min_activity=min_activity)
            elif mode == "Whitelist":
                allow = {g.strip().lower() for g in whitelist.split(",") if g.strip()}
                if allow:
                    su = su[su["Class"].str.lower().isin(allow)]

            # Build TU
            tu = build_tu(t_df, mapping_df=MAPPING_DF)

            # Build Levelwise from the uploaded Level Lessons files (if any)
            lvl_frames = []
            if lvl_up:
                for f in lvl_up:
                    lvl_frames.append(read_csv_flex_from_bytes(f.read()))
            levelwise_df = build_levelwise_from_frames(lvl_frames, mapping_df=MAPPING_DF, logins_df=g_df)

            # Previews
            tab_su, tab_tu, tab_lv = st.tabs(["SU preview", "TU preview", "Levelwise preview"])
            with tab_su: st.dataframe(su.reset_index(drop=True), use_container_width=True)
            with tab_tu: st.dataframe(tu.reset_index(drop=True), use_container_width=True)
            with tab_lv: st.dataframe(levelwise_df.reset_index(drop=True), use_container_width=True)

            # Downloads – one set, with unique keys
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                su.to_excel(w, sheet_name="SU", index=False)
                tu.to_excel(w, sheet_name="TU", index=False)
                if not levelwise_df.empty:
                    levelwise_df.to_excel(w, sheet_name="Levelwise", index=False)
            st.download_button("Download XLSX (SU + TU + Levelwise)", data=xbuf.getvalue(),
                               file_name="SchoolReport_FINAL.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="csv_xlsx")
            st.download_button("Download SU.csv", data=su.to_csv(index=False).encode("utf-8-sig"),
                               file_name="SchoolReport_SU_FINAL.csv", mime="text/csv", key="csv_su_csv")
            st.download_button("Download TU.csv", data=tu.to_csv(index=False).encode("utf-8-sig"),
                               file_name="SchoolReport_TU_FINAL.csv", mime="text/csv", key="csv_tu_csv")
            if not levelwise_df.empty:
                st.download_button("Download Levelwise.csv",
                                   data=levelwise_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="SchoolReport_Levelwise_FINAL.csv", mime="text/csv",
                                   key="csv_levelwise_csv")

            
            # tab_su, tab_tu, tab_lv = st.tabs(["SU preview", "TU preview", "Levelwise preview"])
            # with tab_su:
                # st.dataframe(su, use_container_width=True)
            # with tab_tu:
                # st.dataframe(tu, use_container_width=True)
            # with tab_lv:
                # if level_names:
                    # st.caption("Merged from:\n- " + "\n- ".join(level_names))
                # st.dataframe(levelwise_df, use_container_width=True)

            # # One workbook: SU + TU + Levelwise
            # xbuf = io.BytesIO()
            # with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                # su.to_excel(w, sheet_name="SU", index=False)
                # tu.to_excel(w, sheet_name="TU", index=False)
                # if not levelwise_df.empty:
                    # levelwise_df.to_excel(w, sheet_name="Levelwise", index=False)
            # st.download_button("Download XLSX (SU + TU + Levelwise)", data=xbuf.getvalue(),
                               # file_name="SchoolReport_FINAL.xlsx",
                               # mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")