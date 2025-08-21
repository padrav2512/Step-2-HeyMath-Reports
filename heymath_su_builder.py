# heymath_su_builder.py
# ---------------------------------------------------------------------
# HeyMath! School Report Builder – compact UI, robust charts, ASR & TU
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import zipfile, io, csv, re, os
import altair as alt

# ------------------------ Page & theme -------------------------------
st.set_page_config(
    page_title="HeyMath! School Report Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.data_transformers.disable_max_rows()  # allow large frames

st.markdown("""
<style>
/* Narrow page & reduce top padding */
.block-container {max-width: 1100px; padding-top: .6rem; padding-bottom: 1rem;}
/* Tighter headings */
h1, h2, h3 {margin-top: .2rem;}
/* Compact radio spacing */
div.row-widget.stRadio > div {gap: .75rem}
/* Strong primary button */
button[kind="primary"]{font-weight:600;padding:.6rem 1rem;border-radius:.6rem}
/* Center tables without going full width */
.hm-narrow {max-width: 900px; margin: 0 auto;}
</style>
""", unsafe_allow_html=True)

# ------------------------ Utilities ----------------------------------
def norm(s): return re.sub(r"\s+"," ", str(s or "")).strip().lower()
def normalize_class(s): return norm(s)

def pick_col(df: pd.DataFrame, *cands):
    for c in cands:
        if c in df.columns: return c
    low = {str(c).strip().lower(): c for c in df.columns}
    for c in cands:
        k = str(c).strip().lower()
        if k in low: return low[k]
    return None

def normalize_is_hold(series: pd.Series) -> pd.Series:
    """Normalize isHold-like flags to 0/1, accepting many variants."""
    n = pd.to_numeric(series, errors="coerce")
    if n.notna().any(): return n.fillna(0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    true_vals  = {"1","true","t","yes","y","on","hold","h"}
    false_vals = {"0","false","f","no","n","off","release","released","active"}
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out = out.mask(s.isin(false_vals), 0).mask(s.isin(true_vals), 1)
    return out.fillna(0).astype(int)

# natural class sorting helpers (Grade/Standard 1A…10D, KG/LKG/UKG etc.)
_WORD = {"nursery":-3,"pre-nursery":-3,"prenursery":-3,"lkg":-2,"kg1":-2,"pp1":-2,"ukg":-1,"kg2":-1,"pp2":-1,"kg":-1,"prep":0}
def class_sort_key(label: str):
    s = norm(label)
    if not s: return (9999,"")
    f = s.split()[0]
    if f in _WORD:
        m = re.search(r"([a-z])$", s)
        return (_WORD[f], (m.group(1).upper() if m else ""))
    m = re.search(r"(\d+)\s*([a-z]{0,2})$", s) or re.search(r"(\d+)", s)
    return ((int(m.group(1)), (m.group(2) or "").upper()) if m else (9999, s))

def add_class_sort(df, class_col="Class", out_col="_sort"):
    d=df.copy()
    def to_num(t):
        g,s=t
        sec=(ord(s[0])-64) if isinstance(s,str) and len(s)==1 and "A"<=s<="Z" else 0
        return float(g)*100+sec
    d[out_col]=d[class_col].apply(class_sort_key).apply(to_num)
    return d

# ------------------------ Chart helpers ------------------------------
def render_altair(chart, tag="chart"):
    """Always render the chart or show the exact error from Altair."""
    try:
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.warning(f"{tag}: Altair failed ({type(e).__name__}). Error below; showing no chart.")
        st.exception(e)

def bar_with_labels(df, x, y, title=None, horizontal=False, height=320, width=900,
                    category_sort=None, bar_size=None):
    """Altair bar chart with integer axis & value labels (Altair v5 safe).
       category_sort: None = sort by value; otherwise name of a column to sort by.
       bar_size: fixed pixel width for bars (useful when few categories)."""
    df = df.copy()
    if x not in df.columns or y not in df.columns:
        return alt.Chart(pd.DataFrame({"msg":["No data"]})).mark_text().encode(text="msg:N")
    df = df[df[x].notna()]
    df[y] = pd.to_numeric(df[y], errors="coerce").fillna(0)

    sort_obj = alt.SortField(field=category_sort, order="ascending") if category_sort else ('-x' if horizontal else '-y')
    axis_q = alt.Axis(tickMinStep=1, format=".0f")
    base = alt.Chart(df).properties(width=width, height=height)
    if title: base = base.properties(title=title)

    # bar thickness (optional)
    mark_kwargs = {}
    if bar_size is not None:
        mark_kwargs["size"] = bar_size

    if horizontal:
        bars = base.mark_bar(**mark_kwargs).encode(
            y=alt.Y(f"{x}:N", sort=sort_obj),
            x=alt.X(f"{y}:Q", axis=axis_q),
            tooltip=[x, y],
        )
        text = base.mark_text(align="left", dx=3).encode(
            y=alt.Y(f"{x}:N", sort=sort_obj),
            x=alt.X(f"{y}:Q"),
            text=alt.Text(f"{y}:Q", format=".0f"),
        )
    else:
        bars = base.mark_bar(**mark_kwargs).encode(
            x=alt.X(f"{x}:N", sort=sort_obj),
            y=alt.Y(f"{y}:Q", axis=axis_q),
            tooltip=[x, y],
        )
        text = base.mark_text(align="center", dy=-5).encode(
            x=alt.X(f"{x}:N", sort=sort_obj),
            y=alt.Y(f"{y}:Q"),
            text=alt.Text(f"{y}:Q", format=".0f"),
        )
    return (bars + text).configure_axis(labelLimit=160, grid=True, gridColor="#f2f2f2")

def center_table(df: pd.DataFrame, height=420, key=None):
    st.dataframe(df.reset_index(drop=True), height=height, use_container_width=True, key=key)

def compact_options_form(prefix_key="csv", default_mode="Active (default)"):
    cols = st.columns([3,1])
    with cols[0]:
        mode = st.radio("Grade selection",
                        ["Active (default)", "All", "Whitelist"],
                        index=["Active (default)","All","Whitelist"].index(default_mode),
                        horizontal=True, key=f"{prefix_key}_mode")
    with cols[1]:
        submitted = st.button("Build", type="primary", use_container_width=True, key=f"{prefix_key}_build_btn")
    with st.expander("Advanced (rarely used)", expanded=False):
        min_students = st.number_input("Active: minimum students", min_value=0, value=1, step=1, key=f"{prefix_key}_min_stu")
        min_activity = st.number_input("Active: minimum total activity", min_value=0, value=0, step=1, key=f"{prefix_key}_min_act")
        whitelist = st.text_input('Whitelist grades (comma-separated, e.g. "Grade 1,Grade 2")', value="", key=f"{prefix_key}_wl")
    return submitted, mode, min_students, min_activity, whitelist

# ------------------------ CSV reader ---------------------------------
def read_csv_flex_from_bytes(b: bytes) -> pd.DataFrame:
    if b[:2]==b"PK":  # mislabelled XLSX
        try: return pd.read_excel(io.BytesIO(b), engine="openpyxl")
        except Exception: pass
    text=None
    for enc in ("utf-8-sig","utf-8","cp1252","latin-1"):
        try: text=b.decode(enc); break
        except Exception: continue
    if text is None: text=b.decode("utf-8", errors="replace")
    if not text.strip(): return pd.DataFrame()
    try:
        dialect=csv.Sniffer().sniff(text, delimiters=[",",";","\t","|"]); sep=dialect.delimiter
    except Exception:
        first=next((ln for ln in text.splitlines() if ln.strip()), ",")
        sep=max([",",";","\t","|"], key=lambda d:first.count(d))
    try: df=pd.read_csv(io.StringIO(text), sep=sep, dtype=str, engine="python")
    except pd.errors.EmptyDataError: return pd.DataFrame()
    df=df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    return df

# ------------------------ ZIP detection ------------------------------
def detect_files_in_zip(zip_bytes: bytes):
    assign_df = lessons_df = logins_df = teachers_df = None
    names = {"assign":None,"lessons":None,"logins":None,"teachers":None}
    with zipfile.ZipFile(io.BytesIO(zip_bytes),"r") as zf:
        csv_names=[n for n in zf.namelist() if n.lower().endswith(".csv")]
        read_cache={}
        for name in csv_names:
            try: read_cache[name]=read_csv_flex_from_bytes(zf.read(name))
            except Exception: continue

        # logins
        for name,df in read_cache.items():
            low=norm(name); cols=[norm(c) for c in df.columns]
            if ("logins report" in low) or ("total_students" in cols) or ("total_logins" in cols):
                logins_df, names["logins"] = df, name; break

        # assignments
        school_assign=[]; level_assign=[]; other=[]
        for name,df in read_cache.items():
            low=norm(name); cols=[norm(c) for c in df.columns]
            looks = ("assign" in low) or any(k in " ".join(cols) for k in ["ongoing_aqc","ongoing_worksheet","ongoing_prasso","ongoing_reading"])
            if not looks: continue
            if "school assignments usage" in low: school_assign.append((name,df))
            elif "assignments usage" in low: level_assign.append((name,df))
            else: other.append((name,df))
        if school_assign:
            assign_df, names["assign"]=school_assign[0][1], school_assign[0][0]
        elif level_assign:
            parts=[]
            for _,df in level_assign:
                lvl = next((c for c in df.columns if "level" in norm(c)), None)
                mets= [c for c in df.columns if any(k in norm(c) for k in ["ongoing_aqc","ongoing_worksheet","ongoing_prasso","ongoing_reading"])]
                if not (lvl and mets): continue
                d=df[[lvl]+mets].copy()
                for c in mets: d[c]=pd.to_numeric(d[c], errors="coerce").fillna(0)
                g=d.groupby(lvl, as_index=False).sum()
                ren={lvl:"LEVEL"}
                for m in mets:
                    lm=norm(m)
                    ren[m] = ("ONGOING_AQC" if "ongoing_aqc" in lm else
                              "ONGOING_WORKSHEET" if "ongoing_worksheet" in lm else
                              "ONGOING_PRASSO" if "ongoing_prasso" in lm else
                              "ONGOING_READING")
                parts.append(g.rename(columns=ren))
            if parts:
                agg=parts[0]
                for p in parts[1:]: agg=agg.merge(p, on="LEVEL", how="outer")
                for c in ["ONGOING_AQC","ONGOING_WORKSHEET","ONGOING_PRASSO","ONGOING_READING"]:
                    if c in agg.columns: agg[c]=pd.to_numeric(agg[c], errors="coerce").fillna(0)
                assign_df=agg.groupby("LEVEL", as_index=False).sum(numeric_only=True)
                names["assign"]=f"Aggregated {len(parts)} Assignments files"
        elif other:
            assign_df, names["assign"]=other[0][1], other[0][0]

        # lessons
        school_lessons=[]; level_lessons=[]
        for name,df in read_cache.items():
            low=norm(name); cols=[norm(c) for c in df.columns]
            if not any("lesson" in c for c in cols): continue
            if "school lessons usage" in low: school_lessons.append((name,df))
            elif "level lessons usage" in low: level_lessons.append((name,df))
        if school_lessons:
            lessons_df, names["lessons"]=school_lessons[0][1], school_lessons[0][0]
        elif level_lessons:
            parts=[]
            for _,df in level_lessons:
                lvl=next((c for c in df.columns if "level" in norm(c)), None)
                les=next((c for c in df.columns if "lesson" in norm(c)), None)
                if not (lvl and les): continue
                d=df[[lvl,les]].copy()
                d[les]=pd.to_numeric(d[les], errors="coerce").fillna(0)
                d=d.groupby(lvl, as_index=False)[les].sum()
                d.columns=["LEVEL","LESSONS_ACCESSED"]
                parts.append(d)
            if parts:
                agg=pd.concat(parts, ignore_index=True).groupby("LEVEL", as_index=False)["LESSONS_ACCESSED"].sum()
                lessons_df, names["lessons"]=agg, f"Aggregated {len(parts)} Level Lessons files"

        # teachers (prefer files with totals)
        if teachers_df is None:
            candidates=[]
            for name,df in read_cache.items():
                low=norm(name); cols={norm(c) for c in df.columns}
                looks = ("teachers usage" in low) or ("all teachers usage" in low) or (("teacher" in low or "user" in low) and ("usage" in low))
                if looks:
                    score=0
                    if {"no of logins","total logins","logins"} & cols: score+=2
                    if {"no of lessons accessed","lessons accessed"} & cols: score+=1
                    if {"no of assignments assigned","assignments assigned"} & cols: score+=1
                    candidates.append((score, name, df))
            if not candidates:
                for name,df in read_cache.items():
                    cols={norm(c) for c in df.columns}
                    if any("teacher" in c or "user" in c for c in cols) and any("login" in c for c in cols):
                        candidates.append((0, name, df))
            if candidates:
                candidates.sort(key=lambda t:(-t[0], len(t[1])))
                _, name, df = candidates[0]
                teachers_df, names["teachers"]=df, name

    return assign_df, lessons_df, logins_df, teachers_df, names

# ------------------------ Builders -----------------------------------
def build_su(a_df, l_df, g_df):
    # numerics
    for c in ["TOTAL_STUDENTS","TOTAL_LOGINS"]:
        if c in g_df.columns: g_df[c]=pd.to_numeric(g_df[c], errors="coerce").fillna(0).astype(int)
    if "LESSONS_ACCESSED" in l_df.columns:
        l_df["LESSONS_ACCESSED"]=pd.to_numeric(l_df["LESSONS_ACCESSED"], errors="coerce").fillna(0).astype(int)
    for c in ["ONGOING_AQC","ONGOING_WORKSHEET","ONGOING_PRASSO","ONGOING_READING"]:
        if c in a_df.columns: a_df[c]=pd.to_numeric(a_df[c], errors="coerce").fillna(0).astype(int)

    need=["LEVEL","LEVEL_DISPLAY_NAME","SORT_ORDER","TOTAL_STUDENTS","TOTAL_LOGINS"]
    miss=[c for c in need if c not in g_df.columns]
    if miss: raise ValueError(f"Logins CSV missing required columns: {miss}")

    base=g_df[["LEVEL","LEVEL_DISPLAY_NAME","SORT_ORDER","TOTAL_STUDENTS","TOTAL_LOGINS"]].copy()
    base=base.rename(columns={"LEVEL_DISPLAY_NAME":"Class","TOTAL_STUDENTS":"No of Students","TOTAL_LOGINS":"No of Logins"})

    less=pd.DataFrame(columns=["LEVEL","No of Lessons Accessed"])
    if {"LEVEL","LESSONS_ACCESSED"}.issubset(l_df.columns):
        less=(l_df[["LEVEL","LESSONS_ACCESSED"]]
              .groupby("LEVEL", as_index=False)["LESSONS_ACCESSED"].sum()
              .rename(columns={"LESSONS_ACCESSED":"No of Lessons Accessed"}))

    if "LEVEL" not in a_df.columns:
        for c in a_df.columns:
            if "level" in norm(c): a_df=a_df.rename(columns={c:"LEVEL"}); break
    assign=pd.DataFrame(columns=["LEVEL","Quiz","Worksheet","Prasso","Reading"])
    if "LEVEL" in a_df.columns:
        keep, mapping=["LEVEL"],{}
        if "ONGOING_AQC" in a_df.columns: mapping["ONGOING_AQC"]="Quiz"; keep.append("ONGOING_AQC")
        if "ONGOING_WORKSHEET" in a_df.columns: mapping["ONGOING_WORKSHEET"]="Worksheet"; keep.append("ONGOING_WORKSHEET")
        if "ONGOING_PRASSO" in a_df.columns: mapping["ONGOING_PRASSO"]="Prasso"; keep.append("ONGOING_PRASSO")
        if "ONGOING_READING" in a_df.columns: mapping["ONGOING_READING"]="Reading"; keep.append("ONGOING_READING")
        tmp=a_df[keep].copy()
        for raw in ["ONGOING_AQC","ONGOING_WORKSHEET","ONGOING_PRASSO","ONGOING_READING"]:
            if raw in tmp.columns: tmp[raw]=pd.to_numeric(tmp[raw], errors="coerce").fillna(0)
        tmp=tmp.groupby("LEVEL", as_index=False).sum(numeric_only=True)
        assign=tmp.rename(columns=mapping)
        for col in ["Quiz","Worksheet","Prasso","Reading"]:
            if col not in assign.columns: assign[col]=0
        assign=assign[["LEVEL","Quiz","Worksheet","Prasso","Reading"]]

    su = base.merge(less, on="LEVEL", how="left").merge(assign, on="LEVEL", how="left")
    for c in ["No of Students","No of Logins","No of Lessons Accessed","Quiz","Worksheet","Prasso","Reading"]:
        if c in su.columns: su[c]=pd.to_numeric(su[c], errors="coerce").fillna(0).astype(int)

    if "SORT_ORDER" in su.columns: su=su.sort_values(["SORT_ORDER","Class"], kind="stable")
    else: su=add_class_sort(su,"Class").sort_values(["_sort","Class"], kind="stable").drop(columns="_sort")
    cols=["Class","No of Students","No of Logins","No of Lessons Accessed","Quiz","Worksheet","Prasso","Reading"]
    for c in cols:
        if c not in su.columns: su[c]=0
    return su[cols]

def build_tu(teachers_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    TU builder that works with totals OR row-per-event files.
    Falls back to counting rows when no numeric totals exist.
    """
    if teachers_df is None or teachers_df.empty:
        return pd.DataFrame(columns=["Name","No of logins","No of Lessons Accessed","No of Assignments Assigned"])

    df = teachers_df.copy()

    # case-insensitive picker
    def pick(*alts):
        for a in alts:
            if a in df.columns: return a
        low={c.lower():c for c in df.columns}
        for a in alts:
            if a.lower() in low: return low[a.lower()]
        return None

    name_col    = pick("Name","userName","username","Teacher","Teacher Name","User")
    logins_col  = pick("No of logins","Total Logins","logins","Login Count","Logins")
    lessons_col = pick("No of Lessons Accessed","Lessons Accessed","lesson","Lessons")
    assigns_col = pick("No of Assignments Assigned","Assignments Assigned","assessment","Assignments")

    if name_col is None:
        return pd.DataFrame(columns=["Name","No of logins","No of Lessons Accessed","No of Assignments Assigned"])

    def to_num(s):
        return pd.to_numeric(pd.Series(s, copy=False).astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                             errors="coerce").fillna(0)

    cols=[c for c in [name_col,logins_col,lessons_col,assigns_col] if c]
    tmp=df[cols].copy() if cols else df[[name_col]].copy()

    totals={}
    for c in [logins_col,lessons_col,assigns_col]:
        if c and c in tmp.columns:
            tmp[c]=to_num(tmp[c]); totals[c]=tmp[c].sum()

    direct = tmp.groupby(name_col, as_index=False).sum(numeric_only=True)

    # Fallbacks by counting rows if totals look empty
    need_fb_logins  = (logins_col is None)  or (totals.get(logins_col,0)==0)
    need_fb_lessons = (lessons_col is None) or (totals.get(lessons_col,0)==0)
    need_fb_assigns = (assigns_col is None) or (totals.get(assigns_col,0)==0)

    if need_fb_logins:
        login_like = pick("Login Timestamp","Login Time","Login Date","Last Login","Login")
        if login_like:
            counts = df.groupby(name_col)[login_like].count().rename("No of logins").reset_index()
            direct = direct.merge(counts, on=name_col, how="outer")
        else:
            direct["No of logins"] = direct.get(logins_col, 0)

    if need_fb_lessons:
        lesson_col = pick("Lesson Title","Lesson Id","LessonID","Lesson")
        if lesson_col:
            counts = df.groupby(name_col)[lesson_col].count().rename("No of Lessons Accessed").reset_index()
            direct = direct.merge(counts, on=name_col, how="outer")
        else:
            type_col = pick("Type","Activity Type","Resource Type")
            if type_col:
                counts = df[df[type_col].astype(str).str.contains("lesson", case=False, na=False)] \
                           .groupby(name_col)[type_col].count().rename("No of Lessons Accessed").reset_index()
                direct = direct.merge(counts, on=name_col, how="outer")

    if need_fb_assigns:
        assign_col = pick("Assignment Name","Assignment Id","Assessment Name","Assessment Id","Assessment")
        if assign_col:
            counts = df.groupby(name_col)[assign_col].count().rename("No of Assignments Assigned").reset_index()
            direct = direct.merge(counts, on=name_col, how="outer")
        else:
            type_col = pick("Type","Activity Type","Resource Type")
            if type_col:
                counts = df[df[type_col].astype(str).str.contains("assign|assess", case=False, na=False)] \
                           .groupby(name_col)[type_col].count().rename("No of Assignments Assigned").reset_index()
                direct = direct.merge(counts, on=name_col, how="outer")

    ren={}
    if name_col!="Name": ren[name_col]="Name"
    if logins_col and "No of logins" not in direct.columns: ren[logins_col]="No of logins"
    if lessons_col and "No of Lessons Accessed" not in direct.columns: ren[lessons_col]="No of Lessons Accessed"
    if assigns_col and "No of Assignments Assigned" not in direct.columns: ren[assigns_col]="No of Assignments Assigned"
    direct=direct.rename(columns=ren)

    for c in ["No of logins","No of Lessons Accessed","No of Assignments Assigned"]:
        if c not in direct.columns: direct[c]=0
        direct[c]=to_num(direct[c]).astype(int)

    return direct[["Name","No of logins","No of Lessons Accessed","No of Assignments Assigned"]].sort_values("Name", kind="stable")

def build_levelwise_from_frames(frames, logins_df=None):
    if not frames: return pd.DataFrame(columns=["Level","Class","No of Lessons Accessed"])
    cleaned=[]
    for df in frames:
        d=df.copy()
        if "LEVEL" not in d.columns:
            for c in d.columns:
                if "level" in norm(c): d=d.rename(columns={c:"LEVEL"}); break
        if "LEVEL_DISPLAY_NAME" not in d.columns:
            for c in d.columns:
                if "level_display_name" in norm(c) or "class" in norm(c): d=d.rename(columns={c:"LEVEL_DISPLAY_NAME"}); break
        if "LESSONS_ACCESSED" not in d.columns:
            for c in d.columns:
                lc=norm(c)
                if ("lesson" in lc) and ("access" in lc): d=d.rename(columns={c:"LESSONS_ACCESSED"}); break
        if "LESSONS_ACCESSED" in d.columns: d["LESSONS_ACCESSED"]=pd.to_numeric(d["LESSONS_ACCESSED"], errors="coerce").fillna(0).astype(int)
        cleaned.append(d[[c for c in ["LEVEL","LEVEL_DISPLAY_NAME","LESSONS_ACCESSED"] if c in d.columns]].copy())
    out=pd.concat(cleaned, ignore_index=True, sort=False)
    if "LEVEL_DISPLAY_NAME" not in out.columns and logins_df is not None and {"LEVEL","LEVEL_DISPLAY_NAME"}.issubset(logins_df.columns):
        out=out.merge(logins_df[["LEVEL","LEVEL_DISPLAY_NAME"]].drop_duplicates(), on="LEVEL", how="left")
    for c in ["LEVEL","LEVEL_DISPLAY_NAME","LESSONS_ACCESSED"]:
        if c not in out.columns: out[c] = 0 if c=="LESSONS_ACCESSED" else ""
    out = out.rename(columns={"LEVEL":"Level","LEVEL_DISPLAY_NAME":"Class","LESSONS_ACCESSED":"No of Lessons Accessed"})
    def to_grade(s):
        s=str(s); m=re.search(r'(\d+)', s)
        return f"Grade {int(m.group(1))}" if m else s
    out["Level"]=out["Class"].apply(to_grade)
    out=add_class_sort(out,"Class").sort_values(["_sort","Class"], kind="stable").drop(columns="_sort").reset_index(drop=True)
    return out

def filter_active_grades(su, min_students=1, min_activity=0):
    act_cols=["No of Logins","No of Lessons Accessed","Quiz","Worksheet","Prasso","Reading"]
    su=su.copy(); su["_sum"]=su[act_cols].sum(axis=1)
    return su[(su["No of Students"]>=min_students) | (su["_sum"]>min_activity)].drop(columns="_sum")

# ------------------------ ASR builder --------------------------------
def build_asr_quiz_split(school_assign_df: pd.DataFrame, teacher_quiz_by_class: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lv_col   = pick_col(school_assign_df, "LEVEL_DISPLAY_NAME","Class","CLASS","Level Display Name")
    aqc_col  = pick_col(school_assign_df, "ONGOING_AQC","Ongoing AQC","AQC","ONGOING_AQ")
    hold_col = pick_col(school_assign_df, "isHold","Is Hold","IS_HOLD","is_hold","Hold")
    if not lv_col or not aqc_col:
        raise ValueError(f"School Assignments Usage missing LEVEL_DISPLAY_NAME/ONGOING_AQC. Got: {list(school_assign_df.columns)}")

    d = school_assign_df.copy()
    if hold_col and hold_col in d.columns:
        d = d.loc[normalize_is_hold(d[hold_col]) == 0].copy()

    classes = d[lv_col].dropna().astype(str).map(lambda s: re.sub(r"\s+"," ", s).strip()).unique()
    rows=[]
    for klass in classes:
        key=normalize_class(klass)
        tqa = teacher_quiz_by_class.get(key)

        cap=sp=qt=0
        if tqa is not None and not tqa.empty:
            hold2 = pick_col(tqa, "isHold","Is Hold","IS_HOLD","is_hold","Hold")
            if hold2: tqa = tqa.loc[normalize_is_hold(tqa[hold2]) == 0].copy()
            am_col = pick_col(tqa, "assessmentMode","Assessment Mode","mode")
            if am_col:
                mm = pd.to_numeric(tqa[am_col], errors="coerce")
                cap = int((mm==3).sum()); sp=int((mm==2).sum()); qt=int((mm==1).sum())

        total = cap+sp+qt
        ongoing = (pd.to_numeric(d.loc[d[lv_col].astype(str).str.strip()==klass, aqc_col], errors="coerce")
                    .fillna(0).astype(int).sum())
        rows.append({
            "Class": klass,
            "Quiz Adaptive Practice": cap,
            "Quiz Standard Practice": sp,
            "Quiz Test": qt,
            "Total (Quiz)": total,
            "ONGOING_AQC": int(ongoing),
            "AQC Diff (Total-ONGOING_AQC)": int(total-ongoing),
            "AQC Match?": "Yes" if total==int(ongoing) else "No",
        })
    out=pd.DataFrame(rows)
    if out.empty: return out
    for c in ["Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test","Total (Quiz)","ONGOING_AQC","AQC Diff (Total-ONGOING_AQC)"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    out = add_class_sort(out,"Class").sort_values(["_sort","Class"], kind="stable").drop(columns="_sort").reset_index(drop=True)
    return out

def load_school_assignments_from_zip(zip_bytes: bytes) -> pd.DataFrame | None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes),"r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv") and "school assignments usage" in name.lower():
                return read_csv_flex_from_bytes(zf.read(name))
    return None

def load_teacher_quiz_by_class_from_zip(zip_bytes: bytes) -> dict[str, pd.DataFrame]:
    out={}
    pat_old = re.compile(r"teacher\s*quiz\s*assignment.*?(for|-\s*)\s*(.+?)\.csv$", re.I)
    with zipfile.ZipFile(io.BytesIO(zip_bytes),"r") as zf:
        for name in zf.namelist():
            low=name.lower()
            if not low.endswith(".csv"): continue
            if "teacher" in low and "quiz" in low and "assignment" in low:
                m=re.match(r"TeacherQuizAssignment_(.+?)_", name)
                if m: klass=m.group(1).strip()
                else:
                    m2=pat_old.search(name)
                    klass = m2.group(2).strip() if m2 else re.sub(r"\.csv$","",name,flags=re.I)
                out[normalize_class(klass)]=read_csv_flex_from_bytes(zf.read(name))
    return out

# ------------------------ UI -----------------------------------------
st.title("HeyMath! School Report Builder")
tab_zip, tab_csv = st.tabs(["Upload ZIP (recommended)","Upload CSVs"])

# ================= ZIP FLOW =================
with tab_zip:
    su = st.session_state.get("zip_su", pd.DataFrame())
    tu = st.session_state.get("zip_tu", pd.DataFrame())
    lv = st.session_state.get("zip_lv", pd.DataFrame())
    asr = st.session_state.get("zip_asr", pd.DataFrame())

    up = st.file_uploader("Drop a HeyMath ZIP here", type=["zip"])
    if up is not None:
        zip_bytes = up.read()
        a_df, l_df, g_df, t_df, names = detect_files_in_zip(zip_bytes)
        if any(x is None for x in (a_df,l_df,g_df)):
            st.error("Could not auto-detect Assignments/Lessons/Logins from the ZIP.")
        else:
            st.caption(f"Detected: Assignments=`{names['assign']}`, Lessons=`{names['lessons']}`, Logins=`{names['logins']}`, Teachers=`{names['teachers']}`")
            submitted, mode, min_students, min_activity, whitelist = compact_options_form(prefix_key="zip")
            if submitted:
                su = build_su(a_df,l_df,g_df)
                if mode.startswith("Active"): su=filter_active_grades(su, min_students, min_activity)
                elif mode=="Whitelist":
                    allow={g.strip().lower() for g in whitelist.split(",") if g.strip()}
                    if allow: su=su[su["Class"].str.lower().isin(allow)]
                tu = build_tu(t_df)
                # Levelwise (from any Level Lessons CSVs in ZIP)
                frames=[]
                with zipfile.ZipFile(io.BytesIO(zip_bytes),"r") as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".csv") and "level lessons usage" in name.lower():
                            frames.append(read_csv_flex_from_bytes(zf.read(name)))
                lv = build_levelwise_from_frames(frames, logins_df=g_df)
                # ASR
                sa_df = load_school_assignments_from_zip(zip_bytes)
                tqa_map = load_teacher_quiz_by_class_from_zip(zip_bytes)
                asr = build_asr_quiz_split(sa_df, tqa_map) if sa_df is not None else pd.DataFrame()
                # store
                st.session_state["zip_su"]=su; st.session_state["zip_tu"]=tu
                st.session_state["zip_lv"]=lv; st.session_state["zip_asr"]=asr

    tab1, tab2, tab3, tab4 = st.tabs(["SU preview","TU preview","Levelwise preview","Assignment Summary"])

    with tab1:
        if not su.empty:
            center_table(su, key="su_tbl_zip")
            metrics=[c for c in ["No of Lessons Accessed","No of Logins","No of Students","Quiz","Worksheet","Prasso","Reading"] if c in su.columns]
            if metrics:
                m=st.selectbox("Chart metric", metrics, key="su_metric_zip")
                # natural class order on the x-axis (not value-sorted)
                topn = su.copy()
                topn = add_class_sort(topn, "Class")
                topn[m]=pd.to_numeric(topn[m], errors="coerce").fillna(0).astype(int)
                render_altair(
                    bar_with_labels(topn, x="Class", y=m, height=360, width=900, category_sort="_sort"),
                    "SU chart"
                )
        else:
            st.info("Upload ZIP and click Build.")

    with tab2:
        if not tu.empty:
            center_table(tu, key="tu_tbl_zip")
            t_opts=[c for c in ["No of Assignments Assigned","No of Lessons Accessed","No of logins"] if c in tu.columns]
            if t_opts and "Name" in tu.columns:
                tm=st.selectbox("Chart metric", t_opts, key="tu_metric_zip")
                by = (tu.groupby("Name")[tm].sum(numeric_only=True).sort_values(ascending=False).head(12)).reset_index()
                by[tm]=pd.to_numeric(by[tm], errors="coerce").fillna(0).astype(int)
                render_altair(bar_with_labels(by, x="Name", y=tm, horizontal=True, height=460, width=900), "TU chart")
        else:
            st.info("Provide Teachers Usage (optional) for TU.")

    with tab3:
        if not lv.empty:
            center_table(lv, key="lv_tbl_zip")
            metric="No of Lessons Accessed"
            if {"Class",metric}.issubset(lv.columns):
                tmp=lv[["Class",metric]].copy()
                tmp["GRADE"]=tmp["Class"].astype(str).str.extract(r"(\d+)").fillna("Other")
                st.subheader("Levelwise — per-grade charts")
                for g in sorted(tmp["GRADE"].unique(), key=lambda x:(x=="Other", float(x) if str(x).isdigit() else 9999)):
                    sub=tmp[tmp["GRADE"]==g].sort_values(metric, ascending=False)
                    if sub.empty: continue
                    sub=add_class_sort(sub,"Class")
                    st.markdown(f"**Grade {g}**")
                    # bar_size=18 -> slimmer bars
                    render_altair(bar_with_labels(sub, x="Class", y=metric, height=320, width=900, category_sort="_sort", bar_size=18),
                                  f"Levelwise chart (Grade {g})")
        else:
            st.info("No Levelwise CSVs detected in ZIP.")

    with tab4:
        if not asr.empty:
            center_table(asr[["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test","Total (Quiz)","ONGOING_AQC","AQC Diff (Total-ONGOING_AQC)","AQC Match?"]], key="asr_tbl_zip")
            req = ["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test"]
            if all(c in asr.columns for c in req):
                long = add_class_sort(asr[req].copy(), "Class").melt(
                    id_vars=["Class","_sort"], var_name="Type", value_name="Count"
                )
                long["Count"] = pd.to_numeric(long["Count"], errors="coerce").fillna(0).astype(int)

                sort_obj = alt.SortField(field="_sort", order="ascending")
                axis_q = alt.Axis(tickMinStep=1, format=".0f")

                color = alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test"],
                        range=["#4C78A8", "#F58518", "#54A24B"]  # three distinct colours
                    ),
                    legend=alt.Legend(title=None, orient="top")
                )

                chart = alt.Chart(long).mark_bar().encode(
                    x=alt.X("Class:N", sort=sort_obj),
                    xOffset="Type:N",                    # grouped/clustered bars
                    y=alt.Y("Count:Q", axis=axis_q),
                    color=color,
                    tooltip=["Class","Type","Count"]
                ).properties(width=900, height=380)

                labels = alt.Chart(long).mark_text(dy=-5).encode(
                    x=alt.X("Class:N", sort=sort_obj),
                    xOffset="Type:N",
                    y="Count:Q",
                    text=alt.Text("Count:Q", format=".0f"),
                    detail="Type:N"
                )

                render_altair((chart + labels).configure_axis(labelLimit=160, grid=True, gridColor="#f2f2f2"),
                              "Assignment Summary chart")
        else:
            st.info("ASR: supply School Assignments Usage + Teacher Quiz Assignments in ZIP.")

# ================= CSV FLOW =================
with tab_csv:
    su = st.session_state.get("csv_su", pd.DataFrame())
    tu = st.session_state.get("csv_tu", pd.DataFrame())
    lv = st.session_state.get("csv_lv", pd.DataFrame())
    asr = st.session_state.get("csv_asr", pd.DataFrame())

    st.markdown("### Core CSVs")
    a_up = st.file_uploader("Assignments CSV", type=["csv"], key="a")
    l_up = st.file_uploader("Lessons CSV", type=["csv"], key="l")
    g_up = st.file_uploader("Logins CSV",  type=["csv"], key="g")
    t_up = st.file_uploader("Teachers Usage CSV (optional, for TU)", type=["csv"], key="t")
    lvl_up = st.file_uploader("Level Lessons Usage CSVs (multiple, optional — for Levelwise)",
                              type=["csv"], accept_multiple_files=True, key="lvl")

    st.markdown("### Assignment Summary inputs (CSV mode)")
    sa_up   = st.file_uploader("School Assignments Usage Report (CSV)", type=["csv"], key="asr_sa_csv")
    tqa_ups = st.file_uploader("Teacher Quiz Assignment CSVs (one per class)", type=["csv"], accept_multiple_files=True, key="asr_tqa_csv")

    if sa_up:
        sa_df = read_csv_flex_from_bytes(sa_up.read())
        tqa_map={}
        for f in (tqa_ups or []):
            m=re.match(r"TeacherQuizAssignment_(.+?)_", f.name)
            klass = (m.group(1).strip() if m else f.name)
            tqa_map[normalize_class(klass)] = read_csv_flex_from_bytes(f.read())
        asr = build_asr_quiz_split(sa_df, tqa_map)
        st.session_state["csv_asr"]=asr

    submitted, mode, min_students, min_activity, whitelist = compact_options_form(prefix_key="csv")

    if submitted and a_up and l_up and g_up:
        a_df=read_csv_flex_from_bytes(a_up.read())
        l_df=read_csv_flex_from_bytes(l_up.read())
        g_df=read_csv_flex_from_bytes(g_up.read())
        t_df=read_csv_flex_from_bytes(t_up.read()) if t_up else None

        su = build_su(a_df,l_df,g_df)
        if mode.startswith("Active"): su=filter_active_grades(su, min_students, min_activity)
        elif mode=="Whitelist":
            allow={g.strip().lower() for g in whitelist.split(",") if g.strip()}
            if allow: su=su[su["Class"].str.lower().isin(allow)]

        tu = build_tu(t_df)
        frames=[read_csv_flex_from_bytes(f.read()) for f in (lvl_up or [])]
        lv = build_levelwise_from_frames(frames, logins_df=g_df)

        st.session_state["csv_su"]=su; st.session_state["csv_tu"]=tu; st.session_state["csv_lv"]=lv

    # previews
    tab1, tab2, tab3, tab4 = st.tabs(["School Usage (SU)","Teacher Usage (TU)","Levelwise","Assignment Summary"])

    with tab1:
        if not su.empty:
            center_table(su, key="su_tbl_csv")
            metrics=[c for c in ["No of Lessons Accessed","No of Logins","No of Students","Quiz","Worksheet","Prasso","Reading"] if c in su.columns]
            if metrics:
                m=st.selectbox("Chart metric", metrics, key="su_metric_csv")
                topn = su.sort_values(m, ascending=False).head(12).copy()
                topn[m]=pd.to_numeric(topn[m], errors="coerce").fillna(0).astype(int)
                topn=add_class_sort(topn,"Class")
                render_altair(bar_with_labels(topn, x="Class", y=m, height=360, width=900), "SU chart")
        else:
            st.info("Upload core CSVs and click Build to see SU.")

    with tab2:
        if not tu.empty:
            center_table(tu, key="tu_tbl_csv")
            t_opts=[c for c in ["No of Assignments Assigned","No of Lessons Accessed","No of logins"] if c in tu.columns]
            if t_opts and "Name" in tu.columns:
                tm=st.selectbox("Metric", t_opts, key="tu_metric_csv")
                by = (tu.groupby("Name")[tm].sum(numeric_only=True).sort_values(ascending=False).head(12)).reset_index()
                by[tm]=pd.to_numeric(by[tm], errors="coerce").fillna(0).astype(int)
                render_altair(bar_with_labels(by, x="Name", y=tm, horizontal=True, height=460, width=900), "TU chart")
        else:
            st.info("Upload Teachers Usage (optional) and click Build to see TU.")

    with tab3:
        if not lv.empty:
            center_table(lv, key="lv_tbl_csv")
            metric="No of Lessons Accessed"
            if {"Class",metric}.issubset(lv.columns):
                tmp=lv[["Class",metric]].copy(); tmp["GRADE"]=tmp["Class"].astype(str).str.extract(r"(\d+)").fillna("Other")
                st.subheader("Levelwise — per-grade charts")
                for g in sorted(tmp["GRADE"].unique(), key=lambda x:(x=="Other", float(x) if str(x).isdigit() else 9999)):
                    sub=tmp[tmp["GRADE"]==g].sort_values(metric, ascending=False)
                    if sub.empty: continue
                    sub=add_class_sort(sub,"Class")
                    st.markdown(f"**Grade {g}**")
                    render_altair(bar_with_labels(sub, x="Class", y=metric, height=320, width=900), f"Levelwise chart (Grade {g})")
            else:
                st.info("Expected columns 'Class' and 'No of Lessons Accessed'.")
        else:
            st.info("Upload Level Lessons files and click Build to see Levelwise.")

    with tab4:
        if not asr.empty:
            center_table(asr[["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test","Total (Quiz)","ONGOING_AQC","AQC Diff (Total-ONGOING_AQC)","AQC Match?"]], key="asr_tbl_csv")
            req=["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test"]
            if all(c in asr.columns for c in req):
                long=add_class_sort(asr[req].copy(),"Class").melt(id_vars=["Class","_sort"], var_name="Type", value_name="Count")
                long["Count"]=pd.to_numeric(long["Count"], errors="coerce").fillna(0).astype(int)
                sort_obj = alt.SortField(field="_sort", order="ascending")
                color = alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test"],
                        range=["#4C78A8", "#F58518", "#54A24B"]
                    ),
                    legend=alt.Legend(title=None, orient="top")
                )
                chart = alt.Chart(long).mark_bar().encode(
                    x=alt.X("Class:N", sort=sort_obj),
                    xOffset="Type:N",
                    y=alt.Y("Count:Q", axis=alt.Axis(tickMinStep=1, format=".0f")),
                    color=color,
                    tooltip=["Class","Type","Count"]
                ).properties(width=900, height=380)
                labels = alt.Chart(long).mark_text(dy=-5).encode(
                    x=alt.X("Class:N", sort=sort_obj),
                    xOffset="Type:N",
                    y="Count:Q",
                    text=alt.Text("Count:Q", format=".0f"),
                    detail="Type:N"
                )
                render_altair((chart + labels).configure_axis(labelLimit=160, grid=True, gridColor="#f2f2f2"),
                              "Assignment Summary chart")
                mism = asr[asr.get("AQC Match?", "").astype(str).str.lower()=="no"]
                if not mism.empty:
                    st.warning("Mismatch (Total Quiz ≠ ONGOING_AQC): " + ", ".join(mism["Class"].astype(str).tolist()))
        else:
            st.info("Upload School Assignments Usage + Teacher Quiz Assignment CSVs to build Assignment Summary.")

# ------------------------ Downloads (only if data exists) ------------
has_any = any(
    isinstance(st.session_state.get(k), pd.DataFrame) and not st.session_state[k].empty
    for k in ["zip_su","zip_tu","zip_lv","zip_asr","csv_su","csv_tu","csv_lv","csv_asr"]
)
if has_any:
    st.subheader("Downloads")
    dl_su  = st.session_state.get("zip_su",  st.session_state.get("csv_su",  pd.DataFrame()))
    dl_tu  = st.session_state.get("zip_tu",  st.session_state.get("csv_tu",  pd.DataFrame()))
    dl_lv  = st.session_state.get("zip_lv",  st.session_state.get("csv_lv",  pd.DataFrame()))
    dl_asr = st.session_state.get("zip_asr", st.session_state.get("csv_asr", pd.DataFrame()))

    xbuf=io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
        (dl_su if isinstance(dl_su, pd.DataFrame) else pd.DataFrame()).to_excel(w, "SU", index=False)
        (dl_tu if isinstance(dl_tu, pd.DataFrame) else pd.DataFrame()).to_excel(w, "TU", index=False)
        (dl_lv if isinstance(dl_lv, pd.DataFrame) else pd.DataFrame()).to_excel(w, "Levelwise", index=False)
        (dl_asr if isinstance(dl_asr, pd.DataFrame) else pd.DataFrame()).to_excel(w, "Assignment Summary", index=False)
    xbuf.seek(0)
    st.download_button("Download Consolidated xlsx", data=xbuf.getvalue(),
                       file_name="Consolidated.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    c1,c2 = st.columns(2)
    with c1:
        if not dl_su.empty:  st.download_button("Download SU.csv", data=dl_su.to_csv(index=False).encode("utf-8-sig"), file_name="SU.csv")
        if not dl_lv.empty:  st.download_button("Download Levelwise.csv", data=dl_lv.to_csv(index=False).encode("utf-8-sig"), file_name="Levelwise.csv")
    with c2:
        if not dl_tu.empty:  st.download_button("Download TU.csv", data=dl_tu.to_csv(index=False).encode("utf-8-sig"), file_name="TU.csv")
        if not dl_asr.empty: st.download_button("Download AssignmentSummary.csv", data=dl_asr.to_csv(index=False).encode("utf-8-sig"), file_name="AssignmentSummary.csv")
