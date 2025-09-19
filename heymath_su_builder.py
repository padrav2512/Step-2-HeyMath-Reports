# heymath_su_builder_focusonsu_FINAL.py
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

# st.markdown("""
# <style>
# /* Narrow page & reduce top padding */
# .block-container {max-width: 1100px; padding-top: .6rem; padding-bottom: 1rem;}
# /* Tighter headings */
# h1, h2, h3 {margin-top: .2rem;}
# /* Compact radio spacing */
# div.row-widget.stRadio > div {gap: .75rem}
# /* Strong primary button */
# button[kind="primary"]{font-weight:600;padding:.6rem 1rem;border-radius:.6rem}
# /* Center tables without going full width */
# .hm-narrow {max-width: 900px; margin: 0 auto;}
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Page width + comfortable top padding (prevents title clipping) */
.block-container {max-width: 1100px; padding-top: 1.25rem; padding-bottom: 1rem;}

/* Headings: slight margin + taller line-height */
h1, h2, h3 { margin-top: .4rem; line-height: 1.2; }

/* Center the first H1 on the page (your st.title) */
.block-container h1:first-child { text-align: center; }
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

def try_load_demo_book_bytes(default_path="Demo_ids_classesHandled.xlsx"):
    try:
        with open(default_path, "rb") as f:
            return f.read()
    except Exception:
        return None

def _coerce_metric_key(sel, cols_set):
    """Return a string column name present in cols_set, or None."""
    if isinstance(sel, str):
        return sel if sel in cols_set else None
    if isinstance(sel, dict):
        # Common dict shapes: {"col": "...", "name": "...", "value": "...", "key": "..."}
        for k in ("col", "name", "value", "key"):
            v = sel.get(k)
            if isinstance(v, str) and v in cols_set:
                return v
        # Last resort: try any string value from the dict
        for v in sel.values():
            if isinstance(v, str) and v in cols_set:
                return v
        return None
    # Fallback: stringify
    try:
        s = str(sel)
        return s if s in cols_set else None
    except Exception:
        return None

def _cols_str_set(df):
    return set(map(str, df.columns)) if isinstance(df, pd.DataFrame) else set()


# ------------------------ Chart helpers ------------------------------
# def render_altair(chart, title=None):
    # """Render an Altair chart only if it has rows; otherwise do nothing."""
    # if chart is None:
        # return

    # def _rowcount(c):
        # try:
            # # layered charts
            # if hasattr(c, "layer") and c.layer:
                # return sum(_rowcount(l) for l in c.layer)
            # # simple charts
            # if hasattr(c, "data") and c.data is not alt.Undefined and c.data is not None:
                # return len(c.data)
        # except Exception:
            # pass
        # return 0

    # if _rowcount(chart) == 0:
        # return  # <- no fallback text anymore
    # st.altair_chart(chart, use_container_width=True)

def render_altair(chart, title=None):
    if chart is None:
        return
    st.altair_chart(chart, use_container_width=True)



# def bar_with_labels(df, x, y, title=None, horizontal=False, height=320, width=900,
                    # category_sort=None, bar_size=None):
    # """Altair bar chart with integer axis & value labels (Altair v5 safe).
       # category_sort: None = sort by value; otherwise name of a column to sort by.
       # bar_size: fixed pixel width for bars (useful when few categories)."""
    # df = df.copy()
    # if x not in df.columns or y not in df.columns:
        # return alt.Chart(pd.DataFrame({"msg":["No data"]})).mark_text().encode(text="msg:N")
    # df = df[df[x].notna()]
    # df[y] = pd.to_numeric(df[y], errors="coerce").fillna(0)

    # sort_obj = alt.SortField(field=category_sort, order="ascending") if category_sort else ('-x' if horizontal else '-y')
    # axis_q = alt.Axis(tickMinStep=1, format=".0f")
    # base = alt.Chart(df).properties(width=width, height=height)
    # if title: base = base.properties(title=title)

    # mark_kwargs = {}
    # if bar_size is not None:
        # mark_kwargs["size"] = bar_size

    # if horizontal:
        # bars = base.mark_bar(**mark_kwargs).encode(
            # y=alt.Y(f"{x}:N", sort=sort_obj),
            # x=alt.X(f"{y}:Q", axis=axis_q),
            # tooltip=[x, y],
        # )
        # text = base.mark_text(align="left", dx=3).encode(
            # y=alt.Y(f"{x}:N", sort=sort_obj),
            # x=alt.X(f"{y}:Q"),
            # text=alt.Text(f"{y}:Q", format=".0f"),
        # )
    # else:
        # bars = base.mark_bar(**mark_kwargs).encode(
            # x=alt.X(f"{x}:N", sort=sort_obj),
            # y=alt.Y(f"{y}:Q", axis=axis_q),
            # tooltip=[x, y],
        # )
        # text = base.mark_text(align="center", dy=-5).encode(
            # x=alt.X(f"{x}:N", sort=sort_obj),
            # y=alt.Y(f"{y}:Q"),
            # text=alt.Text(f"{y}:Q", format=".0f"),
        # )
    # return (bars + text).configure_axis(labelLimit=160, grid=True, gridColor="#f2f2f2")
    
def bar_with_labels(df, x, y, title=None, horizontal=False, height=320, width=900,
                    category_sort=None, bar_size=None):
    """Altair bar chart with integer axis & value labels (Altair v5 safe)."""
    # normalize df
    if isinstance(df, pd.DataFrame):
        d = df.copy()
    elif df is None:
        d = pd.DataFrame()
    else:
        d = pd.DataFrame(df)

    # if required columns missing / empty → render nothing
    if x not in d.columns or y not in d.columns:
        return None
    d = d[d[x].notna()]
    if d.empty:
        return None

    d[y] = pd.to_numeric(d[y], errors="coerce")
    if d[y].isna().all():
        return None
    d[y] = d[y].fillna(0)

    # if a custom sort column was requested but doesn't exist, ignore it
    if category_sort and category_sort not in d.columns:
        category_sort = None

    sort_obj = alt.SortField(field=category_sort, order="ascending") if category_sort else ('-x' if horizontal else '-y')
    axis_q = alt.Axis(tickMinStep=1, format=".0f")
    base = alt.Chart(d).properties(width=width, height=height)
    if title:
        base = base.properties(title=title)

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

# def center_table(df: pd.DataFrame, height=420, key=None):
    # st.dataframe(df.reset_index(drop=True), height=height, use_container_width=True, key=key)
def center_table(df: pd.DataFrame, key=None, max_rows_visible: int = 12):
    """Display a dataframe with a height that fits its row count (no ghost blanks)."""
    d = df.reset_index(drop=True)
    # ~34px per row, ~38px header, ~12px padding
    row_h, header_h, pad = 34, 38, 12
    rows = max(1, len(d))
    vis = min(rows, max_rows_visible)
    height = header_h + pad + row_h * vis
    st.dataframe(d, height=int(height), use_container_width=True, key=key)

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
    """TU builder that works with totals OR row-per-event files.
    Falls back to counting rows when no numeric totals exist."""
    if teachers_df is None or teachers_df.empty:
        return pd.DataFrame(columns=["Name","No of logins","No of Lessons Accessed","No of Assignments Assigned"])

    df = teachers_df.copy()

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

def build_tu_enhanced(
    teachers_df: pd.DataFrame | None,
    t_map: dict[str, dict[str, pd.DataFrame]] | None,
    include_hold: bool = False,
    demo_teachers: set | None = None,
    classes_map: dict | None = None,     # <— NEW
) -> pd.DataFrame:
    """
    Output:
      Name | Is Demo Teacher | Classes Handled | No of logins | No of Lessons Accessed |
      No of Assignments Assigned | Quiz | Worksheet | Prasso | Reading
    - Merges Teachers Usage CSV totals (via build_tu) with counts from Teacher*Assignment files.
    - Adds userId (if present in Teachers Usage) and maps to Classes Handled via classes_map.
    """
    classes_map = {str(k).strip().lower(): str(v).strip() for k, v in (classes_map or {}).items()}
    demo_set = {str(s).strip().lower() for s in (demo_teachers or set())}

    # 1) Base from Teachers Usage CSV (your tolerant builder)
    base = build_tu(teachers_df)  # may be empty

    # Grab userId per teacher name if available
    id_map = pd.DataFrame()
    if teachers_df is not None and not teachers_df.empty:
        name_col = pick_col(teachers_df, "Name","userName","username","Teacher","Teacher Name","User")
        id_col   = pick_col(teachers_df, "userId","user_id","User ID","userid")
        if name_col:
            id_map = teachers_df[[name_col] + ([id_col] if id_col else [])].copy()
            id_map[name_col] = id_map[name_col].astype(str).str.strip()
            if id_col:
                id_map[id_col] = id_map[id_col].astype(str).str.strip()
            id_map = id_map.dropna(subset=[name_col]).drop_duplicates(subset=[name_col], keep="first")

    # 2) Counts from Teacher*Assignment files
    from collections import defaultdict
    per = defaultdict(lambda: {"Quiz":0, "Worksheet":0, "Prasso":0, "Reading":0})

    for kind, m in (t_map or {}).items():
        if not m: continue
        for _klass, df in m.items():
            if df is None or df.empty: continue
            tcol = pick_col(df, "teacher","Teacher","Teacher Name","username","userName","Name","user_id","User ID")
            if not tcol: continue
            d = df.copy()
            if "isHold" in d.columns and not include_hold:
                d = d.loc[normalize_is_hold(d["isHold"]) == 0]
            if d.empty: continue
            counts = d.groupby(d[tcol].astype(str).str.strip()).size()
            for nm, n in counts.items():
                per[str(nm).strip()][kind] += int(n)

    rows = []
    for nm, dd in per.items():
        total = int(dd["Quiz"] + dd["Worksheet"] + dd["Prasso"] + dd["Reading"])
        rows.append({"Name": nm, **dd, "No of Assignments Assigned": total})
    ass_df = pd.DataFrame(rows)

    # 3) Merge base + assignment counts
    out = pd.merge(base, ass_df, on="Name", how="outer", suffixes=("","_from_files"))

    # normalize numeric
    for c in ["Quiz","Worksheet","Prasso","Reading","No of logins","No of Lessons Accessed",
              "No of Assignments Assigned","No of Assignments Assigned_from_files"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # prefer max between CSV total and counted total
    if "No of Assignments Assigned_from_files" in out.columns:
        base_col = "No of Assignments Assigned"
        out[base_col] = out[[base_col, "No of Assignments Assigned_from_files"]].max(axis=1)
        out = out.drop(columns=["No of Assignments Assigned_from_files"])

    # 4) Attach userId (if we have it) and map Classes Handled
    # 4) Attach userId (if present) and map Classes Handled
    # Build a (Name -> userId) frame from teachers_df using flexible headers.
    out["userId"] = ""  # ensure the column exists even if we never find IDs

    id_map = pd.DataFrame()
    if teachers_df is not None and not teachers_df.empty:
        name_col = pick_col(teachers_df, "Name","userName","username","Teacher","Teacher Name","User")
        id_col   = pick_col(teachers_df, "userId","user_id","User ID","userid","UserId","USERID","User Id")
        if name_col:
            cols = [name_col] + ([id_col] if id_col else [])
            id_map = teachers_df[cols].copy()
            # normalize columns locally to avoid KeyError
            id_map = id_map.rename(columns={name_col: "_tu_name"})
            if id_col:
                id_map = id_map.rename(columns={id_col: "_tu_userId"})
            id_map["_tu_name"] = id_map["_tu_name"].astype(str).str.strip()
            if "_tu_userId" in id_map.columns:
                id_map["_tu_userId"] = id_map["_tu_userId"].astype(str).str.strip().str.lower()
            id_map = id_map.drop_duplicates(subset=["_tu_name"], keep="first")

    # Merge the ID map (if any) and materialize `userId`
    if not id_map.empty:
        out = out.merge(id_map, left_on="Name", right_on="_tu_name", how="left")
        if "_tu_userId" in out.columns:
            out["userId"] = out["_tu_userId"].astype(str).str.strip()
        out = out.drop(columns=[c for c in ["_tu_name","_tu_userId"] if c in out.columns])

    # Map to Classes Handled using the preloaded classes_map (user_id -> classes_taught)
    classes_map = {str(k).strip().lower(): str(v).strip() for k, v in (classes_map or {}).items()}
    out["Classes Handled"] = out["userId"].astype(str).str.strip().str.lower().map(lambda k: classes_map.get(k, "NA"))

    # 5) Demo flag (by name OR id)
    demo_set = {str(s).strip().lower() for s in (demo_teachers or set())}
    out["Is Demo Teacher"] = out["Name"].astype(str).str.strip().str.lower().isin(demo_set) \
                             | out["userId"].astype(str).str.strip().str.lower().isin(demo_set)

   

    # Final columns in the requested order (Classes Handled = 3rd)
    want = ["Name","Is Demo Teacher","Classes Handled",
            "No of logins","No of Lessons Accessed","No of Assignments Assigned",
            "Quiz","Worksheet","Prasso","Reading"]
    for c in want:
        if c not in out.columns:
            out[c] = 0 if c not in ("Name","Is Demo Teacher","Classes Handled") else ("NA" if c=="Classes Handled" else (False if c=="Is Demo Teacher" else ""))
    return out[want].sort_values("Name", kind="stable")




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

# def build_levelwise_with_assignments(lesson_frames, school_assign_df=None, logins_df=None):
    # """
    # Returns: Levelwise table with
      # Level | Class | No of Lessons Accessed | Assignments Assigned

    # - lesson_frames: list of 'Level Lessons Usage' dataframes (as you already pass)
    # - school_assign_df: a 'School Assignments Usage' dataframe (OPTIONAL). If present,
      # 'Assignments Assigned' is sum of ONGOING_* per class; else 0.
    # - logins_df: used to backfill Class labels from LEVEL when needed.
    # """
    # # ---- Lessons (reuse logic similar to build_levelwise_from_frames) ----
    # lessons = build_levelwise_from_frames(lesson_frames, logins_df=logins_df)
    # if lessons.empty:
        # lessons = pd.DataFrame(columns=["Level","Class","No of Lessons Accessed"])

    # # ---- Assignments (sum ONGOING_* per class) ----
    # assign = pd.DataFrame(columns=["Level","Class","Assignments Assigned"])
    # if school_assign_df is not None and not school_assign_df.empty:
        # d = school_assign_df.copy()
        # # tolerant header picks
        # lv_col   = next((c for c in d.columns if c.lower() in ("level_display_name","class")), None)
        # lvl_col  = next((c for c in d.columns if c.lower() == "level"), None)
        # aqc_col  = next((c for c in d.columns if "ongoing_aqc" in c.lower()), None)
        # wks_col  = next((c for c in d.columns if "ongoing_worksheet" in c.lower()), None)
        # prs_col  = next((c for c in d.columns if "ongoing_prasso" in c.lower()), None)
        # rdg_col  = next((c for c in d.columns if "ongoing_reading" in c.lower()), None)

        # if lv_col or lvl_col:
            # keep = [c for c in [lv_col, lvl_col, aqc_col, wks_col, prs_col, rdg_col] if c]
            # dd = d[keep].copy()
            # for c in [aqc_col, wks_col, prs_col, rdg_col]:
                # if c in dd.columns:
                    # dd[c] = pd.to_numeric(dd[c], errors="coerce").fillna(0).astype(int)

            # # Class label
            # if lv_col in dd.columns:
                # dd = dd.rename(columns={lv_col: "Class"})
            # elif lvl_col in dd.columns:
                # dd = dd.rename(columns={lvl_col: "LEVEL"})
                # # backfill class from logins_df if we have it
                # if logins_df is not None and {"LEVEL","LEVEL_DISPLAY_NAME"}.issubset(logins_df.columns):
                    # dd = dd.merge(
                        # logins_df[["LEVEL","LEVEL_DISPLAY_NAME"]].drop_duplicates(),
                        # on="LEVEL", how="left"
                    # ).rename(columns={"LEVEL_DISPLAY_NAME":"Class"})

            # # Level (nice label) from Class
            # def to_grade(s):
                # s=str(s); m=re.search(r'(\d+)', s)
                # return f"Grade {int(m.group(1))}" if m else s
            # if "Class" in dd.columns:
                # dd["Level"] = dd["Class"].apply(to_grade)

            # # Sum assignments
            # comp = [c for c in [aqc_col, wks_col, prs_col, rdg_col] if c in dd.columns]
            # if comp:
                # dd["Assignments Assigned"] = dd[comp].sum(axis=1).astype(int)
                # assign = dd[["Level","Class","Assignments Assigned"]].copy()

    # # ---- Merge lessons + assignments ----
    # out = pd.merge(
        # lessons,
        # assign,
        # on=["Level","Class"],
        # how="outer",
        # validate="one_to_one"
    # )
    # if "No of Lessons Accessed" not in out.columns:
        # out["No of Lessons Accessed"] = 0
    # if "Assignments Assigned" not in out.columns:
        # out["Assignments Assigned"] = 0
    # out["No of Lessons Accessed"] = pd.to_numeric(out["No of Lessons Accessed"], errors="coerce").fillna(0).astype(int)
    # out["Assignments Assigned"]    = pd.to_numeric(out["Assignments Assigned"], errors="coerce").fillna(0).astype(int)

    # # sort by class as in your helper
    # out = add_class_sort(out, "Class").sort_values(["_sort","Class"], kind="stable").drop(columns="_sort").reset_index(drop=True)
    # return out

def build_levelwise_with_assignments(lesson_frames, school_assign_df=None, logins_df=None):
    """
    Returns: Levelwise with columns:
      Level | Class | No of Lessons Accessed | Assignments Assigned
    """
    # Lessons (reuse your existing builder)
    lessons = build_levelwise_from_frames(lesson_frames, logins_df=logins_df)
    if lessons.empty:
        lessons = pd.DataFrame(columns=["Level","Class","No of Lessons Accessed"])

    # Assignments: sum ONGOING_* per class
    assign = pd.DataFrame(columns=["Level","Class","Assignments Assigned"])
    if school_assign_df is not None and not school_assign_df.empty:
        d = school_assign_df.copy()
        lv_col  = next((c for c in d.columns if c.lower() in ("level_display_name","class")), None)
        lvl_col = next((c for c in d.columns if c.lower() == "level"), None)
        aqc = next((c for c in d.columns if "ongoing_aqc" in c.lower()), None)
        wks = next((c for c in d.columns if "ongoing_worksheet" in c.lower()), None)
        prs = next((c for c in d.columns if "ongoing_prasso" in c.lower()), None)
        rdg = next((c for c in d.columns if "ongoing_reading" in c.lower()), None)
        alt_total = next((c for c in d.columns if "assignments assigned" in c.lower()
                          or "total assignments" in c.lower()
                          or "no of assignments" in c.lower()), None)

        keep = [x for x in [lv_col, lvl_col, aqc, wks, prs, rdg, alt_total] if x]
        if keep:
            dd = d[keep].copy()
            for c in [aqc,wks,prs,rdg,alt_total]:
                if c in dd.columns:
                    dd[c] = pd.to_numeric(dd[c], errors="coerce").fillna(0).astype(int)

            # Class label
            if lv_col in dd.columns:
                dd = dd.rename(columns={lv_col: "Class"})
            elif lvl_col in dd.columns:
                dd = dd.rename(columns={lvl_col: "LEVEL"})
                if logins_df is not None and {"LEVEL","LEVEL_DISPLAY_NAME"}.issubset(logins_df.columns):
                    dd = dd.merge(logins_df[["LEVEL","LEVEL_DISPLAY_NAME"]].drop_duplicates(),
                                  on="LEVEL", how="left").rename(columns={"LEVEL_DISPLAY_NAME":"Class"})

            # Level from Class
            def to_grade(s):
                s=str(s); m=re.search(r"(\d+)", s)
                return f"Grade {int(m.group(1))}" if m else s
            if "Class" in dd.columns:
                dd["Level"] = dd["Class"].apply(to_grade)

            # total assignments
            comp = [c for c in [aqc,wks,prs,rdg] if c in dd.columns]
            if comp:
                dd["Assignments Assigned"] = dd[comp].sum(axis=1).astype(int)
            elif alt_total and alt_total in dd.columns:
                dd["Assignments Assigned"] = dd[alt_total].astype(int)
            assign = dd[["Level","Class","Assignments Assigned"]].copy()

    out = pd.merge(lessons, assign, on=["Level","Class"], how="outer")
    if "No of Lessons Accessed" not in out.columns: out["No of Lessons Accessed"]=0
    if "Assignments Assigned" not in out.columns:    out["Assignments Assigned"]=0
    out["No of Lessons Accessed"] = pd.to_numeric(out["No of Lessons Accessed"], errors="coerce").fillna(0).astype(int)
    out["Assignments Assigned"]    = pd.to_numeric(out["Assignments Assigned"], errors="coerce").fillna(0).astype(int)
    out = add_class_sort(out, "Class").sort_values(["_sort","Class"], kind="stable").drop(columns="_sort").reset_index(drop=True)
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

# ===================== SU Helpers (strict per mapping) =====================
def _extract_class_from_teacher_assignment_name(name: str, base_label: str | None = None) -> str:
    """
    Extracts the class number from file names like:
      'TeacherQuizAssignment_Class 10_...csv'  -> 'Grade 10'
      'TeacherWorksheetAssignment_Standard5_...csv' -> 'Grade 5'
      '...IGCSE 7...' -> 'Grade 7'
    Returns '' if nothing found.
    """
    m = re.search(r"(Class|Grade|Std|Standard|IGCSE|CAM)\s*0*(\d{1,2})", str(name or ""), flags=re.I)
    if not m:
        return ""
    n = int(m.group(2))
    if 1 <= n <= 12:
        return f"Grade {n}"
    return ""


def load_teacher_assignments_by_class_from_zip(zip_bytes: bytes, kind: str, base_label: str | None = None) -> dict[str, pd.DataFrame]:
    patt = re.compile(rf"Teacher\s*{re.escape(kind)}\s*Assignment[^/]*?\.csv$", re.I)
    out={}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            if not patt.search(name):
                continue
            df = read_csv_flex_from_bytes(zf.read(name))
            klass = _extract_class_from_teacher_assignment_name(name, base_label=base_label)
            out[normalize_class(klass)] = df
    return out
    
def _clean_detail_table(df: pd.DataFrame, drop_mode: bool = False) -> pd.DataFrame:
    """Drop empty rows/cols, strip whitespace, de-dup, optionally hide Mode."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    # strip spaces in object cols
    for c in d.columns:
        if pd.api.types.is_string_dtype(d[c]) or d[c].dtype == "object":
            d[c] = d[c].astype(str).str.strip()

    # standardize recipient token
    if "Recipient Token" in d.columns:
        rt = d["Recipient Token"].astype(str).str.strip()
        rt_low = rt.str.lower()
        d = d[~(rt_low.isna() | (rt_low == "") | (rt_low == "__empty__") | (rt_low == "nan"))]

    # drop rows where BOTH teacher and assignment name are blank (if those cols exist)
    t_blank = d["Teacher"].astype(str).str.strip().eq("") if "Teacher" in d.columns else False
    a_blank = d["Assignment Name"].astype(str).str.strip().eq("") if "Assignment Name" in d.columns else False
    if isinstance(t_blank, pd.Series) and isinstance(a_blank, pd.Series):
        d = d[~(t_blank & a_blank)]

    d = d.dropna(how="all").drop_duplicates()

    if drop_mode:
        d = d.drop(columns=["Mode"], errors="ignore")

    # nice order if present
    cols = [c for c in ["Class","Type","Teacher","Assignment Name","Mode","Recipient Token"] if c in d.columns]
    if cols:
        d = d[cols]
    return d.reset_index(drop=True)



# === Class token detection (prefix-based; supports sections like "Class 1A1 SS") ===
CLASS_START = re.compile(r'^\s*(class|standard|std|grade|igcse|cam)\s*0*(1[0-2]|[1-9])', re.I)
def _token_is_class(token: str) -> bool:
    return bool(CLASS_START.match(str(token or "")))

def _split_tokens(val) -> list[str]:
    s = str(val or "")
    # If someone forgot a comma before a class tag (… "Honey bee Class 8C"),
    # insert a comma before known class prefixes followed by a digit.
    s = re.sub(r'(?<![,])\s+(?=(?:class|standard|std|grade)\s*0*(?:1[0-2]|[1-9]))', ', ', s, flags=re.I)
    # Split on ASCII comma or fullwidth comma, allowing stray spaces
    toks = [t.strip() for t in re.split(r'\s*[,，]\s*', s) if t.strip()]
    return toks or ["__EMPTY__"]


def _count_assignments_from_map(
    
    assign_map: dict,
    include_demo_t: bool = False,
    include_demo_s: bool = False,
    include_hold:   bool = False,
    include_indiv:  bool = False,
    demo_teacher_ids: set | None = None,
    demo_student_ids: set | None = None,
):
    """Token-based counting per class with grouped detail rows."""
    from collections import defaultdict, OrderedDict

    eff   = defaultdict(int)
    demo_t = defaultdict(int)
    demo_s = defaultdict(int)
    hold   = defaultdict(int)
    indiv  = defaultdict(int)

    # group stores: key=(Class, Teacher, Title, Mode) -> Ordered unique tokens
    grp_indiv: dict[tuple, OrderedDict] = {}
    grp_demo_t: dict[tuple, OrderedDict] = {}
    grp_demo_s: dict[tuple, OrderedDict] = {}
    grp_hold:   dict[tuple, OrderedDict] = {}

    dt_ids = {str(x).strip().lower() for x in (demo_teacher_ids or set())}
    ds_ids = {str(x).strip().lower() for x in (demo_student_ids or set())}

    def _add_token(store: dict, key: tuple, token: str):
        od = store.setdefault(key, OrderedDict())
        if token not in od:
            od[token] = True  # preserve insertion order

    for klass, df in (assign_map or {}).items():
        if df is None or df.empty:
            continue

        # teacher_col = 'teacher' if 'teacher' in df.columns else None
        # title_col   = '_title' if '_title' in df.columns else ('title' if 'title' in df.columns else None)
        # mode_col    = '_mode' if '_mode' in df.columns else ('assignmentGroupMode' if 'assignmentGroupMode' in df.columns else ('Mode' if 'Mode' in df.columns else ('Type' if 'Type' in df.columns else None)))
       
        # teacher_col = 'teacher' if 'teacher' in df.columns else None
        # title_col   = '_title' if '_title' in df.columns else ('title' if 'title' in df.columns else None)
        # mode_col    = '_mode' if '_mode' in df.columns else ('assignmentGroupMode' if 'assignmentGroupMode' in df.columns else ('Mode' if 'Mode' in df.columns else ('Type' if 'Type' in df.columns else None)))

        # --- with this: ---
        teacher_col = pick_col(
            df,
            "teacher", "Teacher", "Teacher Name", "user_id", "User ID",
            "username", "userName", "Name"
        )
        title_col = pick_col(
            df,
            "_title", "title", "Title", "Assignment Name", "assessmentName",
            "Assessment Name", "Task Title"
        )
        mode_col = pick_col(
            df,
            "_mode", "assignmentGroupMode", "Mode", "Type",
            "assessmentMode", "Assessment Mode", "mode"
        )
        assigned_to_col = pick_col(
            df,
            "assignedTo", "Assigned To", "AssignedTo", "Assigned_To",
            "Recipients", "Recipient", "Recipient Token", "Assigned"
        )

        for _, r in df.iterrows():
            teacher_val = str(r.get(teacher_col, "")).strip()
            teacher_key = teacher_val.lower()
            title = str(r.get(title_col, "")) if title_col else ""
            mode  = str(r.get(mode_col, "")) if mode_col else ""
            #tokens = _split_tokens(r.get('assignedTo'))
            tokens = _split_tokens(r.get(assigned_to_col))


            row_is_hold = False
            if 'isHold' in df.columns:
                row_is_hold = int(pd.to_numeric(r.get('isHold', 0), errors='coerce') or 0) == 1

            for tok in tokens:
                tok_norm = str(tok).strip()
                tok_key  = tok_norm.lower()
                is_class = _token_is_class(tok_norm)

                if row_is_hold:
                    hold[klass] += 1
                    _add_token(grp_hold,   (klass, teacher_val, title, mode), tok_norm)
                    continue

                if teacher_key in dt_ids:
                    demo_t[klass] += 1
                    _add_token(grp_demo_t, (klass, teacher_val, title, mode), tok_norm)
                    continue

                if (not is_class) and (tok_key in ds_ids):
                    demo_s[klass] += 1
                    _add_token(grp_demo_s, (klass, teacher_val, title, mode), tok_norm)
                    continue

                if not is_class:
                    indiv[klass] += 1
                    _add_token(grp_indiv,  (klass, teacher_val, title, mode), tok_norm)
                    continue

                # class token -> SU base
                eff[klass] += 1

    # apply toggles
    if include_hold:
        for k, v in hold.items():   eff[k] += v
    if include_demo_t:
        for k, v in demo_t.items(): eff[k] += v
    if include_demo_s:
        for k, v in demo_s.items(): eff[k] += v
    if include_indiv:
        for k, v in indiv.items():  eff[k] += v

    # build grouped detail rows (comma-separated recipients)
    def _to_rows(store: dict):
        rows=[]
        for (klass, teacher, title, mode), od in store.items():
            tokens = [t for t in od.keys() if t not in (None, "__EMPTY__")]
            rows.append({
                "Class": klass,
                "Teacher": teacher,
                "Assignment Name": title,
                "Mode": mode,
                "Recipient Token": ", ".join(tokens)
            })
        return rows

    detail = {
        "indiv_bands": _to_rows(grp_indiv),
        "demo_t":      _to_rows(grp_demo_t),
        "demo_s":      _to_rows(grp_demo_s),
        "hold":        _to_rows(grp_hold),
    }
    return eff, demo_t, demo_s, hold, detail

def build_su_from_teacher_maps(
    logins_df,
    lessons_df,
    maps_by_kind,
    include_demo_t: bool = False,
    include_demo_s: bool = False,
    include_hold:   bool = False,
    include_indiv:  bool = False,
    demo_teacher_ids: set | None = None,
    demo_student_ids: set | None = None, demo_students_by_class: dict[str,int] | None = None
):

    if logins_df is None or logins_df.empty:
        raise ValueError('School Logins Report missing/empty')
    need=['LEVEL','LEVEL_DISPLAY_NAME','TOTAL_STUDENTS','TOTAL_LOGINS']
    miss=[c for c in need if c not in logins_df.columns]
    if miss:
        raise ValueError(f'Logins CSV missing required columns: {miss}')

    base = logins_df[['LEVEL','LEVEL_DISPLAY_NAME','TOTAL_STUDENTS','TOTAL_LOGINS']].copy()
    base = base.rename(columns={'LEVEL_DISPLAY_NAME':'Class','TOTAL_STUDENTS':'No of Students','TOTAL_LOGINS':'No of Logins'})

    if lessons_df is not None and not lessons_df.empty and {'LEVEL','LESSONS_ACCESSED'}.issubset(lessons_df.columns):
        tmp = lessons_df[['LEVEL','LESSONS_ACCESSED']].copy().rename(columns={'LESSONS_ACCESSED':'No of Lessons Accessed'})
        base = base.merge(tmp, on='LEVEL', how='left')
    else:
        base['No of Lessons Accessed'] = 0

    for c in ['No of Students','No of Logins','No of Lessons Accessed']:
        base[c] = pd.to_numeric(base[c], errors='coerce').fillna(0).astype(int)

    kinds = ['Quiz','Worksheet','Prasso','Reading']
    detail_rows = {'indiv_bands':[], 'demo_t':[], 'demo_s':[], 'hold':[]}
    by_kind_counts = {}

    for k in kinds:
        eff, d_t, d_s, h, detail = _count_assignments_from_map(
            maps_by_kind.get(k) or {},
            include_demo_t=include_demo_t,
            include_demo_s=include_demo_s,
            include_hold=include_hold,
            include_indiv=include_indiv,
            demo_teacher_ids=demo_teacher_ids,
            demo_student_ids=demo_student_ids,
        )
        by_kind_counts[k] = eff

        for key in detail_rows.keys():
            for r in detail[key]:
                rr = dict(r); rr["Type"] = k
                detail_rows[key].append(rr)

    def keyify(lbl: str) -> str:
        s = str(lbl or '').lower().strip()
        m = re.search(r'(\d+)', s)
        return normalize_class(f'grade {int(m.group(1))}') if m else normalize_class(s)

    for k in kinds:
        base[k] = base['Class'].map(lambda c: by_kind_counts.get(k,{}).get(keyify(c), 0)).fillna(0).astype(int)

    out = add_class_sort(base.rename(columns={'LEVEL':'_LEVEL'}), 'Class').sort_values(['_sort','Class'], kind='stable').drop(columns='_sort')
    cols = ['Class','No of Students','No of Logins','No of Lessons Accessed','Quiz','Worksheet','Prasso','Reading']
    for c in cols:
        if c not in out.columns: out[c]=0
    out = out[cols]
    # --- NEW: subtract demo-student counts per class in SU ---
    if demo_students_by_class:
        adj = out.copy()
        adj["_norm_class"] = adj["Class"].map(normalize_class)
        adj["No of Students"] = (
            adj.apply(
                lambda r: max(
                    0,
                    int(r["No of Students"]) - int(demo_students_by_class.get(r["_norm_class"], 0))
                ),
                axis=1
            )
        )
        out = adj.drop(columns=["_norm_class"])

    det = {k: pd.DataFrame(v, columns=["Class","Type","Teacher","Assignment Name","Mode","Recipient Token"]) for k,v in detail_rows.items()}
    return out, det

# ---- Demo ids / Classes Handled workbook loader ----
def load_demo_book(xls_bytes) -> tuple[set, set, dict, dict]:
    demo_t, demo_s, classes_map = set(), set(), {}
    demo_students_by_class: dict[str,int] = {}

    try:
        xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    except Exception:
        return demo_t, demo_s, classes_map

    def first_text_col(df):
        for c in df.columns:
            ser = df[c].dropna().astype(str).str.strip()
            if not ser.empty and (ser != "").any():
                return ser.tolist()
        return []

    for sh in xls.sheet_names:
        low = sh.lower()
        df = pd.read_excel(xls, sheet_name=sh)
        if "demo" in low and "teacher" in low:
            demo_t.update([str(x).strip() for x in first_text_col(df) if str(x).strip()])
        elif "demo" in low and "student" in low:
            demo_s.update([str(x).strip() for x in first_text_col(df) if str(x).strip()])
        elif "class" in low:
            if len(df.columns) >= 2:
                tcol, ccol = df.columns[:2]
                for _, r in df.iterrows():
                    t = str(r.get(tcol, "")).strip()
                    cl = str(r.get(ccol, "")).strip()
                    if t and cl:
                        classes_map.setdefault(t, []).append(cl)
            else:
                cls = first_text_col(df)
                if cls:
                    classes_map.setdefault("_all", []).extend([str(x).strip() for x in cls if str(x).strip()])
        # --- NEW: demo student counts per class (sheet: demo_stud_ids) ---
        try:
            if "demo_stud_ids" in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name="demo_stud_ids")
                # Pick the class column heuristically
                # (accepts 'Class', 'Level', 'LEVEL_DISPLAY_NAME' or the first text-like column)
                cls_col = None
                for c in ["Class","CLASS","Level","LEVEL","LEVEL_DISPLAY_NAME"]:
                    if c in df.columns: 
                        cls_col = c
                        break
                if cls_col is None:
                    for c in df.columns:
                        if df[c].astype(str).str.strip().ne("").any():
                            cls_col = c
                            break

                if cls_col:
                    # normalize class labels using your existing helper
                    df["_norm_class"] = df[cls_col].astype(str).map(normalize_class)
                    demo_students_by_class = (
                        df[df["_norm_class"].astype(str).str.strip().ne("")]
                          .groupby("_norm_class")["_norm_class"]
                          .size()
                          .to_dict()
                    )
        except Exception:
            pass

    return demo_t, demo_s, classes_map, demo_students_by_class


def _df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def _df_to_xlsx_bytes(df, sheet_name="data"):
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        pass
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()

# ------------------------ UI -----------------------------------------
st.title("HeyMath! School Report Builder")
tab_zip, tab_csv = st.tabs(["Upload ZIP (recommended)","Upload CSVs"])

# ================= ZIP FLOW =================
with tab_zip:
    # Inputs
    up = st.file_uploader("Drop a HeyMath ZIP here", type=["zip"])
    #demo_book = st.file_uploader("Upload Demo_ids_classesHandled.xlsx (optional — demo ids)", type=["xlsx"], key="zip_demo_book")
    demo_book = None
    
    su = st.session_state.get("zip_su", pd.DataFrame())
    tu = st.session_state.get("zip_tu", pd.DataFrame())
    lv = st.session_state.get("zip_lv", pd.DataFrame())
    asr = st.session_state.get("zip_asr", pd.DataFrame())

    submitted, mode, min_students, min_activity, whitelist = compact_options_form(prefix_key="zip")

    if up is not None:
        zip_bytes = up.read()
        a_df, l_df, g_df, t_df, names = detect_files_in_zip(zip_bytes)
        st.session_state["zip_assign_df"] = a_df 
        if any(x is None for x in (a_df,l_df,g_df)):
            st.error("Could not auto-detect Assignments/Lessons/Logins from the ZIP.")
        else:
            st.caption(f"Detected: Assignments=`{names['assign']}`, Lessons=`{names['lessons']}`, Logins=`{names['logins']}`, Teachers=`{names['teachers']}`")

       
        # --- before building SU, load the demo workbook automatically if present ---
        auto_demo_bytes = try_load_demo_book_bytes()                  # uses local Demo_ids_classesHandled.xlsx if present
        demo_bytes = auto_demo_bytes or (demo_book.read() if demo_book else None)

        demo_t_ids, demo_t_names, demo_s_ids = set(), set(), set()
        classes_map = {}  # user_id(lower) -> classes_taught

        if demo_bytes:
            try:
                xls = pd.ExcelFile(io.BytesIO(demo_bytes))

                # demo teachers
                if "demo_tchr_ids" in xls.sheet_names:
                    tdf = pd.read_excel(xls, "demo_tchr_ids")
                    if "user_id" in tdf.columns:
                        demo_t_ids = set(tdf["user_id"].astype(str).str.strip())
                    if "Name" in tdf.columns:
                        demo_t_names = set(tdf["Name"].astype(str).str.strip())

                # demo students
                if "demo_stud_ids" in xls.sheet_names:
                    sdf = pd.read_excel(xls, "demo_stud_ids")
                    if "user_id" in sdf.columns:
                        demo_s_ids = set(sdf["user_id"].astype(str).str.strip())

                # classes_handled sheet -> classes_map
                cls_sheet = next((s for s in xls.sheet_names if "class" in s.lower() and "handle" in s.lower()), None)
                if cls_sheet:
                    cdf = pd.read_excel(xls, cls_sheet)
                    uid_col = next((c for c in cdf.columns if c.lower() in ("user_id","userid","user id")), None)
                    taught_col = next((c for c in cdf.columns if "classes_taught" in c.lower() or "classes handled" in c.lower()), None)
                    if uid_col and taught_col:
                        tmp = cdf[[uid_col, taught_col]].copy()
                        tmp[uid_col]    = tmp[uid_col].astype(str).str.strip().str.lower()
                        tmp[taught_col] = tmp[taught_col].astype(str).str.strip().replace({"": "NA"})
                        classes_map = {u: t for u, t in zip(tmp[uid_col], tmp[taught_col]) if u}

                if auto_demo_bytes:
                    st.caption("Loaded demo workbook from local file: Demo_ids_classesHandled.xlsx")
                elif demo_book:
                    st.caption(f"Loaded demo workbook from upload: {demo_book.name}")
                # --- NEW: count demo students by class (normalized label) ---
                demo_students_by_class = {}
                try:
                    if "demo_stud_ids" in xls.sheet_names:
                        sdf = pd.read_excel(xls, "demo_stud_ids")
                        # Pick a class-like column robustly (you already have pick_col + normalize_class)
                        cls_col = pick_col(sdf, "Class","LEVEL_DISPLAY_NAME","LEVEL","Level","class")
                        if cls_col:
                            sdf["_norm_class"] = sdf[cls_col].astype(str).map(normalize_class)
                            demo_students_by_class = sdf["_norm_class"].value_counts().to_dict()
                except Exception:
                    demo_students_by_class = {}

                # Keep it for later use (re-renders)
                st.session_state["zip_demo_students_by_class"] = demo_students_by_class

            except Exception as e:
                st.warning(f"Could not read Demo_ids_classesHandled.xlsx ({type(e).__name__}). Proceeding without demo ids.")

        # store in session for preview recompute
        st.session_state["zip_demo_t_ids"]   = demo_t_ids
        st.session_state["zip_demo_t_names"] = demo_t_names
        st.session_state["zip_demo_s_ids"]   = demo_s_ids
        st.session_state["zip_classes_map"]   = classes_map
        
        if submitted:
            # Build Teacher*Assignment maps from ZIP
            t_map = {
                "Quiz":       load_teacher_assignments_by_class_from_zip(zip_bytes, "Quiz"),
                "Worksheet":  load_teacher_assignments_by_class_from_zip(zip_bytes, "Worksheet"),
                "Prasso":     load_teacher_assignments_by_class_from_zip(zip_bytes, "Prasso"),
                "Reading":    load_teacher_assignments_by_class_from_zip(zip_bytes, "Reading"),
            }

            # Combined demo teachers (ids + names, lowercased)
            combined_demo_teachers = {s.strip().lower() for s in demo_t_ids} | {s.strip().lower() for s in demo_t_names}

            # Initial SU (all toggles off by default)
            
            su, su_details = build_su_from_teacher_maps(
                g_df, l_df, t_map,
                include_demo_t=False, include_demo_s=False, include_hold=False, include_indiv=False,
                demo_teacher_ids=combined_demo_teachers,
                demo_student_ids=demo_s_ids
            )

        
            # --- NEW: subtract per-class demo-student counts from SU "No of Students" ---
            dsbc = st.session_state.get("zip_demo_students_by_class", {})
            if dsbc and not su.empty and "No of Students" in su.columns and "Class" in su.columns:
                su["_norm_class"] = su["Class"].astype(str).map(normalize_class)
                su["No of Students"] = su.apply(
                    lambda r: max(0, int(r["No of Students"]) - int(dsbc.get(r["_norm_class"], 0))),
                    axis=1
                )
                su.drop(columns=["_norm_class"], inplace=True)
# # (optional) quick debug of teacher-assignment files found
            # if st.checkbox("Show TU debug", value=False, key="tu_dbg_toggle"):
                # with st.expander("Debug: Teacher assignment files found"):
                    # for kind, m in (t_map or {}).items():
                        # classes = sorted([k for k, df in (m or {}).items() if df is not None and not df.empty])
                        # st.write(f"{kind}: {classes}")
            

            # --- SU filters ---
            if mode.startswith("Active"):
                su = filter_active_grades(su, min_students, min_activity)
            elif mode == "Whitelist":
                allow = {g.strip().lower() for g in whitelist.split(",") if g.strip()}
                if allow:
                    su = su[su["Class"].str.lower().isin(allow)]

        
            combined_demo_teachers = {s.strip().lower() for s in demo_t_ids} | {s.strip().lower() for s in demo_t_names}
            tu = build_tu_enhanced(
                teachers_df=t_df,
                t_map=t_map,
                include_hold=False,
                demo_teachers=combined_demo_teachers,
                classes_map=st.session_state.get("zip_classes_map", {})  # <-- uses what we just stored
            )
            # Recompute total assignments from components (ZIP TU)
            for _c in ["Quiz", "Worksheet", "Prasso", "Reading"]:
                if _c not in tu.columns:
                    tu[_c] = 0
            tu["No of Assignments Assigned"] = (
                pd.to_numeric(tu["Quiz"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(tu["Worksheet"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(tu["Prasso"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(tu["Reading"], errors="coerce").fillna(0).astype(int))


            st.session_state["zip_tu"] = tu


            # Levelwise (from any "Level Lessons Usage" CSVs found inside the ZIP)
            frames = []
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".csv") and "level lessons usage" in name.lower():
                        try:
                            frames.append(read_csv_flex_from_bytes(zf.read(name)))
                        except Exception as e:
                            st.warning(f"Could not read Level Lessons file '{name}' ({type(e).__name__}).")

            lv = build_levelwise_from_frames(frames, logins_df=g_df)
            st.session_state["zip_lv"] = lv



            # ASR
            sa_df = load_school_assignments_from_zip(zip_bytes)
            tqa_map = load_teacher_quiz_by_class_from_zip(zip_bytes)
            asr = build_asr_quiz_split(sa_df, tqa_map) if sa_df is not None else pd.DataFrame()

            # store
            st.session_state["zip_su"]=su
            st.session_state["zip_su_details"]=su_details
            st.session_state["zip_tu"]=tu
            st.session_state["zip_t_map"]=t_map
            st.session_state["zip_g_df"]=g_df
            st.session_state["zip_l_df"]=l_df
            st.session_state["zip_lv"]=lv
            st.session_state["zip_asr"]=asr

    tab1, tab2, tab3, tab4 = st.tabs(["SU preview","TU preview","Levelwise preview","Assignment Summary"])

    with tab1:
        st.markdown("### SU preview")

        # Ensure defaults
        if "zip_demo_t_ids" not in st.session_state:
            st.session_state["zip_demo_t_ids"] = set()
        if "zip_demo_t_names" not in st.session_state:
            st.session_state["zip_demo_t_names"] = set()
        if "zip_demo_s_ids" not in st.session_state:
            st.session_state["zip_demo_s_ids"] = set()

        # === Options (all unchecked by default) ===
        with st.expander("SU counting options", expanded=False):
            opt_hold  = st.checkbox("Count on Hold Assignments", value=False, key="zip_view_hold")
            opt_indiv = st.checkbox("Count Assignments to Individuals & Bands", value=False, key="zip_view_indiv")
            opt_demo_t = st.checkbox("Count assignments by demo teachers", value=False, key="zip_view_demo_t")
            opt_demo_s = st.checkbox("Count assignments to demo students", value=False, key="zip_view_demo_s")

        # Recompute SU with current toggles
        _g = st.session_state.get("zip_g_df"); _l = st.session_state.get("zip_l_df"); _m = st.session_state.get("zip_t_map")
        demo_t_ids = st.session_state.get("zip_demo_t_ids", set())
        demo_t_names = st.session_state.get("zip_demo_t_names", set())
        demo_s_ids = st.session_state.get("zip_demo_s_ids", set())
        combined_demo_teachers = {s.strip().lower() for s in demo_t_ids} | {s.strip().lower() for s in demo_t_names}

        st.caption(f"Loaded demo teachers: ids={len(demo_t_ids)}, names={len(demo_t_names)}; demo students={len(demo_s_ids)}")

        if _g is not None and _l is not None and _m is not None:
            try:
                su_view, det_view = build_su_from_teacher_maps(
                    _g, _l, _m,
                    include_demo_t=opt_demo_t,
                    include_demo_s=opt_demo_s,
                    include_hold=opt_hold,
                    include_indiv=opt_indiv,
                    demo_teacher_ids=combined_demo_teachers,
                    demo_student_ids=demo_s_ids
                )
            except Exception as e:
                st.warning(f"Recompute failed: {e}")
                su_view, det_view = st.session_state.get("zip_su"), st.session_state.get("zip_su_details")
        else:
            su_view, det_view = st.session_state.get("zip_su"), st.session_state.get("zip_su_details")
        
        # Apply Grade selection to the current preview as well
        if isinstance(su_view, pd.DataFrame) and not su_view.empty:
            if mode.startswith("Active"):
                su_view = filter_active_grades(su_view, min_students, min_activity)
            elif mode == "Whitelist":
                allow = {g.strip().lower() for g in whitelist.split(",") if g.strip()}
                if allow:
                    su_view = su_view[su_view["Class"].astype(str).str.lower().isin(allow)]
            # mode == "All" -> no filter
        # --- SU TABLE (show first) ---
        if isinstance(su_view, pd.DataFrame) and not su_view.empty:
            center_table(su_view, key="zip_su_tbl")
        else:
            st.info("No SU data to show yet. Upload a ZIP and click Build.")
        # don't st.stop(); allow the chart/info below if you prefer

        # --- metric picker (string-only) ---
        cols_set = _cols_str_set(su_view)
        metrics = [c for c in ["No of Lessons Accessed","No of Students","No of Logins","Quiz","Worksheet","Prasso","Reading"] if c in cols_set]
        m_sel = st.selectbox("Chart metric", metrics, index=0 if metrics else None, key="su_metric_zip")
        m = _coerce_metric_key(m_sel, cols_set)

        if isinstance(su_view, pd.DataFrame) and not su_view.empty and m:
            plot_df = su_view[["Class", m]].copy()
            plot_df = add_class_sort(plot_df, "Class")
            plot_df[m] = pd.to_numeric(plot_df[m], errors="coerce").fillna(0).astype(int)

            nrows = len(plot_df)
            dyn_h = max(260, min(26 * max(nrows, 1) + 60, 1400))

            chart = bar_with_labels(
                plot_df, x="Class", y=m,
                height=dyn_h, width=900, category_sort="_sort", bar_size=18
            )
            render_altair(chart, "SU chart")
        else:
            st.info("No SU data for the selected metric.")

 
        # === Detail tables (grouped, comma-separated) ===
        det = det_view or {}
        _df_ib = _clean_detail_table(det.get("indiv_bands", pd.DataFrame()), drop_mode=True)  # hide Mode
        _df_t  = _clean_detail_table(det.get("demo_t", pd.DataFrame()))
        _df_s  = _clean_detail_table(det.get("demo_s", pd.DataFrame()))
        _df_h  = _clean_detail_table(det.get("hold",   pd.DataFrame()))

        # define helper BEFORE calls
        def _dl(df: pd.DataFrame, base: str, key_prefix: str):
            if df is not None and not df.empty:
                _csv = _df_to_csv_bytes(df)
                st.download_button("Download CSV", _csv,
                                   file_name=f"{base}.csv", mime="text/csv",
                                   key=f"{key_prefix}_csv")
                try:
                    _xlsx = _df_to_xlsx_bytes(df, sheet_name=base)
                    st.download_button("Download XLSX", _xlsx,
                                       file_name=f"{base}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       key=f"{key_prefix}_xlsx")
                except Exception:
                    pass

        # render tables once (after helper is defined)
        if not _df_ib.empty:
            st.subheader("Assignments to Individuals & Bands")
            center_table(_df_ib, key="zip_su_tbl_indiv")
            _dl(_df_ib, "assignments_to_individuals_bands", "zip_indiv")

        if not _df_t.empty:
            st.subheader("Assignments issued by demo teachers")
            center_table(_df_t, key="zip_su_tbl_demo_t")
            _dl(_df_t, "assignments_by_demo_teachers", "zip_demo_t")

        if not _df_s.empty:
            st.subheader("Assignments issued to demo students")
            center_table(_df_s, key="zip_su_tbl_demo_s")
            _dl(_df_s, "assignments_to_demo_students", "zip_demo_s")

        if not _df_h.empty:
            st.subheader("Assignments on Hold")
            center_table(_df_h, key="zip_su_tbl_hold")
            _dl(_df_h, "assignments_on_hold", "zip_hold")

     
    with tab2:
        st.markdown("### TU preview")

        tu = st.session_state.get("zip_tu", pd.DataFrame())
        if not isinstance(tu, pd.DataFrame) or tu.empty:
            st.info("Provide Teachers Usage (optional) and/or Teacher*Assignment files to build TU.")
        else:
            # ---------- Filters (all in TU; Streamlit keeps you on this tab on rerun) ----------
            colf1, colf2, colf3, colf4 = st.columns([1,1,1,2])
            with colf1:
                only_demo = st.checkbox("Only demo", value=False, key="tu_only_demo")
            with colf2:
                hide_demo = st.checkbox("Hide demo", value=True, key="tu_hide_demo")
            with colf3:
                hide_na_classes = st.checkbox("Hide NA classes", value=False, key="tu_hide_na_classes")  # (1)
            with colf4:
                search = st.text_input("Filter by teacher name (contains)", value="", key="tu_search")
            # ---------- Zero filters (enabled by default) ----------
            cza, czb = st.columns([1,1])
            with cza:
                hide_zero_lessons = st.checkbox("Hide zero Lessons Accessed", value=True, key="tu_hide_zero_lessons")
            with czb:
                hide_zero_assign  = st.checkbox("Hide zero Assignments assigned", value=True, key="tu_hide_zero_assign")

            
            view = tu.copy()
            # Recompute total assignments from components (ZIP TU view)
            for _c in ["Quiz", "Worksheet", "Prasso", "Reading"]:
                if _c not in view.columns:
                    view[_c] = 0
            view["No of Assignments Assigned"] = (
                pd.to_numeric(view["Quiz"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(view["Worksheet"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(view["Prasso"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(view["Reading"], errors="coerce").fillna(0).astype(int)
)


            # (1) Hide NA in Classes Handled
            if "Classes Handled" in view.columns and hide_na_classes:
                view = view[view["Classes Handled"].astype(str).str.strip().str.upper() != "NA"]

            # Demo filters (name or id flagged earlier in build_tu_enhanced)
            if "Is Demo Teacher" in view.columns:
                if only_demo:
                    view = view[view["Is Demo Teacher"]]
                elif hide_demo:
                    view = view[~view["Is Demo Teacher"]]

            # Name contains
            if search and "Name" in view.columns:
                s = search.strip().lower()
                view = view[view["Name"].astype(str).str.lower().str.contains(s)]
                
            # apply the zero filters to the same 'view' df that drives both table and chart
            if hide_zero_lessons and "No of Lessons Accessed" in view.columns:
                view = view[pd.to_numeric(view["No of Lessons Accessed"], errors="coerce").fillna(0).astype(int) > 0]

            if hide_zero_assign and "No of Assignments Assigned" in view.columns:
                view = view[pd.to_numeric(view["No of Assignments Assigned"], errors="coerce").fillna(0).astype(int) > 0]

            # (2) Apply Active: minimum total activity from the ZIP controls to TU as well:
           
            if mode.startswith("Active"):
                lessons = pd.to_numeric(view.get("No of Lessons Accessed"), errors="coerce").fillna(0)
                assigns = pd.to_numeric(view.get("No of Assignments Assigned"), errors="coerce").fillna(0)
                tot = lessons.add(assigns, fill_value=0)
                view = view[tot > int(min_activity)]


            # (We do not apply SU's "Whitelist" to TU; it's grade-based. If you want a TU whitelist by name, say the word.)

            # ---------- Table ----------
            center_table(view, key="tu_tbl_zip")

            # ---------- Chart (3,4,5) ----------
            # Show ALL entries with dynamic height
            nrows = len(view)
            dyn_h = max(260, min(26 * max(nrows, 1) + 60, 1400))  # scale height, cap at 1400

            # Metric selector, including double-bar option
            chart_choices = [
                "No of Assignments Assigned",
                "No of Lessons Accessed",
                "No of logins",
                "Quiz", "Worksheet", "Prasso", "Reading",
                "Lessons vs Assignments (double bars)"  # (5)
            ]
            m = st.selectbox("Chart metric", chart_choices, index=0, key="tu_chart_metric")

            if m == "Lessons vs Assignments (double bars)":
                # Build grouped horizontal bars with legend for Lessons vs Assignments
                need_cols = {"No of Lessons Accessed","No of Assignments Assigned"}
                if need_cols.issubset(view.columns):
                    plot = (
                        alt.Chart(
                            view.rename(columns={
                                "No of Lessons Accessed":"Lessons",
                                "No of Assignments Assigned":"Assignments"
                            })
                            .melt(id_vars=["Name","Classes Handled"] if "Classes Handled" in view.columns else ["Name"],
                                  value_vars=["Lessons","Assignments"],
                                  var_name="Metric", value_name="Value")
                        )
                        .mark_bar()
                        .encode(
                            y=alt.Y("Name:N", sort='-x'),
                            yOffset="Metric:N",
                            x=alt.X("Value:Q", axis=alt.Axis(tickMinStep=1, format=".0f")),
                            color=alt.Color("Metric:N", legend=alt.Legend(title=None, orient="top")),
                            tooltip=["Name","Metric","Value"] + (["Classes Handled"] if "Classes Handled" in view.columns else [])
                        )
                        .properties(height=dyn_h, width=900)
                    )
                    render_altair(plot, "TU double-bar chart")
                else:
                    st.info("Lessons/Assignments columns not found for the double-bar chart.")
            else:
                # Single metric horizontal bar chart for ALL rows
                if m in view.columns:
                    plot_df = view[["Name", m]].copy()
                    plot_df[m] = pd.to_numeric(plot_df[m], errors="coerce").fillna(0).astype(int)
                    plot = (
                        alt.Chart(plot_df)
                        .mark_bar()
                        .encode(
                            y=alt.Y("Name:N", sort='-x'),
                            x=alt.X(f"{m}:Q", axis=alt.Axis(tickMinStep=1, format=".0f")),
                            tooltip=["Name", m]
                        )
                        .properties(height=dyn_h, width=900)
                    )
                    # value labels (optional; can clutter if many rows)
                    labels = alt.Chart(plot_df).mark_text(align="left", dx=3).encode(
                        y=alt.Y("Name:N", sort='-x'), x=alt.X(f"{m}:Q"),
                        text=alt.Text(f"{m}:Q", format=".0f")
                    )
                    render_altair((plot + labels).configure_axis(labelLimit=200, grid=True, gridColor="#f2f2f2"), "TU chart")
                else:
                    st.info(f"Column '{m}' not found in TU.")
            # ---------- Downloads ----------
            tu_dl = view.drop(columns=["Is Demo teacher", "Is Demo Teacher", "Is Demo"], errors="ignore")
            st.download_button("Download TU.csv",
                               data=tu_dl.to_csv(index=False).encode("utf-8-sig"),
                               file_name="TU.csv", key="tu_dl_csv.zip")
            # === Downloads (Consolidated & individual CSVs) ===

            # Pick the first non-empty DataFrame from (zip, csv); else empty
            def _pick_df(*candidates):
                for d in candidates:
                    if isinstance(d, pd.DataFrame) and not d.empty:
                        return d
                return pd.DataFrame()

            dl_su  = _pick_df(st.session_state.get("zip_su"),  st.session_state.get("csv_su"))
            dl_tu  = _pick_df(st.session_state.get("zip_tu"),  st.session_state.get("csv_tu"))
            dl_lv  = _pick_df(st.session_state.get("zip_lv"),  st.session_state.get("csv_lv"))
            dl_asr = _pick_df(st.session_state.get("zip_asr"), st.session_state.get("csv_asr"))

            # Build Consolidated.xlsx in-memory
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                # SU sheet (as is)
                (dl_su if isinstance(dl_su, pd.DataFrame) else pd.DataFrame()).to_excel(w, "SU", index=False)

                # TU sheet (drop demo cols, recompute total)
                _dl_tu = dl_tu.copy() if isinstance(dl_tu, pd.DataFrame) else pd.DataFrame()
                for _c in ["Quiz", "Worksheet", "Prasso", "Reading"]:
                    if _c not in _dl_tu.columns:
                        _dl_tu[_c] = 0
                if not _dl_tu.empty:
                    _dl_tu["No of Assignments Assigned"] = (
                        pd.to_numeric(_dl_tu["Quiz"], errors="coerce").fillna(0).astype(int)
                        + pd.to_numeric(_dl_tu["Worksheet"], errors="coerce").fillna(0).astype(int)
                        + pd.to_numeric(_dl_tu["Prasso"], errors="coerce").fillna(0).astype(int)
                        + pd.to_numeric(_dl_tu["Reading"], errors="coerce").fillna(0).astype(int)
                    )
                _dl_tu = _dl_tu.drop(columns=["Is Demo teacher","Is Demo Teacher","Is Demo"], errors="ignore")
                _dl_tu.to_excel(w, "TU", index=False)

                # Levelwise & Assignment Summary as is
                (dl_lv if isinstance(dl_lv, pd.DataFrame) else pd.DataFrame()).to_excel(w, "Levelwise", index=False)
                (dl_asr if isinstance(dl_asr, pd.DataFrame) else pd.DataFrame()).to_excel(w, "Assignment Summary", index=False)

            consolidated_bytes = xbuf.getvalue()
            st.download_button(
                "Download Consolidated.xlsx",
                data=consolidated_bytes,
                file_name="Consolidated.xlsx",
                key="dl_consolidated_xlsx_unique"
            )

            # Individual CSV downloads (match what’s in consolidated)
            c1, c2 = st.columns(2)
            with c1:
                if not dl_su.empty:
                    st.download_button(
                        "Download SU.csv",
                        data=dl_su.to_csv(index=False).encode("utf-8-sig"),
                        file_name="SU.csv",
                        key="dl_su_csv_unique"
                    )
                if not dl_lv.empty:
                    st.download_button(
                        "Download Levelwise.csv",
                        data=dl_lv.to_csv(index=False).encode("utf-8-sig"),
                        file_name="Levelwise.csv",
                        key="dl_lv_csv_unique"
                    )
            with c2:
                if not dl_tu.empty:
                    _dl_tu_csv = dl_tu.copy()
                    for _c in ["Quiz", "Worksheet", "Prasso", "Reading"]:
                        if _c not in _dl_tu_csv.columns:
                            _dl_tu_csv[_c] = 0
                    _dl_tu_csv["No of Assignments Assigned"] = (
                        pd.to_numeric(_dl_tu_csv["Quiz"], errors="coerce").fillna(0).astype(int)
                        + pd.to_numeric(_dl_tu_csv["Worksheet"], errors="coerce").fillna(0).astype(int)
                        + pd.to_numeric(_dl_tu_csv["Prasso"], errors="coerce").fillna(0).astype(int)
                        + pd.to_numeric(_dl_tu_csv["Reading"], errors="coerce").fillna(0).astype(int)
                    )
                    _dl_tu_csv = _dl_tu_csv.drop(columns=["Is Demo teacher","Is Demo Teacher","Is Demo"], errors="ignore")
                    st.download_button(
                        "Download TU.csv",
                        data=_dl_tu_csv.to_csv(index=False).encode("utf-8-sig"),
                        file_name="TU.csv",
                        key="dl_tu_csv_unique"
                    )
                if not dl_asr.empty:
                    st.download_button(
                        "Download AssignmentSummary.csv",
                        data=dl_asr.to_csv(index=False).encode("utf-8-sig"),
                        file_name="AssignmentSummary.csv",
                        key="dl_asr_csv_unique"
                    )

                try:
                    xlsx_bytes = _df_to_xlsx_bytes(view, sheet_name="TU")
                    st.download_button("Download TU.xlsx", data=xlsx_bytes,
                                       file_name="TU.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       key="tu_dl_xlsx")
                except Exception:
                    pass

    
            
    with tab3:
        st.markdown("### Levelwise preview")

        lv = st.session_state.get("zip_lv", pd.DataFrame())
        if lv is None or lv.empty:
            st.info("Drop Level Lessons Usage report(s) to populate Levelwise.")
        else:
            # keep only required columns
            cols = [c for c in ["Level", "Class", "No of Lessons Accessed"] if c in lv.columns]
            lv_view = lv[cols].copy()

            # === controls ABOVE the table ===
            show_zero = st.checkbox(
                "Show sections with 0 lessons",
                value=False,                 # default: hide zero-lesson sections
                key="lv_show_zero"
            )

            # apply the filter BEFORE rendering anything
            if not show_zero and "No of Lessons Accessed" in lv_view.columns:
                lv_view = lv_view[
                    pd.to_numeric(lv_view["No of Lessons Accessed"], errors="coerce")
                      .fillna(0).astype(int) > 0
                ]

            # === table (now reflects the filter) ===
            center_table(lv_view, key="lv_tbl_zip")

            # === one chart per Level (uses filtered lv_view) ===
            # if {"Level", "Class", "No of Lessons Accessed"}.issubset(lv_view.columns):
                # def _lvl_key(s):
                    # m = re.search(r"\d+", str(s))
                    # return (int(m.group()) if m else 999, str(s))
                # levels = sorted(lv_view["Level"].dropna().unique().tolist(), key=_lvl_key)

                # for lvl in levels:
                    # sub = lv_view[lv_view["Level"] == lvl][["Class", "No of Lessons Accessed"]].copy()
                    # if sub.empty:
                        # continue
                    # sub = add_class_sort(sub, "Class")
                    # sub["No of Lessons Accessed"] = (
                        # pd.to_numeric(sub["No of Lessons Accessed"], errors="coerce").fillna(0).astype(int)
                    # )
                    # dyn_h = max(260, min(28 * len(sub) + 80, 900))
                    # st.markdown(f"#### {lvl}")
                    # render_altair(
                        # bar_with_labels(
                            # sub, x="Class", y="No of Lessons Accessed",
                            # height=dyn_h, width=900, category_sort="_sort", bar_size=18
                        # ),
                        # f"Levelwise chart — {lvl}",
                    # )
                # table already shown above...

            # one chart per Level (uses lv_view after filters)
            if {"Level","Class","No of Lessons Accessed"}.issubset(lv_view.columns):
                def _lvl_key(s):
                    m_ = re.search(r"\d+", str(s))
                    return (int(m_.group()) if m_ else 999, str(s))
                levels = sorted(lv_view["Level"].dropna().unique().tolist(), key=_lvl_key)

                for lvl in levels:
                    sub = lv_view[lv_view["Level"] == lvl][["Class","No of Lessons Accessed"]].copy()
                    if sub.empty:
                        continue
                    sub = add_class_sort(sub, "Class")
                    sub["No of Lessons Accessed"] = pd.to_numeric(sub["No of Lessons Accessed"], errors="coerce").fillna(0).astype(int)

                    dyn_h = max(260, min(28 * len(sub) + 80, 900))   # compute INSIDE loop

                    st.markdown(f"#### {lvl}")
                    chart = bar_with_labels(
                        sub, x="Class", y="No of Lessons Accessed",
                        height=dyn_h, width=900, category_sort="_sort", bar_size=18
                    )
                    render_altair(chart, f"Levelwise chart — {lvl}")

            else:
                st.info("Levelwise needs columns: Level, Class, No of Lessons Accessed.")

            # nrows = len(tmp)
            # dyn_h = max(320, min(28 * max(nrows, 1) + 80, 1200))

            

            # Downloads (table view only)
           
            # tu_dl = view.drop(columns=["Is Demo teacher", "Is Demo Teacher", "Is Demo"], errors="ignore")
            # st.download_button("Download TU.csv",
                               # data=tu_dl.to_csv(index=False).encode("utf-8-sig"),
                               # file_name="TU.csv", key="tu_dl_csv.zip")

            
            
            try:
                xlsx_bytes = _df_to_xlsx_bytes(lv_view, sheet_name="Levelwise")
                st.download_button(
                    "Download Levelwise.xlsx", data=xlsx_bytes, file_name="Levelwise.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="lv_dl_xlsx"
                )
            except Exception:
                pass


    with tab4:
        asr = st.session_state.get("zip_asr", pd.DataFrame())
        if not asr.empty:
            # --- ASR hide-all-zero rows (CSV) ---
            asr_hide_zeros_csv = st.checkbox(
                "Hide rows where Quiz Adaptive/Standard/Test are all zero",
                value=True, key="asr_hide_zeros_csv"
            )
            asr_view = asr.copy()
            if asr_hide_zeros_csv:
                qap = pd.to_numeric(asr_view.get("Quiz Adaptive Practice", 0), errors="coerce").fillna(0).astype(int)
                qsp = pd.to_numeric(asr_view.get("Quiz Standard Practice", 0), errors="coerce").fillna(0).astype(int)
                qt  = pd.to_numeric(asr_view.get("Quiz Test", 0),             errors="coerce").fillna(0).astype(int)
                asr_view = asr_view[(qap > 0) | (qsp > 0) | (qt > 0)]

            center_table(asr_view[["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test","Total (Quiz)","ONGOING_AQC","AQC Diff (Total-ONGOING_AQC)","AQC Match?"]], key="asr_tbl_csv")

            # # --- ASR hide-all-zero rows (ZIP) ---
            # asr_hide_zeros_zip = st.checkbox(
                # "Hide rows where Quiz Adaptive/Standard/Test are all zero",
                # value=True, key="asr_hide_zeros_zip"
            # )
            # asr_view = asr.copy()
            # if asr_hide_zeros_zip:
                # qap = pd.to_numeric(asr_view.get("Quiz Adaptive Practice", 0), errors="coerce").fillna(0).astype(int)
                # qsp = pd.to_numeric(asr_view.get("Quiz Standard Practice", 0), errors="coerce").fillna(0).astype(int)
                # qt  = pd.to_numeric(asr_view.get("Quiz Test", 0),             errors="coerce").fillna(0).astype(int)
                # asr_view = asr_view[(qap > 0) | (qsp > 0) | (qt > 0)]

            # center_table(asr_view[["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test","Total (Quiz)","ONGOING_AQC","AQC Diff (Total-ONGOING_AQC)","AQC Match?"]], key="asr_tbl_zip")

            req = ["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test"]
            if all(c in asr_view.columns for c in req):
                long = add_class_sort(asr_view[req].copy(), "Class").melt(
                    id_vars=["Class","_sort"], var_name="Type", value_name="Count"
                )
                 
                long["Count"] = pd.to_numeric(long["Count"], errors="coerce").fillna(0).astype(int)

                sort_obj = alt.SortField(field="_sort", order="ascending")
                axis_q = alt.Axis(tickMinStep=1, format=".0f")

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
    demo_book_csv = st.file_uploader(
        "Upload Demo_ids_classesHandled.xlsx (optional for demo teacher/student ids)",
        type=["xlsx"], key="csv_demo_book"
    )
    sa_up   = st.file_uploader("School Assignments Usage Report (CSV)", type=["csv"], key="asr_sa_csv")
    tqa_ups = st.file_uploader("Teacher Quiz Assignment CSVs (one per class)", type=["csv"],
                               accept_multiple_files=True, key="asr_tqa_csv")

    # --- NEW: count demo-students per class for CSV flow ---
    demo_students_by_class_csv = {}
    if demo_book_csv:
        try:
            xls = pd.ExcelFile(io.BytesIO(demo_book_csv.read()))
            if "demo_stud_ids" in xls.sheet_names:
                sdf = pd.read_excel(xls, "demo_stud_ids")
                cls_col = pick_col(sdf, "Class","LEVEL_DISPLAY_NAME","LEVEL","Level","class")
                if cls_col:
                    sdf["_norm_class"] = sdf[cls_col].astype(str).map(normalize_class)
                    demo_students_by_class_csv = sdf["_norm_class"].value_counts().to_dict()
        except Exception:
            pass
    st.session_state["csv_demo_students_by_class"] = demo_students_by_class_csv

    # Build ASR (CSV) if provided
    if sa_up:
        sa_df = read_csv_flex_from_bytes(sa_up.read())
        st.session_state["csv_school_assign_df"] = sa_df
        tqa_map = {}
        for f in (tqa_ups or []):
            m = re.match(r"TeacherQuizAssignment_(.+?)_", f.name)
            klass = (m.group(1).strip() if m else f.name)
            tqa_map[normalize_class(klass)] = read_csv_flex_from_bytes(f.read())
        asr = build_asr_quiz_split(sa_df, tqa_map)
        st.session_state["csv_asr"] = asr

    submitted, mode, min_students, min_activity, whitelist = compact_options_form(prefix_key="csv")

    if submitted and a_up and l_up and g_up:
        a_df = read_csv_flex_from_bytes(a_up.read())
        l_df = read_csv_flex_from_bytes(l_up.read())
        g_df = read_csv_flex_from_bytes(g_up.read())
        t_df = read_csv_flex_from_bytes(t_up.read()) if t_up else None

        # SU from CSVs
        su = build_su(a_df, l_df, g_df)

        # Subtract demo-student counts (CSV)
        dsbc = st.session_state.get("csv_demo_students_by_class", {})
        if dsbc and not su.empty and "No of Students" in su.columns and "Class" in su.columns:
            su["_norm_class"] = su["Class"].astype(str).map(normalize_class)
            su["No of Students"] = su.apply(
                lambda r: max(0, int(r["No of Students"]) - int(dsbc.get(r["_norm_class"], 0))),
                axis=1
            )
            su.drop(columns=["_norm_class"], inplace=True)

        # SU filters
        if mode.startswith("Active"):
            su = filter_active_grades(su, min_students, min_activity)
        elif mode == "Whitelist":
            allow = {g.strip().lower() for g in whitelist.split(",") if g.strip()}
            if allow:
                su = su[su["Class"].str.lower().isin(allow)]

        # TU + Levelwise (CSV)
        tu = build_tu(t_df)
        
        # Recompute total assignments from components (CSV TU)
        for _c in ["Quiz", "Worksheet", "Prasso", "Reading"]:
            if _c not in tu.columns:
                tu[_c] = 0
        tu["No of Assignments Assigned"] = (
            pd.to_numeric(tu["Quiz"], errors="coerce").fillna(0).astype(int)
            + pd.to_numeric(tu["Worksheet"], errors="coerce").fillna(0).astype(int)
            + pd.to_numeric(tu["Prasso"], errors="coerce").fillna(0).astype(int)
            + pd.to_numeric(tu["Reading"], errors="coerce").fillna(0).astype(int)
        )

        frames = [read_csv_flex_from_bytes(f.read()) for f in (lvl_up or [])]
        lv = build_levelwise_with_assignments(
            lesson_frames=frames,
            school_assign_df=st.session_state.get("csv_school_assign_df"),
            logins_df=g_df
        )

        st.session_state["csv_su"] = su
        st.session_state["csv_tu"] = tu
        st.session_state["csv_lv"] = lv

    # ...render tables/charts using st.session_state["csv_*"] as you already do...

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
            # --- TU (CSV) zero filters ---
            cz1, cz2 = st.columns([1,1])
            with cz1:
                tu_hide_zero_lessons_csv = st.checkbox(
                    "Hide zero Lessons Accessed", value=True, key="tu_hide_zero_lessons_csv"
                )
            with cz2:
                tu_hide_zero_assign_csv = st.checkbox(
                    "Hide zero Assignments assigned", value=True, key="tu_hide_zero_assign_csv"
                )

            tu_view = tu.copy()
            if tu_hide_zero_lessons_csv and "No of Lessons Accessed" in tu_view.columns:
                tu_view = tu_view[pd.to_numeric(tu_view["No of Lessons Accessed"], errors="coerce").fillna(0).astype(int) > 0]
            if tu_hide_zero_assign_csv and "No of Assignments Assigned" in tu_view.columns:
                tu_view = tu_view[pd.to_numeric(tu_view["No of Assignments Assigned"], errors="coerce").fillna(0).astype(int) > 0]
            tu_view = tu_view.drop(
                columns=["Is Demo teacher", "Is Demo Teacher", "Is Demo"],  # tolerate name variants
                errors="ignore"
            )

            center_table(tu_view, key="tu_tbl_csv")
            t_opts=[c for c in ["No of Assignments Assigned","No of Lessons Accessed","No of logins"] if c in tu_view.columns]
            if t_opts and "Name" in tu_view.columns:
                cols_set = _cols_str_set(tu_view)
                tm_sel = st.selectbox("Metric", t_opts, index=0 if t_opts else None, key="tu_metric_csv")
                tm = _coerce_metric_key(tm_sel, cols_set)

                if tm:
                    by = (tu_view.groupby("Name")[tm].sum(numeric_only=True).sort_values(ascending=False).head(12)).reset_index()
                    by[tm] = pd.to_numeric(by[tm], errors="coerce").fillna(0).astype(int)
                    render_altair(bar_with_labels(by, x="Name", y=tm, horizontal=True, height=460, width=900), "TU chart")
                else:
                    st.info("Pick a metric available in TU to draw the chart.")
            else:
                st.info("Upload Teachers Usage (optional) and click Build to see TU.")

    
    with tab3:
        lv = st.session_state.get("zip_lv", pd.DataFrame())
        if not lv.empty:
            
            # Table stays as-is
            center_table(lv, key="lv_tbl_zip")

            # ── Chart selector: single metrics or a double-bar option ──
            chart_choice = st.selectbox(
                "Chart",
                ["No of Lessons Accessed", "Assignments Assigned", "Lessons vs Assignments (double bars)"],
                index=0,
                key="lv_chart_choice",
            )

            # dynamic height so all classes fit comfortably
            nrows = len(lv)
            dyn_h = max(320, min(28 * max(nrows, 1) + 80, 1200))

            if chart_choice == "Lessons vs Assignments (double bars)":
                need = {"Class", "No of Lessons Accessed", "Assignments Assigned"}
                if need.issubset(lv.columns):
                    tmp = lv[["Class", "No of Lessons Accessed", "Assignments Assigned"]].copy()
                    tmp = add_class_sort(tmp, "Class")  # adds _sort for natural class order

                    # reshape for grouped bars
                    mdf = tmp.melt(
                        id_vars=["Class", "_sort"],
                        value_vars=["No of Lessons Accessed", "Assignments Assigned"],
                        var_name="Metric",
                        value_name="Value",
                    )
                    mdf["Value"] = pd.to_numeric(mdf["Value"], errors="coerce").fillna(0).astype(int)

                    # grouped vertical bars with legend & tooltip
                    chart = (
                        alt.Chart(mdf)
                        .mark_bar()
                        .encode(
                            x=alt.X("Class:N", sort=alt.SortField(field="_sort", order="ascending")),
                            xOffset="Metric:N",
                            y=alt.Y("Value:Q", axis=alt.Axis(tickMinStep=1, format=".0f")),
                            color=alt.Color("Metric:N", legend=alt.Legend(title=None, orient="top")),
                            tooltip=["Class", "Metric", "Value"],
                        )
                        .properties(height=dyn_h, width=900)
                    )
                    render_altair(chart, "Levelwise double-bar chart")
                else:
                    missing = need - set(lv.columns)
                    st.info(f"Missing column(s) for double-bar chart: {', '.join(missing)}")
            else:
                # single-metric chart (your existing style)
                metric = chart_choice  # either Lessons or Assignments
                if {"Class", metric}.issubset(lv.columns):
                    tmp = lv[["Class", metric]].copy()
                    tmp = add_class_sort(tmp, "Class")
                    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").fillna(0).astype(int)
                    render_altair(
                        bar_with_labels(tmp, x="Class", y=metric, height=dyn_h, width=900, category_sort="_sort", bar_size=18),
                        f"Levelwise chart ({metric})",
                    )
                else:
                    st.info(f"Column '{metric}' not found in Levelwise.")

        else:
            st.info("No Levelwise files found.")


    with tab4:
        if not asr.empty:
            center_table(asr[["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test","Total (Quiz)","ONGOING_AQC","AQC Diff (Total-ONGOING_AQC)","AQC Match?"]], key="asr_tbl_csv")
            req = ["Class","Quiz Adaptive Practice","Quiz Standard Practice","Quiz Test"]
            if all(c in asr_view.columns for c in req):
                long = add_class_sort(asr_view[req].copy(), "Class").melt(
                    id_vars=["Class","_sort"], var_name="Type", value_name="Count"
                )
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
    # st.subheader("Downloads")
    # dl_su  = st.session_state.get("zip_su",  st.session_state.get("csv_su",  pd.DataFrame()))
    # # Build TU sheet without demo column and with recomputed total
    # _dl_tu = dl_tu.copy() if isinstance(dl_tu, pd.DataFrame) else pd.DataFrame()
    # for _c in ["Quiz", "Worksheet", "Prasso", "Reading"]:
        # if _c not in _dl_tu.columns:
            # _dl_tu[_c] = 0
    # _dl_tu["No of Assignments Assigned"] = (
        # pd.to_numeric(_dl_tu["Quiz"], errors="coerce").fillna(0).astype(int)
        # + pd.to_numeric(_dl_tu["Worksheet"], errors="coerce").fillna(0).astype(int)
        # + pd.to_numeric(_dl_tu["Prasso"], errors="coerce").fillna(0).astype(int)
        # + pd.to_numeric(_dl_tu["Reading"], errors="coerce").fillna(0).astype(int)
    # )
    # _dl_tu = _dl_tu.drop(columns=["Is Demo teacher", "Is Demo Teacher", "Is Demo"], errors="ignore")
    # _dl_tu.to_excel(w, "TU", index=False)

    # dl_tu  = st.session_state.get("zip_tu",  st.session_state.get("csv_tu",  pd.DataFrame()))
    # dl_lv  = st.session_state.get("zip_lv",  st.session_state.get("csv_lv",  pd.DataFrame()))
    # dl_asr = st.session_state.get("zip_asr", st.session_state.get("csv_asr", pd.DataFrame()))

    # xbuf=io.BytesIO()
    # with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
        # (dl_su if isinstance(dl_su, pd.DataFrame) else pd.DataFrame()).to_excel(w, "SU", index=False)
        # (dl_tu if isinstance(dl_tu, pd.DataFrame) else pd.DataFrame()).to_excel(w, "TU", index=False)
        # (dl_lv if isinstance(dl_lv, pd.DataFrame) else pd.DataFrame()).to_excel(w, "Levelwise", index=False)
        # (dl_asr if isinstance(dl_asr, pd.DataFrame) else pd.DataFrame()).to_excel(w, "Assignment Summary", index=False)
    # xbuf.seek(0)
    # st.download_button("Download Consolidated xlsx", data=xbuf.getvalue(),
                       # file_name="Consolidated.xlsx",
                       # mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    # Build TU sheet without demo column and with recomputed total
    # Gather latest frames for downloads (prefer ZIP, fall back to CSV, else empty)
    # dl_su  = st.session_state.get("zip_su")  or st.session_state.get("csv_su")  or pd.DataFrame()
    # dl_tu  = st.session_state.get("zip_tu")  or st.session_state.get("csv_tu")  or pd.DataFrame()
    # dl_lv  = st.session_state.get("zip_lv")  or st.session_state.get("csv_lv")  or pd.DataFrame()
    # dl_asr = st.session_state.get("zip_asr") or st.session_state.get("csv_asr") or pd.DataFrame()
    
    # --- Ensure download DataFrames exist ---
    def _pick_df(*candidates):
        for d in candidates:
            if isinstance(d, pd.DataFrame) and not d.empty:
                return d
        return pd.DataFrame()

    dl_su  = _pick_df(st.session_state.get("zip_su"),  st.session_state.get("csv_su"))
    dl_tu  = _pick_df(st.session_state.get("zip_tu"),  st.session_state.get("csv_tu"))
    dl_lv  = _pick_df(st.session_state.get("zip_lv"),  st.session_state.get("csv_lv"))
    dl_asr = _pick_df(st.session_state.get("zip_asr"), st.session_state.get("csv_asr"))


    
    _dl_tu = dl_tu.copy() if isinstance(dl_tu, pd.DataFrame) else pd.DataFrame()
    for _c in ["Quiz", "Worksheet", "Prasso", "Reading"]:
        if _c not in _dl_tu.columns:
            _dl_tu[_c] = 0
    _dl_tu["No of Assignments Assigned"] = (
        pd.to_numeric(_dl_tu["Quiz"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(_dl_tu["Worksheet"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(_dl_tu["Prasso"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(_dl_tu["Reading"], errors="coerce").fillna(0).astype(int)
    )
    _dl_tu = _dl_tu.drop(columns=["Is Demo teacher", "Is Demo Teacher", "Is Demo"], errors="ignore")
    _dl_tu.to_excel(w, "TU", index=False)

    c1,c2 = st.columns(2)
    with c1:
        if not dl_su.empty:  st.download_button("Download SU.csv", data=dl_su.to_csv(index=False).encode("utf-8-sig"), file_name="SU.csv")
        if not dl_lv.empty:  st.download_button("Download Levelwise.csv", data=dl_lv.to_csv(index=False).encode("utf-8-sig"), file_name="Levelwise.csv")
    with c2:
        # if not dl_tu.empty:  st.download_button("Download TU.csv", data=dl_tu.to_csv(index=False).encode("utf-8-sig"), file_name="TU.csv")
        if not dl_tu.empty:
            _dl_tu_csv = dl_tu.copy()
            # optional: recompute the total for the global CSV too
            for _c in ["Quiz","Worksheet","Prasso","Reading"]:
                if _dl_tu_csv.get(_c) is None:
                    _dl_tu_csv[_c] = 0
            _dl_tu_csv["No of Assignments Assigned"] = (
                pd.to_numeric(_dl_tu_csv["Quiz"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(_dl_tu_csv["Worksheet"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(_dl_tu_csv["Prasso"], errors="coerce").fillna(0).astype(int)
                + pd.to_numeric(_dl_tu_csv["Reading"], errors="coerce").fillna(0).astype(int)
            )
            _dl_tu_csv = _dl_tu_csv.drop(columns=["Is Demo teacher","Is Demo Teacher","Is Demo"], errors="ignore")

            st.download_button("Download TU.csv",
                               data=_dl_tu_csv.to_csv(index=False).encode("utf-8-sig"),
                               file_name="TU.csv",  key="dl_tu_csv_global")
        
        
        if not dl_asr.empty: st.download_button("Download AssignmentSummary.csv", data=dl_asr.to_csv(index=False).encode("utf-8-sig"), file_name="AssignmentSummary.csv")
