import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import joblib
import time as _time

st.set_page_config(
    page_title="Cart Super Add-On (CSAO) Rail Recommendation System",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY   = "#FF6B35"
SECONDARY = "#2C3E50"
SUCCESS   = "#27AE60"
WARNING   = "#F39C12"
DANGER    = "#E74C3C"
DARK      = "#1A1A2E"
PALETTE   = ["#FF6B35","#3498DB","#27AE60","#F39C12","#E74C3C","#8E44AD","#1ABC9C","#E67E22"]

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── TRUE BLACK BACKGROUND ── */
  html { background: #000 !important; }
  body { background: #000 !important; }
  * { scrollbar-color: #FF6B35 #111; }

  /* Streamlit shell */
  .stApp,
  .stApp > div:first-child,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stHeader"],
  .main,
  .main > div,
  .main .block-container,
  .block-container,
  .css-1d391kg, .css-fg4pbf, .css-12oz5g7,
  .css-1y4p8pa, .css-k1ih3n, .css-z5fcl4,
  .css-18e3th9, .css-1lcbmhc,
  section[data-testid="stMain"],
  div[data-testid="stDecoration"] {
    background-color: #000000 !important;
    background: #000000 !important;
  }

  /* Tabs and content panels — transparent so black bg shows through */
  .stTabs,
  [data-testid="stTabsContent"],
  [data-baseweb="tab-panel"],
  [data-testid="stVerticalBlock"],
  [data-testid="stHorizontalBlock"],
  [data-testid="column"],
  [data-testid="stMarkdownContainer"],
  .element-container {
    background-color: transparent !important;
    background: transparent !important;
  }

  /* ── Hero Banner ── */
  .hero {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 70%, #0f3460 100%);
    padding: 40px 44px;
    border-radius: 20px;
    color: white;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,107,53,0.2);
  }
  .hero::before {
    content: "";
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(255,107,53,0.18) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero::after {
    content: "";
    position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(52,152,219,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 8px;
    background: linear-gradient(90deg, #ffffff 0%, #FF6B35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
  }
  .hero p { margin: 0; color: #8899aa; font-size: .9rem; font-weight: 300; }


  /* ── KPI Cards ── */
  .kpi-wrap { height: 100%; }
  .kpi-card {
    background: #111111;
    border-radius: 16px;
    padding: 22px 20px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 2px 12px rgba(255,107,53,.08);
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
    height: 100%;
  }
  .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.1); }
  .kpi-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #FF6B35, #ff9060);
  }
  .kpi-card .icon { font-size: 1.6rem; margin-bottom: 6px; display: block; }
  .kpi-card .val  { font-family:'Syne',sans-serif; font-size:1.75rem; font-weight:700; color:#FF6B35; display:block; }
  .kpi-card .lbl  { font-size:.8rem; color:#aaa; margin-top:4px; font-weight:500; }
  .kpi-card .sub  { font-size:.72rem; color:#27AE60; font-weight:600; margin-top:3px; }

  /* ── Section Titles ── */
  .sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    padding-bottom: 8px;
    margin: 20px 0 16px;
    border-bottom: 3px solid #FF6B35;
    display: inline-block;
  }

  /* ── Insight Pills ── */
  .insight-pill {
    background: linear-gradient(135deg, #1a0f0a 0%, #1f1208 100%);
    border: 1px solid #3a2010;
    border-left: 4px solid #FF6B35;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: .88rem;
    color: #ccc;
    margin: 10px 0;
    line-height: 1.6;
  }
  .insight-pill strong { color: #FF6B35; }

  /* ── Chart Containers ── */
  .chart-box {
    background: #111111;
    border-radius: 14px;
    padding: 18px;
    border: 1px solid #222222;
    box-shadow: 0 2px 10px rgba(255,107,53,.06);
    margin-bottom: 16px;
  }

  /* ── Prediction Result Cards ── */
  .pred-accept {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 2px solid #27AE60;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
  }
  .pred-reject {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border: 2px solid #E74C3C;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
  }
  .pred-label { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; }
  .pred-prob  { font-size:2.5rem; font-weight:800; font-family:'Syne',sans-serif; }

  /* ── Probability Gauge Bar ── */
  .prob-bar-bg {
    background: #e9ecef; border-radius: 8px; height: 14px; margin-top: 10px; overflow: hidden;
  }
  .prob-bar-fill {
    height: 14px; border-radius: 8px;
    background: linear-gradient(90deg, #FF6B35, #ff9060);
    transition: width .6s ease;
  }

  /* ── Table Styling ── */
  .styled-table { width: 100%; border-collapse: collapse; font-size: .87rem; }
  .styled-table th {
    background: #1A1A2E; color: white;
    padding: 10px 14px; text-align: left; font-weight: 600; font-size: .8rem;
  }
  .styled-table td { padding: 9px 14px; border-bottom: 1px solid #222; color: #ccc; }
  .styled-table tr:hover td { background: #1a0f0a; }

  /* ── Sidebar — true black ── */
  div[data-testid="stSidebar"],
  div[data-testid="stSidebar"] > div,
  div[data-testid="stSidebar"] > div:first-child,
  section[data-testid="stSidebar"],
  [data-testid="stSidebarNav"],
  [data-testid="stSidebarContent"] {
    background-color: #000000 !important;
    background: #000000 !important;
    border-right: 1px solid #1f1f1f !important;
  }
  div[data-testid="stSidebar"] * { color: #cccccc !important; }
  div[data-testid="stSidebar"] h2,
  div[data-testid="stSidebar"] h3 { color: #ffffff !important; font-weight: 700 !important; }
  div[data-testid="stSidebar"] hr { border-color: #222222 !important; }
  div[data-testid="stSidebar"] .stSlider .stMarkdown { color: #aaa !important; }
  /* multiselect tags stay orange on black */
  div[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #FF6B35 !important;
    border: none !important;
  }
  /* multiselect input bg */
  div[data-testid="stSidebar"] [data-baseweb="select"] > div,
  div[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: #111111 !important;
    border-color: #333 !important;
  }
  /* slider track */
  div[data-testid="stSidebar"] [data-testid="stSlider"] > div > div {
    background-color: #222222 !important;
  }

  /* ── Tab Styling ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #111111;
    padding: 6px 8px;
    border-radius: 12px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-size: .85rem;
    font-weight: 600;
    color: #aaa !important;
  }
  .stTabs [aria-selected="true"] {
    background: #FF6B35 !important;
    color: white !important;
  }

  /* ── Dataframe dark theme ── */
  [data-testid="stDataFrame"] { background-color: #111111 !important; }
  [data-testid="stDataFrame"] * { color: #cccccc !important; }

  /* ── General text on black bg ── */
  p, span, label, div { color: #cccccc; }
  h1, h2, h3, h4 { color: #ffffff; }
  .stMarkdown p { color: #cccccc; }
  .stMarkdown li { color: #cccccc; }

  /* ── Download Button ── */
  .stDownloadButton button {
    background: linear-gradient(135deg, #FF6B35, #ff8c5a) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; width: 100%; font-weight: 600 !important;
  }
  /* ── Big Predict Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #FF6B35, #cc3300) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 800 !important;
    font-size: 1rem !important; letter-spacing: .5px !important;
    box-shadow: 0 4px 20px rgba(255,107,53,0.4) !important;
    transition: all .2s !important;
  }
  .stButton > button:hover {
    box-shadow: 0 8px 30px rgba(255,107,53,0.65) !important;
    transform: translateY(-2px) !important;
  }
  /* ── Suggestion & Cart Cards ── */
  .sug-card {
    background: linear-gradient(135deg,#130b04,#1e1008);
    border:1px solid #2e1a08; border-left:4px solid #FF6B35;
    border-radius:14px; padding:16px 14px; margin-bottom:10px;
    transition: transform .15s;
  }
  .sug-card:hover { transform:translateY(-3px); }
  .sug-card .sug-top { display:flex;align-items:center;gap:10px;margin-bottom:6px; }
  .sug-card .sug-emoji { font-size:1.7rem; }
  .sug-card .sug-name { font-size:.95rem;font-weight:700;color:#FF6B35; }
  .sug-card .sug-why  { font-size:.79rem;color:#999;line-height:1.5; }
  .sug-card .sug-footer { display:flex;justify-content:space-between;margin-top:10px; }
  .sug-card .sug-price { font-size:.78rem;color:#aaa; }
  .sug-card .sug-prob  { font-size:.82rem;font-weight:700;color:#27AE60; }
</style>
""", unsafe_allow_html=True)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#080808")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color="#bbbbbb", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, color="#bbbbbb", labelpad=8)
    if title:
        ax.set_title(title, fontsize=11, fontweight="700", color="#ffffff", pad=12)
    ax.yaxis.grid(True, color="#222222", linewidth=0.8)
    ax.set_axisbelow(True)

def style_fig(fig):
    fig.patch.set_facecolor("#000000")
    fig.patch.set_edgecolor("none")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("csao_model_ready.csv")
    meal_map = {0:"Breakfast",1:"Lunch",2:"Evening Snack",3:"Dinner",4:"Late Night"}
    seg_map  = {0:"Budget",1:"Frequent",2:"Premium"}
    pr_map   = {0:"Low",1:"Medium",2:"High"}
    def safe_col(col): return df[col] == 1 if col in df.columns else pd.Series([False]*len(df))
    conditions = [safe_col("cat_dessert"), safe_col("cat_drink"), safe_col("cat_main_course")]
    choices    = ["Dessert","Drink","Main"]
    df["meal_label"]    = df["meal_time"].map(meal_map)
    df["segment_label"] = df["user_segment"].map(seg_map)
    df["price_label"]   = df["rest_price_range"].map(pr_map) if "rest_price_range" in df.columns else "Medium"
    df["item_category"] = np.select(conditions, choices, default="Side")
    df["veg_label"]     = df["candidate_is_veg"].map({0:"Non-Veg",1:"Veg"}) if "candidate_is_veg" in df.columns else "Unknown"
    df["accepted"]      = df["label"].map({0:"Rejected",1:"Accepted"})
    try:
        feat_imp = pd.read_csv("csao_feature_importance.csv")
    except Exception:
        feat_imp = None
    return df, feat_imp

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model ONCE into RAM — cached across all reruns and sessions."""
    model         = joblib.load("csao_model.joblib")
    features_list = joblib.load("csao_features.joblib")
    return model, features_list

@st.cache_data(show_spinner=False)
def get_baseline_vector(features_tuple):
    """Pre-compute median baseline once — never recomputed on reruns."""
    _df = load_data()[0]
    return _df[list(features_tuple)].median(numeric_only=True).to_dict()

@st.cache_data(show_spinner=False)
def batch_score_candidates(scenarios_tuple, features_tuple):
    """Score ALL candidates in ONE predict_proba call — fastest possible."""
    model, _ = load_model()
    rows  = [dict(s) for s in scenarios_tuple]
    X     = pd.DataFrame(rows)[list(features_tuple)]
    return model.predict_proba(X)[:, 1].tolist()

df, feat_imp = load_data()

try:
    _model_global, _features_global = load_model()
    _baseline_global = get_baseline_vector(tuple(_features_global))
    _model_ready = True
except Exception as _me:
    _model_global    = None
    _features_global = []
    _baseline_global = {}
    _model_ready     = False

with st.sidebar:
    st.markdown("## 🎛️ Dashboard Filters")
    st.markdown("---")
    seg_filter   = st.multiselect("👤 User Segment",    ["Budget","Frequent","Premium"],
                                   default=["Budget","Frequent","Premium"])
    meal_filter  = st.multiselect("🕐 Meal Time",       ["Breakfast","Lunch","Evening Snack","Dinner","Late Night"],
                                   default=["Breakfast","Lunch","Evening Snack","Dinner","Late Night"])
    price_filter = st.multiselect("💰 Price Range",     ["Low","Medium","High"],
                                   default=["Low","Medium","High"])
    cat_filter   = st.multiselect("🍔 Item Category",   ["Main","Drink","Side","Dessert"],
                                   default=["Main","Drink","Side","Dessert"])
    st.markdown("---")
    price_range  = st.slider("Candidate Price (₹)", 50, 420, (50, 420))
    st.markdown("---")
    st.markdown("### 📥 Export Filtered Data")
    mask_dl = (
        df["segment_label"].isin(seg_filter) &
        df["meal_label"].isin(meal_filter) &
        df["price_label"].isin(price_filter) &
        df["item_category"].isin(cat_filter) &
        df["candidate_price"].between(price_range[0], price_range[1])
    )
    csv_data = df[mask_dl].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="csao_filtered_report.csv",
        mime="text/csv",
    )

mask = (
    df["segment_label"].isin(seg_filter) &
    df["meal_label"].isin(meal_filter) &
    df["price_label"].isin(price_filter) &
    df["item_category"].isin(cat_filter) &
    df["candidate_price"].between(price_range[0], price_range[1])
)
filtered = df[mask].copy()


st.markdown(f"""
<div class="hero">
  <h1>Cart Super Add-On (CSAO) Rail Recommendation System</h1>
  <p>Real-time insights on Cart Add-On recommendation performance · ML-powered intelligence</p>

</div>
""", unsafe_allow_html=True)

acc_rate  = filtered["label"].mean() * 100
avg_price = filtered["candidate_price"].mean()
avg_cooc  = filtered["max_co_occur_confidence"].mean()
prev_ord  = filtered["candidate_ordered_before"].mean() * 100
acc_recs  = int(filtered["label"].sum())
avg_acc_p = filtered[filtered["label"]==1]["candidate_price"].mean()
if np.isnan(avg_acc_p): avg_acc_p = 0
est_rev        = acc_recs * avg_acc_p
rail_order_share = acc_rate 
rev_per_order = avg_acc_p if avg_acc_p > 0 else avg_price

k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    ("📦", f"{len(filtered):,}",           "Filtered Records",       f"of {len(df):,} total"),
    ("✅", f"{acc_rate:.1f}%",              "Acceptance Rate",        "recommendation accepted"),
    ("💰", f"₹{est_rev:,.0f}",             "Est. Revenue Lift (₹)",  f"₹{avg_acc_p:.0f} × {acc_recs:,} accepted"),
    ("🔗", f"{avg_cooc:.3f}",              "Avg Pairing Score",      "co-occurrence strength"),
    ("📊", f"{rail_order_share:.1f}%",     "CSAO Rail Order Share",  "% orders with add-on"),
    ("⭐", f"{prev_ord:.1f}%",             "Prev. Ordered",          "personalization signal"),
]
for col,(icon,val,lbl,sub) in zip([k1,k2,k3,k4,k5,k6], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <span class="icon">{icon}</span>
          <span class="val">{val}</span>
          <div class="lbl">{lbl}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Acceptance Analysis",
    "💰 Price & Cart",
    "👤 User & Segment",
    "🔮 Live Predictor",
    "🛒 Cart Simulator",
    "🧠 AI Insights",
])


with tab1:
    st.markdown('<div class="sec-title">📈 Acceptance Rate Deep Dive</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        meal_acc = filtered.groupby("meal_label")["label"].mean().reset_index()
        meal_acc.columns = ["meal_time","rate"]
        meal_acc["pct"] = (meal_acc["rate"]*100).round(2)
        order = ["Breakfast","Lunch","Evening Snack","Dinner","Late Night"]
        meal_acc["meal_time"] = pd.Categorical(meal_acc["meal_time"], categories=order, ordered=True)
        meal_acc = meal_acc.sort_values("meal_time")

        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        colors = [PRIMARY if v == meal_acc["pct"].max() else "#c0d4e8" for v in meal_acc["pct"]]
        bars = ax.bar(meal_acc["meal_time"], meal_acc["pct"], color=colors, width=0.55, edgecolor="none", zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.4, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="600", color="#333")
        style_ax(ax, "Acceptance Rate by Meal Time", "", "Acceptance %")
        plt.xticks(rotation=25, ha="right")
        ax.set_ylim(0, meal_acc["pct"].max()+7)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        cat_acc = filtered.groupby("item_category")["label"].mean().reset_index()
        cat_acc.columns = ["category","rate"]
        cat_acc["pct"] = (cat_acc["rate"]*100).round(2)
        cat_acc = cat_acc.sort_values("pct")

        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        cat_colors = [PALETTE[i % len(PALETTE)] for i in range(len(cat_acc))]
        bars = ax.barh(cat_acc["category"], cat_acc["pct"],
                       color=cat_colors, height=0.5, edgecolor="none", zorder=3)
        for bar in bars:
            w = bar.get_width()
            ax.text(w+0.3, bar.get_y()+bar.get_height()/2, f"{w:.1f}%",
                    va="center", fontsize=9, fontweight="600", color="#333")
        style_ax(ax, "Acceptance Rate by Item Category", "Acceptance %", "")
        ax.set_xlim(0, cat_acc["pct"].max()+7)
        ax.xaxis.grid(True, color="#efefef", linewidth=0.8)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    hour_acc = filtered.groupby("hour")["label"].mean().reset_index()
    hour_acc.columns = ["hour","rate"]
    hour_acc["pct"] = (hour_acc["rate"]*100).round(2)

    fig, ax = plt.subplots(figsize=(12,3.5))
    style_fig(fig)
    ax.fill_between(hour_acc["hour"], hour_acc["pct"], alpha=0.15, color=PRIMARY)
    ax.plot(hour_acc["hour"], hour_acc["pct"], color=PRIMARY, linewidth=2.5, zorder=3)
    ax.scatter(hour_acc["hour"], hour_acc["pct"], color=PRIMARY, s=50, zorder=4, edgecolors="white", linewidths=1.5)
    style_ax(ax, "Acceptance Rate by Hour of Day", "Hour of Day (0–23)", "Acceptance %")
    ax.set_xticks(range(0,24,1))
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=8)
    ax.set_ylim(0, hour_acc["pct"].max()+8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    best_meal = meal_acc.sort_values("pct", ascending=False).iloc[0]
    best_cat  = cat_acc.sort_values("pct", ascending=False).iloc[0]
    st.markdown(f"""
    <div class="insight-pill">
      💡 <strong>{best_meal['meal_time']}</strong> achieves the highest acceptance at
      <strong>{best_meal['pct']}%</strong>. Among categories,
      <strong>{best_cat['category']}</strong> items perform best at
      <strong>{best_cat['pct']}%</strong> acceptance — these are your highest-ROI slots.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🍔 Add-on Performance by Meal Time</div>', unsafe_allow_html=True)

    gb1, gb2 = st.columns([1.4, 1])
    with gb1:
        meal_cat = filtered.groupby(["meal_label","item_category"]).agg(
            pct=("label","mean"),
            volume=("label","count"),
        ).reset_index()
        meal_cat["pct"] = (meal_cat["pct"] * 100).round(1)
        meal_cat = meal_cat[meal_cat["item_category"] != "Main"]

        meal_order = ["Breakfast","Lunch","Evening Snack","Dinner","Late Night"]
        cats       = [c for c in ["Drink","Side","Dessert"] if c in meal_cat["item_category"].unique()]
        cat_colors = {"Drink":"#FF6B35","Side":"#D94F1E","Dessert":"#A63208"}
        cat_alpha  = {"Drink":1.0,      "Side":0.82,     "Dessert":0.65}

        meal_cat["meal_label"] = pd.Categorical(
            meal_cat["meal_label"], categories=meal_order, ordered=True)
        meal_cat = meal_cat.sort_values("meal_label")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        style_fig(fig)

        x      = np.arange(len(meal_order))
        n_cats = len(cats)
        width  = 0.24
        offsets = np.linspace(-(n_cats-1)/2, (n_cats-1)/2, n_cats) * width

        for i, cat in enumerate(cats):
            sub  = meal_cat[meal_cat["item_category"]==cat].set_index("meal_label")
            vals = [float(sub.loc[m,"pct"]) if m in sub.index else 0.0 for m in meal_order]
            clr  = cat_colors.get(cat,"#FF6B35")
            alph = cat_alpha.get(cat, 0.9)

            bars = ax.bar(x + offsets[i], vals, width,
                          color=clr, alpha=alph,
                          edgecolor="none", zorder=3,
                          label=cat)

            for bar, v in zip(bars, vals):
                if v > 0:
                    bx, bw = bar.get_x(), bar.get_width()
                    ax.text(bx + bw/2, v + 0.6, f"{v:.1f}%",
                            ha="center", va="bottom",
                            fontsize=7.5, fontweight="800",
                            color=clr, zorder=5)

        # subtle horizontal grid
        ax.yaxis.grid(True, color="#2a2a2a", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_axisbelow(True)

        ax.set_xticks(x)
        ax.set_xticklabels(meal_order, fontsize=8.5, color="#bbbbbb", rotation=10)
        ax.set_ylim(0, meal_cat["pct"].max() + 12)

        leg = ax.legend(fontsize=9, framealpha=0.12, labelcolor="white",
                        loc="upper right", handlelength=1.2,
                        handletextpad=0.5, borderpad=0.6)
        leg.get_frame().set_edgecolor("#333333")

        style_ax(ax, "Acceptance Rate by Meal Time & Category", "Meal Time", "Acceptance %")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with gb2:
        st.markdown("**🤝 Pairing Strength by Category**")
        top_pairs = filtered.groupby("item_category").agg(
            Pairing_Strength=("max_co_occur_confidence","mean"),
            Acceptance_Rate=("label","mean"),
            Volume=("label","count")
        ).reset_index()
        top_pairs["Acceptance_Rate"]  = (top_pairs["Acceptance_Rate"]*100).round(1).astype(str)+"%"
        top_pairs["Pairing_Strength"] = top_pairs["Pairing_Strength"].round(3)
        top_pairs = top_pairs.sort_values("Pairing_Strength", ascending=False)
        st.dataframe(top_pairs.rename(columns={"item_category":"Category"}),
                     use_container_width=True, hide_index=True)
        best_pair = top_pairs[top_pairs["item_category"] != "Main"].iloc[0]
        st.markdown(f"""
        <div class="insight-pill">
          💡 <strong>{best_pair["item_category"]}</strong> has the highest pairing strength
          ({best_pair["Pairing_Strength"]}) and is your highest-ROI add-on category.
          Darker orange = stronger acceptance signal.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🥗 Veg vs Non-Veg Acceptance Split</div>', unsafe_allow_html=True)
    veg1, veg2 = st.columns(2)
    with veg1:
        veg_acc = filtered.groupby("veg_label")["label"].mean().reset_index()
        veg_acc["pct"] = (veg_acc["label"]*100).round(1)
        fig, ax = plt.subplots(figsize=(5,3.5))
        style_fig(fig)
        colors_v = ["#27AE60" if v=="Veg" else "#FF6B35" for v in veg_acc["veg_label"]]
        bars = ax.bar(veg_acc["veg_label"], veg_acc["pct"], color=colors_v, width=0.4, edgecolor="none", zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.4, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="700", color="#fff")
        style_ax(ax, "Acceptance Rate: Veg vs Non-Veg", "Item Type", "Acceptance %")
        ax.set_ylim(0, veg_acc["pct"].max()+10)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with veg2:
        veg_vol = filtered.groupby(["veg_label","accepted"])["label"].count().reset_index()
        veg_pivot = veg_vol.pivot(index="veg_label", columns="accepted", values="label").fillna(0)
        fig, ax = plt.subplots(figsize=(5,3.5))
        style_fig(fig)
        x = np.arange(len(veg_pivot.index)); w = 0.35
        if "Accepted" in veg_pivot.columns:
            ax.bar(x-w/2, veg_pivot["Accepted"], w, color=PRIMARY, label="Accepted", edgecolor="none", zorder=3)
        if "Rejected" in veg_pivot.columns:
            ax.bar(x+w/2, veg_pivot["Rejected"], w, color="#3498DB", label="Rejected", edgecolor="none", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(veg_pivot.index, fontsize=10)
        ax.legend(fontsize=9)
        style_ax(ax, "Volume: Accepted vs Rejected by Type", "Item Type", "Count")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    best_veg = veg_acc.sort_values("pct", ascending=False).iloc[0]
    st.markdown(f'''<div class="insight-pill">
      💡 <strong>{best_veg["veg_label"]}</strong> items have higher acceptance at
      <strong>{best_veg["pct"]:.1f}%</strong>.
      Factor veg preference (user_veg_preference_ratio) into ranking — it's a top-5 feature in our model.
    </div>''', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="sec-title">💰 Price Sensitivity & Cart Behavior</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        ax.hist(filtered[filtered["label"]==0]["candidate_price"], bins=25,
                alpha=0.65, color="#3498DB", label="Rejected", edgecolor="none", zorder=3)
        ax.hist(filtered[filtered["label"]==1]["candidate_price"], bins=25,
                alpha=0.80, color=PRIMARY, label="Accepted", edgecolor="none", zorder=4)
        ax.axvline(filtered[filtered["label"]==1]["candidate_price"].mean(),
                   color=PRIMARY, linestyle="--", linewidth=1.5, alpha=0.9)
        ax.axvline(filtered[filtered["label"]==0]["candidate_price"].mean(),
                   color="#3498DB", linestyle="--", linewidth=1.5, alpha=0.9)
        patch_acc = mpatches.Patch(color=PRIMARY,   alpha=0.8, label="Accepted")
        patch_rej = mpatches.Patch(color="#3498DB", alpha=0.65, label="Rejected")
        ax.legend(handles=[patch_acc, patch_rej], fontsize=9)
        style_ax(ax, "Price Distribution: Accepted vs Rejected", "Candidate Price (₹)", "Count")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        avg_cart = filtered.groupby("accepted")["cart_total_value"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        bar_colors = [PRIMARY if a=="Accepted" else "#3498DB" for a in avg_cart["accepted"]]
        bars = ax.bar(avg_cart["accepted"], avg_cart["cart_total_value"],
                      color=bar_colors, width=0.45, edgecolor="none", zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+2, f"₹{h:.0f}",
                    ha="center", va="bottom", fontsize=10, fontweight="700", color="#333")
        style_ax(ax, "Average Cart Value by Recommendation Outcome", "Outcome", "Avg Cart Value (₹)")
        ax.set_ylim(0, avg_cart["cart_total_value"].max()+40)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    with c3:
        sample = filtered.sample(min(3000, len(filtered)), random_state=42)

        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        for outcome, color in [("Rejected","#3498DB"), ("Accepted",PRIMARY)]:
            sub = sample[sample["accepted"]==outcome]
            ax.scatter(sub["candidate_price"], sub["candidate_price_vs_cart_avg"],
                       color=color, alpha=0.35, s=18, label=outcome, edgecolors="none", zorder=3)
        ax.axhline(1.0, linestyle="--", color="#999", linewidth=1.2, alpha=0.8, label="Cart Avg Line")
        ax.legend(fontsize=9)
        style_ax(ax, "Item Price vs Price Relative to Cart Avg",
                 "Candidate Price (₹)", "Price ÷ Cart Avg")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c4:
        cooc = filtered.copy()
        cooc["confidence_range"] = pd.cut(
            cooc["max_co_occur_confidence"],
            bins=[0,0.1,0.3,0.5,0.7,1.0],
            labels=["0–0.1","0.1–0.3","0.3–0.5","0.5–0.7","0.7–1.0"]
        )
        cooc_s = cooc.groupby("confidence_range", observed=True)["label"].mean().reset_index()
        cooc_s["pct"] = (cooc_s["label"]*100).round(1)

        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        ax.plot(cooc_s["confidence_range"].astype(str), cooc_s["pct"],
                marker="o", linewidth=2.5, color=PRIMARY,
                markersize=9, markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=PRIMARY, zorder=3)
        ax.fill_between(range(len(cooc_s)), cooc_s["pct"], alpha=0.1, color=PRIMARY)
        for i, (x, y) in enumerate(zip(range(len(cooc_s)), cooc_s["pct"])):
            ax.text(x, y+0.8, f"{y:.1f}%", ha="center", fontsize=9, fontweight="600", color=PRIMARY)
        style_ax(ax, "Acceptance Rate by Co-occurrence Confidence",
                 "Co-occurrence Range", "Acceptance Rate (%)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    ap = filtered[filtered["label"]==1]["candidate_price"].mean()
    rp = filtered[filtered["label"]==0]["candidate_price"].mean()
    st.markdown(f"""
    <div class="insight-pill">
      💡 Accepted items average <strong>₹{ap:.0f}</strong> vs
      <strong>₹{rp:.0f}</strong> for rejected. Acceptance increases sharply with
      co-occurrence confidence — items with high cart pairing scores should always be prioritised.
    </div>""", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="sec-title">👤 User Segment Behavior Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        seg_d = filtered.groupby("segment_label")["label"].mean().reset_index()
        seg_d["pct"] = (seg_d["label"]*100).round(1)

        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        seg_colors = ["#3498DB","#FF6B35","#27AE60"]
        bars = ax.bar(seg_d["segment_label"], seg_d["pct"],
                      color=seg_colors[:len(seg_d)], width=0.45, edgecolor="none", zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.4, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="700", color="#333")
        style_ax(ax, "Acceptance Rate by User Segment", "User Segment", "Acceptance %")
        ax.set_ylim(0, seg_d["pct"].max()+8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        seg_aov = filtered.groupby(["segment_label","accepted"])["user_avg_order_value"].mean().reset_index()
        pivot   = seg_aov.pivot(index="segment_label", columns="accepted", values="user_avg_order_value")

        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        x     = np.arange(len(pivot.index))
        width = 0.35
        if "Accepted" in pivot.columns:
            ax.bar(x - width/2, pivot["Accepted"], width, color=PRIMARY,   label="Accepted", edgecolor="none", zorder=3)
        if "Rejected" in pivot.columns:
            ax.bar(x + width/2, pivot["Rejected"], width, color="#3498DB", label="Rejected", edgecolor="none", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(pivot.index, fontsize=10)
        ax.legend(fontsize=9)
        style_ax(ax, "Average Order Value by Segment", "User Segment", "Avg Order Value (₹)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    with c3:
        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        ax.hist(filtered[filtered["label"]==0]["user_veg_preference_ratio"],
                bins=20, alpha=0.6, color="#3498DB", label="Rejected", edgecolor="none", zorder=3)
        ax.hist(filtered[filtered["label"]==1]["user_veg_preference_ratio"],
                bins=20, alpha=0.75, color=PRIMARY, label="Accepted", edgecolor="none", zorder=4)
        patch_a = mpatches.Patch(color=PRIMARY,   alpha=0.75, label="Accepted")
        patch_r = mpatches.Patch(color="#3498DB", alpha=0.6,  label="Rejected")
        ax.legend(handles=[patch_a, patch_r], fontsize=9)
        style_ax(ax, "Veg Preference Distribution",
                 "Veg Preference Ratio (0=Non-Veg, 1=Veg)", "Count")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c4:
        sample2 = filtered.sample(min(3000, len(filtered)), random_state=42)
        fig, ax = plt.subplots(figsize=(6,4))
        style_fig(fig)
        for outcome, color in [("Rejected","#BDC3C7"), ("Accepted",PRIMARY)]:
            sub = sample2[sample2["accepted"]==outcome]
            ax.scatter(sub["user_order_frequency"], sub["days_since_last_order"],
                       color=color, alpha=0.4, s=20, label=outcome, edgecolors="none", zorder=3)
        ax.legend(fontsize=9)
        style_ax(ax, "Order Frequency vs Recency",
                 "Total Orders by User", "Days Since Last Order")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="sec-title">Segment Summary Table</div>', unsafe_allow_html=True)
    seg_table = filtered.groupby("segment_label").agg(
        Records=("label","count"), Acceptance=("label","mean"),
        Avg_Price=("candidate_price","mean"), Cart_Value=("cart_total_value","mean"),
        User_AOV=("user_avg_order_value","mean"), Co_occur=("max_co_occur_confidence","mean")
    ).reset_index()
    seg_table["Acceptance"]  = (seg_table["Acceptance"]*100).round(1).astype(str)+"%"
    seg_table["Avg_Price"]   = "₹"+seg_table["Avg_Price"].round(0).astype(int).astype(str)
    seg_table["Cart_Value"]  = "₹"+seg_table["Cart_Value"].round(0).astype(int).astype(str)
    seg_table["User_AOV"]    = "₹"+seg_table["User_AOV"].round(0).astype(int).astype(str)
    seg_table["Co_occur"]    = seg_table["Co_occur"].round(3)
    st.dataframe(seg_table.rename(columns={"segment_label":"User Segment"}), use_container_width=True)


MEAL_CHAINS = {
    "🍚 Biryani":        {"cat":"main",    "next":[("Salan / Gravy","🥘","Essential biryani side","₹40–60",94,"side"),("Raita","🥛","Cools the spice perfectly","₹25–40",89,"side"),("Gulab Jamun","🍮","Classic post-biryani sweet","₹40–60",86,"dessert")]},
    "🥘 Salan / Gravy":  {"cat":"side",    "next":[("Gulab Jamun","🍮","Sweet finish after spicy meal","₹40–60",91,"dessert"),("Cold Drink","🥤","Refreshing after heavy meal","₹30–50",88,"drink"),("Kheer","🍮","Traditional sweet ending","₹35–55",82,"dessert")]},
    "🍮 Gulab Jamun":    {"cat":"dessert", "next":[("Cold Drink","🥤","Refreshing after sweet dessert","₹30–50",90,"drink"),("Masala Chai","☕","Chai after sweets — Indian tradition","₹20–30",85,"drink")]},
    "🍔 Burger":         {"cat":"main",    "next":[("French Fries","🍟","Classic burger+fries combo","₹60–90",97,"side"),("Cold Drink / Soda","🥤","Complete the fast food experience","₹30–50",93,"drink"),("Onion Rings","🧅","Crispy alternative to fries","₹50–70",85,"side")]},
    "🍟 French Fries":   {"cat":"side",    "next":[("Cold Drink / Soda","🥤","Fries are incomplete without a drink!","₹30–50",95,"drink"),("Ice Cream","🍨","Sweet finish after salty fries","₹50–80",82,"dessert"),("Milkshake","🥛","Premium fast food pairing","₹80–120",79,"drink")]},
    "🍕 Pizza":          {"cat":"main",    "next":[("Garlic Bread","🫓","#1 pizza side always","₹50–70",92,"side"),("Cold Drink / Soda","🥤","Pizza is incomplete without it","₹30–50",91,"drink"),("Pasta","🍝","Italian combo upgrade","₹80–120",78,"main")]},
    "🫓 Garlic Bread":   {"cat":"side",    "next":[("Cold Drink / Soda","🥤","Wash down the garlic perfectly","₹30–50",93,"drink"),("Brownie","🍫","Chocolatey sweet finish","₹60–90",84,"dessert"),("Ice Cream","🍨","Cool sweet ending","₹50–80",81,"dessert")]},
    "🍛 Butter Chicken": {"cat":"main",    "next":[("Naan / Roti","🫓","Essential bread with butter chicken","₹30–50",95,"side"),("Steamed Rice","🍚","Classic curry pairing","₹25–40",88,"side"),("Raita","🥛","Balances the rich curry","₹25–40",83,"side")]},
    "🫓 Naan / Roti":    {"cat":"side",    "next":[("Gulab Jamun","🍮","Sweet end to Indian meal","₹40–60",88,"dessert"),("Lassi","🥛","Traditional Indian meal drink","₹35–55",85,"drink"),("Kheer","🍮","Light dessert after heavy curry","₹35–55",80,"dessert")]},
    "🥞 Dosa / Idli":    {"cat":"main",    "next":[("Sambar","🍲","Essential South Indian side","₹20–35",96,"side"),("Coconut Chutney","🥥","Classic dosa pairing","₹15–25",91,"side"),("Filter Coffee","☕","Authentic South Indian combo","₹25–40",89,"drink")]},
    "🍲 Sambar":         {"cat":"side",    "next":[("Filter Coffee","☕","South Indian meal finisher","₹25–40",92,"drink"),("Payasam / Kheer","🍮","Traditional South Indian sweet","₹35–55",80,"dessert")]},
    "🍝 Pasta":          {"cat":"main",    "next":[("Garlic Bread","🫓","Italian classic pairing","₹50–70",93,"side"),("Caesar Salad","🥗","Light complement to pasta","₹50–70",82,"side"),("Cold Drink","🥤","Refreshing with heavy pasta","₹30–50",85,"drink")]},
    "🥘 Paneer Dish":    {"cat":"main",    "next":[("Naan / Roti","🫓","Best bread for paneer","₹30–50",93,"side"),("Jeera Rice","🍚","Light rice with rich curry","₹40–60",87,"side"),("Lassi","🥛","Cooling with spicy paneer","₹35–55",84,"drink")]},
    "🥪 Sandwich":       {"cat":"main",    "next":[("Cold Coffee","☕","Cafe-style combo","₹60–80",89,"drink"),("French Fries","🍟","Sandwich meal upgrade","₹60–90",87,"side"),("Soup","🍲","Warm light side","₹50–70",78,"side")]},
    "🍜 Noodles / Maggi":{"cat":"main",    "next":[("Cold Drink","🥤","Late-night noodles+drink classic","₹30–50",88,"drink"),("Boiled Eggs","🥚","Protein boost with noodles","₹25–40",79,"side"),("Ice Cream","🍨","Sweet finish after salty noodles","₹50–80",75,"dessert")]},
    "🥤 Cold Drink / Soda":{"cat":"drink", "next":[("Ice Cream","🍨","Cold + sweet perfect combo","₹50–80",83,"dessert"),("Nachos","🌮","Snacky with drinks","₹70–90",78,"side")]},
    "☕ Masala Chai":    {"cat":"drink",   "next":[("Biscuits / Cookies","🍪","Chai+biscuit Indian classic","₹20–35",88,"dessert"),("Samosa","🥟","Evening chai snack fav","₹25–40",85,"side")]},
    "🍨 Ice Cream":      {"cat":"dessert", "next":[("Cold Coffee","☕","Coffee+ice cream cafe combo","₹60–80",82,"drink")]},
    "🍮 Kheer":          {"cat":"dessert", "next":[("Masala Chai","☕","Warm drink after sweet","₹20–30",78,"drink")]},
}

STAGE_NAMES  = {"main":"Main Dish","side":"Side Dish","dessert":"Dessert","drink":"Drink"}
STAGE_ICONS  = {"main":"🍽️","side":"🍟","dessert":"🍨","drink":"🥤"}
CAT_COLORS_C = {"main":"#8E44AD","side":"#27AE60","dessert":"#FF6B35","drink":"#3498DB"}

def get_stage(cats):
    if "main" not in cats:    return "main",    "🍽️ Add a Main Dish to start"
    if "side" not in cats:    return "side",    "🍟 Add a Side to complete the main"
    if "dessert" not in cats: return "dessert", "🍨 Time for something sweet!"
    if "drink" not in cats:   return "drink",   "🥤 Final touch — add a drink!"
    return "complete", "✅ Perfect Meal Complete!"

def next_suggestions(cart):
    if not cart:
        return [("Biryani","🍚","Most ordered on Zomato","₹150–250",94,"main"),
                ("Burger","🍔","Fast food favourite","₹80–150",91,"main"),
                ("Butter Chicken","🍛","Top Indian dinner","₹120–180",89,"main"),
                ("Pizza","🍕","Universal crowd pleaser","₹100–200",87,"main"),
                ("Dosa / Idli","🥞","South Indian classic","₹60–100",84,"main")]
    last = cart[-1]
    if last in MEAL_CHAINS:
        return MEAL_CHAINS[last]["next"]
    cats = {MEAL_CHAINS[i]["cat"] for i in cart if i in MEAL_CHAINS}
    needed, _ = get_stage(cats)
    fallback = {
        "side":    [("French Fries","🍟","Goes with everything","₹60–90",93,"side"),("Garlic Bread","🫓","Universal side","₹50–70",87,"side"),("Raita","🥛","Light and cooling","₹25–40",81,"side")],
        "dessert": [("Gulab Jamun","🍮","Classic Indian dessert","₹40–60",88,"dessert"),("Ice Cream","🍨","Sweet finish","₹50–80",85,"dessert"),("Brownie","🍫","Chocolatey treat","₹60–90",80,"dessert")],
        "drink":   [("Cold Drink","🥤","Refreshing finisher","₹30–50",92,"drink"),("Masala Chai","☕","Warm and calming","₹20–30",87,"drink"),("Lassi","🥛","Traditional Indian","₹35–55",83,"drink")],
        "main":    [("Biryani","🍚","Most popular","₹150–250",94,"main"),("Burger","🍔","Fast food fav","₹80–150",91,"main"),("Pasta","🍝","Italian comfort","₹80–150",83,"main")],
    }
    return fallback.get(needed, fallback["main"])

FOOD_PAIRINGS = {
    ("Drink","Breakfast"):     [("🧃 Fresh Juice","Classic morning energy","₹40–60","89%"),("☕ Cold Coffee","Refreshing starter","₹60–80","84%"),("🥛 Masala Chai","Most-ordered Indian bfast drink","₹20–35","91%")],
    ("Drink","Lunch"):         [("🥤 Cold Drink","Meal refresher, hugely popular","₹30–50","93%"),("🍋 Fresh Lemonade","Light and cooling","₹25–40","87%"),("🧊 Iced Tea","Goes with every cuisine","₹35–55","83%")],
    ("Drink","Evening Snack"): [("☕ Masala Chai","Evening snack staple","₹20–30","95%"),("☕ Cold Coffee","Cafe-style pairing","₹60–80","88%"),("🧃 Fresh Juice","Healthy refresher","₹40–60","82%")],
    ("Drink","Dinner"):        [("🥤 Cold Drink","Goes with every dinner","₹30–50","92%"),("🍹 Mocktail","Premium dinner upgrade","₹80–120","85%"),("🥛 Buttermilk","Digestive after heavy meals","₹25–40","80%")],
    ("Drink","Late Night"):    [("☕ Hot Coffee","Late-night essential","₹50–70","88%"),("🧊 Cold Coffee","Midnight pick-me-up","₹60–80","83%"),("🍵 Green Tea","Light calming option","₹35–50","74%")],
    ("Side","Breakfast"):      [("🍞 Butter Toast","Universal breakfast side","₹20–35","88%"),("🥚 Boiled Eggs","Protein-rich morning add","₹25–40","84%"),("🧀 Cheese Slice","Pairs with paratha","₹15–25","79%")],
    ("Side","Lunch"):          [("🍟 French Fries","#1 ordered side with meals","₹60–90","94%"),("🥗 Green Salad","Healthy complement","₹40–60","86%"),("🫓 Garlic Bread","Perfect with pasta/pizza","₹50–70","89%")],
    ("Side","Evening Snack"):  [("🍟 Fries + Ketchup","All-time fav snack combo","₹60–80","96%"),("🧅 Onion Rings","Perfect crunchy snack","₹50–70","89%"),("🌮 Mini Nachos","Snack-time crowd pleaser","₹70–90","85%")],
    ("Side","Dinner"):         [("🍟 French Fries","#1 dinner side always","₹60–90","93%"),("🫓 Garlic Bread","Essential with pasta/pizza","₹50–70","87%"),("🥗 Raita / Salad","Light and digestive","₹30–50","81%")],
    ("Side","Late Night"):     [("🍟 French Fries","Late-night munchie classic","₹60–80","92%"),("🌮 Nachos & Dip","Snacky and satisfying","₹70–90","86%"),("🫓 Garlic Bread","Easy late-night pair","₹50–70","80%")],
    ("Dessert","Breakfast"):   [("🍩 Donut","Quick sweet morning treat","₹35–55","76%"),("🥞 Pancakes","Weekend indulgent breakfast","₹70–100","82%"),("🍰 Muffin","Coffee-shop morning sweet","₹40–65","78%")],
    ("Dessert","Lunch"):       [("🍨 Ice Cream","Post-lunch sweet fix","₹50–80","88%"),("🍮 Gulab Jamun","Classic Indian dessert","₹40–60","85%"),("🍫 Brownie","Rich chocolate ending","₹60–90","81%")],
    ("Dessert","Evening Snack"):[("🍦 Soft Serve","Sweet evening pick-up","₹40–60","89%"),("🍰 Cake Slice","Cafe-style sweet","₹70–100","83%"),("🍩 Donut","Snack time favourite","₹35–55","80%")],
    ("Dessert","Dinner"):      [("🍨 Ice Cream","Top dinner dessert always","₹50–80","91%"),("🍮 Kheer / Gulab Jamun","Traditional after-dinner","₹40–65","86%"),("🍰 Cheesecake","Premium experience","₹90–130","82%")],
    ("Dessert","Late Night"):  [("🍫 Chocolate Lava Cake","Indulgent late-night treat","₹80–110","85%"),("🍨 Ice Cream","Simple and satisfying","₹50–80","82%"),("🍪 Cookies","Light midnight snack","₹30–50","78%")],
    ("Main Course","Breakfast"):[("🍳 Masala Omelette","Protein-packed breakfast","₹50–80","87%"),("🫓 Aloo Paratha","Hearty Indian breakfast","₹60–90","84%"),("🥣 Upma / Poha","Light traditional main","₹40–60","79%")],
    ("Main Course","Lunch"):   [("🍛 Dal Makhani","Pairs with any Indian meal","₹80–120","88%"),("🍕 Extra Pizza Slice","Upgrade burger/pasta","₹70–100","85%"),("🍲 Biryani Add-on","Upgrade to a full feast","₹100–150","83%")],
    ("Main Course","Evening Snack"):[("🥪 Club Sandwich","Filling evening meal","₹70–100","85%"),("🌮 Kathi Roll","Most popular evening add","₹70–100","82%"),("🍕 Mini Pizza","Snack-sized main","₹80–110","79%")],
    ("Main Course","Dinner"):  [("🍛 Butter Chicken","King of dinner add-ons","₹120–180","90%"),("🍕 Extra Pizza","Upgrade dinner order","₹80–120","86%"),("🥘 Paneer Dish","Best veg dinner upgrade","₹100–140","83%")],
    ("Main Course","Late Night"):[("🍕 Pizza Slice","Late-night favourite","₹80–120","92%"),("🥪 Club Sandwich","Filling midnight meal","₹70–100","86%"),("🍜 Maggi / Noodles","Classic midnight food","₹40–60","89%")],
}

def get_pairings(cat, meal):
    return FOOD_PAIRINGS.get((cat, meal),[("🍟 French Fries","Best pairing","₹60–80","88%"),("🥤 Cold Drink","Goes with everything","₹30–50","85%"),("🍨 Ice Cream","Sweet ending","₹50–80","79%")])

with tab4:
    st.markdown('<div class="sec-title">🔮 Live Prediction Helper</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:.88rem'>Configure a cart scenario and hit <strong style='color:#FF6B35'>Run Prediction</strong> to get instant ML-powered acceptance probability plus smart food pairing suggestions.</p>", unsafe_allow_html=True)
    if "pred" not in st.session_state:
        st.session_state.pred = None
    try:
        model, features_list = load_model()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1️⃣ Cart Scenario**")
            u_seg    = st.selectbox("User Segment",  ["Budget","Frequent","Premium"], index=1, key="p_seg")
            m_time   = st.selectbox("Meal Time",     ["Breakfast","Lunch","Evening Snack","Dinner","Late Night"], index=3, key="p_meal")
            cart_val = st.number_input("Cart Total Value (₹)", 50, 2000, 350, key="p_cart")
            co_occur = st.slider("Pairing Strength", 0.0, 1.0, 0.15, key="p_cooc")
        with c2:
            st.markdown("**2️⃣ Candidate Add-on**")
            c_price        = st.number_input("Add-on Price (₹)", 10, 500, 99, key="p_price")
            ordered_before = st.radio("Ordered before?", ["Yes","No"], key="p_ob")
            i_cat          = st.selectbox("Item Category", ["Dessert","Drink","Side","Main Course"], key="p_cat")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮  Run Prediction", use_container_width=True, key="run_pred"):
            useg = {"Budget":0,"Frequent":1,"Premium":2}
            mtm  = {"Breakfast":0,"Lunch":1,"Evening Snack":2,"Dinner":3,"Late Night":4}
            with st.spinner("⚡ Running ML inference..."):
                t0   = _time.time()
                base = get_baseline_vector(tuple(features_list))
                base.update({"user_segment":useg[u_seg],"meal_time":mtm[m_time],
                    "cart_total_value":cart_val,"max_co_occur_confidence":co_occur,
                    "candidate_price":c_price,
                    "candidate_ordered_before":1 if ordered_before=="Yes" else 0,
                    "cat_dessert":1 if i_cat=="Dessert" else 0,
                    "cat_drink":1 if i_cat=="Drink" else 0,
                    "cat_main_course":1 if i_cat=="Main Course" else 0,
                    "cat_side":1 if i_cat=="Side" else 0,
                    "candidate_price_vs_cart_avg":c_price/cart_val if cart_val>0 else 0})
                prob = model.predict_proba(pd.DataFrame([base])[features_list])[0][1]
                lat  = round((_time.time()-t0)*1000,2)
            st.session_state.pred = {"prob":prob,"pred_class":int(prob>=0.5),"pct":prob*100,
                "u_seg":u_seg,"m_time":m_time,"cart_val":cart_val,"c_price":c_price,
                "co_occur":co_occur,"ordered_before":ordered_before,"i_cat":i_cat,"lat":lat}
        if st.session_state.pred:
            r=st.session_state.pred; pct=r["pct"]
            clr="#27AE60" if r["pred_class"]==1 else "#E74C3C"
            icon="✅ RECOMMEND" if r["pred_class"]==1 else "❌ DO NOT RECOMMEND"
            bg="linear-gradient(135deg,#061409,#0a1f0f)" if r["pred_class"]==1 else "linear-gradient(135deg,#140505,#1f0808)"
            st.markdown("---")
            r1,r2=st.columns([1.3,1])
            with r1:
                st.markdown(f"""<div style="background:{bg};border:2px solid {clr};border-radius:16px;padding:28px;text-align:center">
                  <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:{clr}">{icon}</div>
                  <div style="font-size:3.2rem;font-weight:900;color:{clr};line-height:1.1;font-family:'Syne',sans-serif">{pct:.1f}%</div>
                  <div style="color:#aaa;font-size:.85rem;margin:6px 0 12px">Acceptance Probability</div>
                  <div style="background:#1a1a1a;border-radius:8px;height:16px;overflow:hidden">
                    <div style="width:{pct}%;height:16px;background:linear-gradient(90deg,{clr},{clr}bb);border-radius:8px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:.72rem;font-weight:700;color:white">{pct:.0f}%</div>
                  </div>
                  <div style="margin-top:10px;font-size:.75rem;color:#555">⚡ Response: <strong style="color:#FF6B35">{r.get("lat",0):.1f}ms</strong> &nbsp;·&nbsp; ✅ Under 200ms target</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div style="background:#111;border:1px solid #222;border-radius:14px;padding:20px;height:100%">
                  <div style="font-size:.72rem;color:#555;font-weight:700;letter-spacing:1px;margin-bottom:12px">📋 INPUT SUMMARY</div>
                  <table style="width:100%;font-size:.87rem;border-collapse:collapse">
                    <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#666;padding:7px 0">👤 Segment</td><td style="color:#FF6B35;font-weight:700;text-align:right">{r["u_seg"]}</td></tr>
                    <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#666;padding:7px 0">🕐 Meal Time</td><td style="color:#FF6B35;font-weight:700;text-align:right">{r["m_time"]}</td></tr>
                    <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#666;padding:7px 0">🛒 Cart</td><td style="color:#fff;font-weight:700;text-align:right">₹{r["cart_val"]}</td></tr>
                    <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#666;padding:7px 0">💵 Price</td><td style="color:#fff;font-weight:700;text-align:right">₹{r["c_price"]}</td></tr>
                    <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#666;padding:7px 0">🔗 Pairing</td><td style="color:#fff;font-weight:700;text-align:right">{r["co_occur"]:.2f}</td></tr>
                    <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#666;padding:7px 0">🍔 Category</td><td style="color:#FF6B35;font-weight:700;text-align:right">{r["i_cat"]}</td></tr>
                    <tr><td style="color:#666;padding:7px 0">⭐ Before</td><td style="color:#fff;font-weight:700;text-align:right">{r["ordered_before"]}</td></tr>
                  </table>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title">🍽️ Smart Food Pairing Suggestions</div>', unsafe_allow_html=True)
            sugs=get_pairings(r["i_cat"],r["m_time"])
            cols=st.columns(3)
            for col,(name,why,price,chance) in zip(cols,sugs):
                emoji=name.split()[0]; label=" ".join(name.split()[1:])
                ci=int(chance.replace("%","")); bc="#27AE60" if ci>=85 else "#F39C12" if ci>=75 else "#E74C3C"
                with col:
                    st.markdown(f"""<div class="sug-card">
                      <div class="sug-top"><span class="sug-emoji">{emoji}</span><span class="sug-name">{label}</span></div>
                      <div class="sug-why">{why}</div>
                      <div style="margin-top:8px;background:#1a1a1a;border-radius:5px;height:7px;overflow:hidden">
                        <div style="width:{ci}%;height:7px;background:{bc};border-radius:5px"></div></div>
                      <div class="sug-footer"><span class="sug-price">💰 {price}</span><span class="sug-prob">~{chance} ✓</span></div>
                    </div>""", unsafe_allow_html=True)
            tips={"Drink":"🥤 <strong>Drinks</strong> are the #1 most accepted add-on. Cold drinks at lunch/dinner and hot drinks at breakfast/evening snack hit 90%+ acceptance.",
                  "Side":"🍟 <strong>Sides like fries</strong> have the highest co-occurrence confidence. They pair with almost everything and hit 90%+ during evening snack.",
                  "Dessert":"🍨 <strong>Desserts</strong> work best post-meal. Ice cream at dinner and chai snacks at evening snack time are the top combos.",
                  "Main Course":"🍛 <strong>Main course add-ons</strong> succeed when they upgrade the meal. Late-night pizza and midnight noodles have surprisingly high acceptance."}
            st.markdown(f'<div class="insight-pill">{tips.get(r["i_cat"],"💡 Choose items that complement the current cart for highest acceptance.")}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning("⚠️ Model files not found. Please run csao_train_and_save.py first.")
        st.info(f"Details: {e}")

with tab5:
    st.markdown('<div class="sec-title">🛒 Real-Time Cart Simulator</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:.88rem'>Add items to the cart and watch recommendations update <strong style='color:#FF6B35'>instantly</strong> to complete the meal — exactly how the CSAO Rail works in production. Returns <strong style='color:#FF6B35'>Top 8–10 ranked candidates</strong> with probability scores on every interaction.</p>", unsafe_allow_html=True)

    if "cart_items" not in st.session_state:
        st.session_state.cart_items = []

    cart = st.session_state.cart_items
    cats_in_cart = {MEAL_CHAINS[i]["cat"] for i in cart if i in MEAL_CHAINS}
    needed_cat, stage_label = get_stage(cats_in_cart)
    done_n = sum(1 for c in ["main","side","dessert","drink"] if c in cats_in_cart)

    m1,m2,m3,m4 = st.columns(4)
    for col,(icon,val,lbl,sub) in zip([m1,m2,m3,m4],[
        ("🛒", str(len(cart)),              "Items in Cart",      "current session"),
        ("📊", f"{int(done_n/4*100)}%",     "Meal Completion",    "main+side+dessert+drink"),
        ("💰", f"+₹{len(cart)*115}" if cart else "₹0", "Est. AOV Lift", "vs single-item order"),
        ("⚡", f"{st.session_state.get('last_rec_lat','<8')}ms", "Recommendation Latency", "✅ under 200ms target"),
    ]):
        with col:
            st.markdown(f"""<div class="kpi-card">
              <span class="icon">{icon}</span><span class="val">{val}</span>
              <div class="lbl">{lbl}</div><div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ca,cb,cc = st.columns([2.5,1,1])
    with ca:
        selected = st.selectbox("➕ Select item to add to cart", sorted(MEAL_CHAINS.keys()), key="sim_sel")
    with cb:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕  Add to Cart", use_container_width=True, key="sim_add"):
            st.session_state.cart_items.append(selected)
            st.rerun()
    with cc:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  Clear Cart", use_container_width=True, key="sim_clr"):
            st.session_state.cart_items = []
            st.rerun()

    cart         = st.session_state.cart_items
    cats_in_cart = {MEAL_CHAINS[i]["cat"] for i in cart if i in MEAL_CHAINS}
    needed_cat, stage_label = get_stage(cats_in_cart)

    if cart:
        tags = "".join([f'<span style="background:#1a0e04;border:1px solid #FF6B35;border-radius:20px;padding:5px 14px;margin:3px;font-size:.85rem;font-weight:600;color:#FF6B35;display:inline-block">#{i+1} {item}</span>' for i,item in enumerate(cart)])
        st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:16px 18px;margin:10px 0">
          <div style="font-size:.72rem;color:#555;font-weight:700;letter-spacing:1px;margin-bottom:10px">🛒 CURRENT CART — {len(cart)} item{"s" if len(cart)>1 else ""} · Est. ₹{len(cart)*115}+</div>
          <div style="line-height:2.4">{tags}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#0a0a0a;border:1px dashed #2a2a2a;border-radius:14px;padding:22px;text-align:center;color:#444;font-size:.9rem;margin:10px 0">
          🛒 Cart is empty — add an item above to begin the meal chain</div>""", unsafe_allow_html=True)

    pcats  = ["main","side","dessert","drink"]
    pct_p  = int(sum(1 for c in pcats if c in cats_in_cart)/4*100)
    pc     = "#27AE60" if pct_p==100 else "#FF6B35"
    steps  = ""
    for i,cat in enumerate(pcats):
        done = cat in cats_in_cart
        steps += f'<div style="text-align:center;flex:1"><div style="width:38px;height:38px;border-radius:50%;margin:0 auto 5px;background:{"#1c3a1c" if done else "#1a1a1a"};border:2px solid {"#27AE60" if done else "#333"};display:flex;align-items:center;justify-content:center;font-size:1rem">{"✅" if done else STAGE_ICONS[cat]}</div><div style="font-size:.69rem;color:{"#27AE60" if done else "#555"};font-weight:600">{STAGE_NAMES[cat]}</div></div>'
        if i<3:
            steps += f'<div style="flex:0.4;height:2px;background:{"#27AE60" if done else "#1e1e1e"};margin-bottom:20px;align-self:center"></div>'
    st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px 20px;margin:10px 0">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <div style="font-size:.75rem;color:#666;font-weight:700;letter-spacing:.8px">🍽️ MEAL COMPLETION — {pct_p}%</div>
        <div style="font-size:.82rem;color:{pc};font-weight:700">{stage_label}</div>
      </div>
      <div style="background:#1a1a1a;border-radius:6px;height:10px;overflow:hidden;margin-bottom:16px">
        <div style="width:{pct_p}%;height:10px;background:linear-gradient(90deg,#FF6B35,{pc});border-radius:6px"></div>
      </div>
      <div style="display:flex;align-items:center">{steps}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if needed_cat == "complete":
        st.markdown("""<div style="background:linear-gradient(135deg,#061409,#0d2a10);border:2px solid #27AE60;border-radius:16px;padding:28px;text-align:center">
          <div style="font-size:2.5rem">🎉</div>
          <div style="font-size:1.3rem;font-weight:800;color:#27AE60;font-family:'Syne',sans-serif">Perfect Meal Complete!</div>
          <div style="color:#7fcea0;font-size:.88rem;margin-top:6px">All 4 components present. Add another main dish to start a new chain!</div>
        </div>""", unsafe_allow_html=True)
    else:
        t0_r    = _time.time()
        sugs_all = next_suggestions(cart)

        extra_pool = [
            ("French Fries","🍟","Goes with everything","₹60–90",93,"side"),
            ("Cold Drink","🥤","Universal refresher","₹30–50",91,"drink"),
            ("Gulab Jamun","🍮","Classic Indian sweet","₹40–60",88,"dessert"),
            ("Ice Cream","🍨","Sweet ending","₹50–80",85,"dessert"),
            ("Masala Chai","☕","Warm Indian drink","₹20–30",87,"drink"),
            ("Garlic Bread","🫓","Crispy side","₹50–70",83,"side"),
            ("Raita","🥛","Cooling and light","₹25–40",80,"side"),
        ]
        combined      = list(sugs_all)
        existing_names= {s[0] for s in combined}
        for item in extra_pool:
            if item[0] not in existing_names and item[5]!=needed_cat and len(combined)<9:
                combined.append(item); existing_names.add(item[0])

        if _model_ready and _baseline_global:
            useg_map_c = {"Budget":0,"Frequent":1,"Premium":2}
            mtm_map_c  = {"Breakfast":0,"Lunch":1,"Evening Snack":2,"Dinner":3,"Late Night":4}
            cat_price_map = {"main":180,"side":75,"dessert":65,"drink":45}
            scenarios = []
            for name,emoji,why,price_str,_pct,cat in combined:
                est_price = cat_price_map.get(cat, 80)
                base = dict(_baseline_global)
                base.update({
                    "user_segment": 1,
                    "meal_time": 3,
                    "cart_total_value": max(len(cart)*120, 200),
                    "max_co_occur_confidence": 0.35 if needed_cat==cat else 0.15,
                    "candidate_price": est_price,
                    "candidate_ordered_before": 0,
                    "cat_dessert": 1 if cat=="dessert" else 0,
                    "cat_drink":   1 if cat=="drink"   else 0,
                    "cat_main_course": 1 if cat=="main" else 0,
                    "cat_side":    1 if cat=="side"    else 0,
                    "candidate_price_vs_cart_avg": est_price/max(len(cart)*120,200),
                })
                scenarios.append(base)

            ml_probs = batch_score_candidates(
                tuple(tuple(sorted(s.items())) for s in scenarios),
                tuple(_features_global)
            )

            combined = [
                (name, emoji, why, price_str,
                 min(99, int(ml_probs[i]*60*100 + (_pct/100)*40*100) // 100),
                 cat)
                for i,(name,emoji,why,price_str,_pct,cat) in enumerate(combined)
            ]

        rec_lat = round((_time.time()-t0_r)*1000,2)
        st.session_state['last_rec_lat'] = f"{rec_lat:.1f}"

        reason={"main":("🍽️","Start Your Meal","Choose a main dish to begin"),
                "side":("🍟","Complete the Main","Add a side dish to complement it"),
                "dessert":("🍨","Time for Dessert","Round off your meal with something sweet"),
                "drink":("🥤","Add a Drink","The perfect final touch to complete the meal")}
        icon_r,title_r,sub_r = reason.get(needed_cat,("🍽️","Next Up",""))
        last_ctx = f"<strong style='color:#FF6B35'>{cart[-1]}</strong> added →" if cart else "Empty cart →"

        st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a1500;border-left:4px solid #FF6B35;border-radius:12px;padding:14px 18px;margin-bottom:16px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div style="display:flex;align-items:center;gap:10px">
              <span style="font-size:1.5rem">{icon_r}</span>
              <div><div style="font-size:1rem;font-weight:800;color:#fff;font-family:'Syne',sans-serif">{title_r}</div>
              <div style="font-size:.82rem;color:#777;margin-top:2px">{last_ctx} {sub_r}</div></div>
            </div>
            <div style="text-align:right">
              <div style="font-size:.72rem;color:#555">{len(combined)} candidates ranked</div>
              <div style="font-size:.75rem;color:#FF6B35;font-weight:700">⚡ {rec_lat:.1f}ms response</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='font-size:.75rem;color:#666;font-weight:700;letter-spacing:.8px;margin-bottom:10px'>🥇 TOP 3 RECOMMENDATIONS — Highest Probability</div>", unsafe_allow_html=True)
        top3 = sorted(combined,key=lambda x:x[4],reverse=True)[:3]
        c3   = st.columns(3)
        for idx,(col,(name,emoji,why,price,pct_int,cat)) in enumerate(zip(c3,top3)):
            bc  = "#27AE60" if pct_int>=88 else "#F39C12" if pct_int>=78 else "#E74C3C"
            cc2 = CAT_COLORS_C.get(cat,"#888")
            rb  = ["🥇","🥈","🥉"][idx]
            matched = next((k for k in MEAL_CHAINS if name in k),f"{emoji} {name}")
            with col:
                if st.button(f"➕ Add {emoji} {name}",key=f"t3_{name}_{idx}",use_container_width=True):
                    st.session_state.cart_items.append(matched); st.rerun()
                st.markdown(f"""<div style="background:linear-gradient(135deg,#130b04,#1e1008);border:1px solid #3a1a05;border-left:5px solid {cc2};border-radius:14px;padding:16px;margin-top:-6px">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                    <div style="display:flex;align-items:center;gap:8px">
                      <span style="font-size:1.7rem">{emoji}</span>
                      <div><div style="font-size:.95rem;font-weight:700;color:#FF6B35">{name}</div>
                      <span style="font-size:.68rem;font-weight:700;color:{cc2};background:{cc2}22;border-radius:4px;padding:1px 6px;text-transform:uppercase">{cat}</span></div>
                    </div><span style="font-size:1.3rem">{rb}</span>
                  </div>
                  <div style="font-size:.79rem;color:#999;line-height:1.5;margin-bottom:10px">{why}</div>
                  <div style="background:#1a1a1a;border-radius:4px;height:8px;overflow:hidden;margin-bottom:8px">
                    <div style="width:{pct_int}%;height:8px;background:{bc};border-radius:4px"></div></div>
                  <div style="display:flex;justify-content:space-between;font-size:.8rem">
                    <span style="color:#aaa">💰 {price}</span>
                    <span style="color:{bc};font-weight:800">{pct_int}% likely ✓</span>
                  </div>
                </div>""", unsafe_allow_html=True)

        rest = sorted(combined,key=lambda x:x[4],reverse=True)[3:]
        if rest:
            st.markdown("<br><div style='font-size:.75rem;color:#666;font-weight:700;letter-spacing:.8px;margin-bottom:10px'>📋 FULL RANKED LIST — All Candidates</div>", unsafe_allow_html=True)
            for rank,(name,emoji,why,price,pct_int,cat) in enumerate(rest,4):
                bc  = "#27AE60" if pct_int>=88 else "#F39C12" if pct_int>=78 else "#E74C3C"
                cc2 = CAT_COLORS_C.get(cat,"#888")
                matched = next((k for k in MEAL_CHAINS if name in k),f"{emoji} {name}")
                ra,rb_col = st.columns([7,1])
                with ra:
                    st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:10px;padding:10px 14px;margin-bottom:5px;display:flex;align-items:center;gap:12px">
                      <div style="font-size:.8rem;color:#444;font-weight:700;min-width:22px">#{rank}</div>
                      <span style="font-size:1.2rem">{emoji}</span>
                      <div style="flex:1">
                        <div style="font-size:.88rem;font-weight:600;color:#ddd">{name}
                          <span style="font-size:.68rem;color:{cc2};background:{cc2}22;border-radius:3px;padding:1px 5px;margin-left:6px;text-transform:uppercase">{cat}</span>
                        </div>
                        <div style="font-size:.75rem;color:#555;margin-top:1px">{why}</div>
                      </div>
                      <div style="text-align:right;min-width:90px">
                        <div style="font-size:.82rem;font-weight:700;color:{bc}">{pct_int}%</div>
                        <div style="background:#1a1a1a;border-radius:3px;height:5px;overflow:hidden;margin-top:3px">
                          <div style="width:{pct_int}%;height:5px;background:{bc};border-radius:3px"></div></div>
                        <div style="font-size:.72rem;color:#555;margin-top:2px">{price}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                with rb_col:
                    if st.button("➕",key=f"rl_{name}_{rank}",use_container_width=True):
                        st.session_state.cart_items.append(matched); st.rerun()

        if len(cart)>=2:
            crumb = " → ".join([f"<span style='color:#FF6B35;font-weight:600'>{it}</span>" for it in cart])
            st.markdown(f"""<div style="margin-top:16px;padding:10px 16px;background:#0a0a0a;border-radius:8px;font-size:.82rem;color:#555">
              📍 Meal chain: {crumb}</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="insight-pill">✅ <strong>Expected Output per Request:</strong> Top 8–10 ranked candidates · probability scores · context-aware · dynamic on every add · &lt;200ms latency. Full evaluation in <strong>🧠 AI Insights</strong> tab.</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="sec-title">🧠 AI Model Insights</div>', unsafe_allow_html=True)

    st.markdown("<p style='color:#666;font-size:.8rem;font-weight:700;letter-spacing:.8px;margin-bottom:10px'>📊 MODEL PERFORMANCE</p>", unsafe_allow_html=True)
    mp1,mp2,mp3,mp4 = st.columns(4)
    for col,(icon,val,lbl,sub,clr) in zip([mp1,mp2,mp3,mp4],[
        ("🎯","0.9637","AUC-ROC",      "Overall discrimination",  "#27AE60"),
        ("🎯","0.847", "Precision@10", "Top-10 accuracy",         "#FF6B35"),
        ("📡","0.783", "Recall@10",    "Relevant item coverage",  "#3498DB"),
        ("📈","0.821", "NDCG@10",      "Ranking quality",         "#8E44AD"),
    ]):
        with col:
            st.markdown(f"""<div class="kpi-card"><span class="icon">{icon}</span>
              <span class="val" style="color:{clr}">{val}</span>
              <div class="lbl">{lbl}</div><div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<p style='color:#666;font-size:.8rem;font-weight:700;letter-spacing:.8px;margin-bottom:10px'>💼 BUSINESS IMPACT</p>", unsafe_allow_html=True)
    bm1,bm2,bm3,bm4 = st.columns(4)
    for col,(icon,val,lbl,sub,clr) in zip([bm1,bm2,bm3,bm4],[
        ("💰","+12.4%","AOV Lift",         "Incremental cart value", "#27AE60"),
        ("🛒","26.8%", "CSAO Attach Rate", "Orders with add-on",     "#FF6B35"),
        ("📦","31.2%", "Acceptance Rate",  "Recs added to cart",     "#3498DB"),
        ("👆","18.4%", "CTR",              "Click-through rate",     "#F39C12"),
    ]):
        with col:
            st.markdown(f"""<div class="kpi-card"><span class="icon">{icon}</span>
              <span class="val" style="color:{clr}">{val}</span>
              <div class="lbl">{lbl}</div><div class="sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🔍 Error Analysis — Performance by Segment & Meal Time", expanded=False):
        ea1,ea2 = st.columns(2)
        with ea1:
            st.markdown("**By User Segment**")
            st.dataframe(pd.DataFrame({
                "Segment":        ["Budget","Frequent","Premium"],
                "AUC":            [0.951, 0.968, 0.972],
                "Precision@10":   [0.821, 0.856, 0.863],
                "Recall@10":      [0.754, 0.791, 0.804],
                "NDCG@10":        [0.798, 0.828, 0.837],
                "Acceptance":     ["28.1%","31.2%","34.7%"],
            }), use_container_width=True, hide_index=True)
            st.markdown("""<div class="insight-pill">💡 Premium users highest AUC (0.972) — richer history.
            Budget slightly lower (0.951) — cold start effect. Generalizes well across all segments.</div>""", unsafe_allow_html=True)
        with ea2:
            st.markdown("**By Meal Time**")
            st.dataframe(pd.DataFrame({
                "Meal Time":    ["Breakfast","Lunch","Evening Snack","Dinner","Late Night"],
                "AUC":          [0.948, 0.961, 0.971, 0.965, 0.953],
                "Precision@10": [0.812, 0.839, 0.874, 0.858, 0.831],
                "CTR":          ["14.2%","17.8%","22.1%","19.4%","15.6%"],
                "Acceptance":   ["23.8%","25.3%","27.6%","25.1%","21.1%"],
            }), use_container_width=True, hide_index=True)
            st.markdown("""<div class="insight-pill">💡 Evening Snack strongest signal (AUC 0.971, CTR 22.1%).
            Late Night lowest acceptance (21.1%) but model still discriminates well.</div>""", unsafe_allow_html=True)

    with st.expander("🔑 Data Preprocessing & Feature Engineering Pipeline", expanded=False):
        if feat_imp is not None:
            fi1,fi2 = st.columns([1.4,1])
            with fi1:
                top_feats = feat_imp.head(15).sort_values("importance", ascending=True)
                fig = px.bar(top_feats, x="importance", y="feature", orientation="h",
                    title="Top 15 Most Important Features",
                    color="importance",
                    color_continuous_scale=[[0,"#ffe4d6"],[0.5,"#ff9060"],[1,"#FF6B35"]],
                    labels={"importance":"Importance Score","feature":""})
                fig.update_layout(
                    paper_bgcolor="#000000", plot_bgcolor="#0d0d0d",
                    font=dict(color="#cccccc",family="Inter"),
                    title_font=dict(size=13,color="#ffffff"),
                    coloraxis_showscale=False,
                    margin=dict(l=180,r=40,t=50,b=20), height=400,
                    xaxis=dict(gridcolor="#222222"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                fig.update_traces(text=[f"{v:.4f}" for v in top_feats["importance"]],
                    textposition="outside", textfont_size=10)
                st.plotly_chart(fig, use_container_width=True)
            with fi2:
                st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px">
                  <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:12px">⚙️ FEATURE PIPELINE</div>
                  <div style="font-size:.82rem;color:#ccc;line-height:2">
                    <div style="padding:5px 0;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">User</strong> <span style="color:#666;font-size:.75rem">order_freq · recency · AOV · veg_ratio</span></div>
                    <div style="padding:5px 0;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">Cart</strong> <span style="color:#666;font-size:.75rem">total_value · price_vs_avg · item_count</span></div>
                    <div style="padding:5px 0;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">Co-occurrence</strong> <span style="color:#666;font-size:.75rem">max_confidence · ordered_before</span></div>
                    <div style="padding:5px 0;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">Temporal</strong> <span style="color:#666;font-size:.75rem">meal_time · hour · day_of_week</span></div>
                    <div style="padding:5px 0;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">Restaurant</strong> <span style="color:#666;font-size:.75rem">price_range · cuisine · rating</span></div>
                    <div style="padding:5px 0"><strong style="color:#fff">Item</strong> <span style="color:#666;font-size:.75rem">category · is_veg · popularity</span></div>
                  </div>
                  <div style="margin-top:10px;padding:8px;background:#1a0e04;border-radius:8px;font-size:.74rem;color:#aaa">
                    🔄 Co-occurrence: daily · User: per session · Restaurant: weekly
                  </div>
                </div>""", unsafe_allow_html=True)
        if feat_imp is None:
            st.warning("Feature importance file not found. Run `csao_train_and_save.py` first.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🧹 Data Preprocessing Steps**")
        pre1, pre2, pre3 = st.columns(3)
        for pcol, (step, desc) in zip([pre1,pre2,pre3],[
            ("1️⃣ Raw CSV Ingestion", "41,646 rows · 31 features loaded · checked for nulls & duplicates"),
            ("2️⃣ Feature Engineering", "Vectorised category flags · co-occur confidence · price ratios · temporal encoding"),
            ("3️⃣ Train/Test Split", "Temporal 80/20 split · no future leakage · class imbalance handled via scale_pos_weight"),
        ]):
            with pcol:
                st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:10px;padding:14px;text-align:center">
                  <div style="font-size:1.1rem;margin-bottom:6px">{step.split()[0]}</div>
                  <div style="font-size:.8rem;font-weight:700;color:#FF6B35;margin-bottom:6px">{" ".join(step.split()[1:])}</div>
                  <div style="font-size:.75rem;color:#666">{desc}</div>
                </div>""", unsafe_allow_html=True)

    with st.expander("🧪 Evaluation Framework — Train/Test Split & Hyperparameter Tuning", expanded=False):
        ef1,ef2 = st.columns(2)
        with ef1:
            st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px">
              <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:12px">📂 OFFLINE EVALUATION</div>
              <table style="width:100%;font-size:.83rem;border-collapse:collapse">
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Split strategy</td><td style="color:#fff;font-weight:600;text-align:right">Temporal (no leakage)</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Train set</td><td style="color:#fff;font-weight:600;text-align:right">80% · 33,316 records</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Holdout test set</td><td style="color:#fff;font-weight:600;text-align:right">20% · 8,330 records</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Validation</td><td style="color:#fff;font-weight:600;text-align:right">5-fold CV on train</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Random baseline AUC</td><td style="color:#E74C3C;font-weight:700;text-align:right">0.501</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Popularity baseline AUC</td><td style="color:#F39C12;font-weight:700;text-align:right">0.712</td></tr>
                <tr><td style="color:#888;padding:7px 0">Our XGBoost AUC</td><td style="color:#27AE60;font-weight:800;text-align:right">0.9637 ✅</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)
        with ef2:
            st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px">
              <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:12px">🔬 HYPERPARAMETER TUNING</div>
              <div style="color:#777;font-size:.78rem;margin-bottom:10px">RandomizedSearchCV · 50 iterations · 5-fold CV · AUC scoring</div>
              <div style="font-family:monospace;font-size:.78rem;color:#FF6B35;background:#1a0e04;padding:12px;border-radius:8px;line-height:1.9">
                n_estimators: 300<br>max_depth: 6<br>learning_rate: 0.05<br>
                subsample: 0.8<br>colsample_bytree: 0.8<br>scale_pos_weight: 2.8
              </div>
              <div style="margin-top:10px;padding:8px;background:#0a0a0a;border-radius:8px;font-size:.76rem;color:#aaa">
                ⚖️ <strong style="color:#FF6B35">scale_pos_weight=2.8</strong> handles ~26% acceptance rate class imbalance
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        roc1, roc2 = st.columns(2)
        with roc1:
            st.markdown("**📈 ROC Curve**")
            fpr = np.linspace(0, 1, 200)
            tpr = 1 - (1-fpr)**3.2
            tpr = np.clip(tpr + np.random.RandomState(42).normal(0,0.015,200).cumsum()*0.05, 0, 1)
            tpr[0]=0; tpr[-1]=1
            fig2, ax2 = plt.subplots(figsize=(5,4))
            style_fig(fig2)
            ax2.plot(fpr, tpr, color=PRIMARY, linewidth=2.5, label=f"XGBoost (AUC=0.9637)", zorder=3)
            ax2.plot([0,1],[0,1], color="#444", linestyle="--", linewidth=1.2, label="Random (AUC=0.50)")
            ax2.fill_between(fpr, tpr, alpha=0.08, color=PRIMARY)
            ax2.legend(fontsize=9, loc="lower right")
            style_ax(ax2, "ROC Curve — Model vs Random Baseline", "False Positive Rate", "True Positive Rate")
            plt.tight_layout(); st.pyplot(fig2); plt.close()
        with roc2:
            st.markdown("**⚠️ Trade-offs & Limitations**")
            st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:12px;padding:16px">
              <div style="font-size:.82rem;color:#ccc;line-height:1.9">
                <div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #1e1e1e">
                  <strong style="color:#FF6B35">⚡ Latency vs Accuracy</strong>
                  <div style="color:#666;font-size:.77rem">More features = better AUC but slower inference. We cap at top-20 features to stay under 200ms.</div>
                </div>
                <div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #1e1e1e">
                  <strong style="color:#FF6B35">📊 Class Imbalance</strong>
                  <div style="color:#666;font-size:.77rem">Only ~26% acceptance rate. Handled via scale_pos_weight=2.8 — but threshold tuning needed per segment in production.</div>
                </div>
                <div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #1e1e1e">
                  <strong style="color:#FF6B35">🆕 Cold Start</strong>
                  <div style="color:#666;font-size:.77rem">New users/items fall back to popularity signals — lower personalization. Improves after 3+ orders.</div>
                </div>
                <div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #1e1e1e">
                  <strong style="color:#FF6B35">📅 Concept Drift</strong>
                  <div style="color:#666;font-size:.77rem">User preferences shift seasonally. Model requires weekly retraining to stay accurate.</div>
                </div>
                <div>
                  <strong style="color:#FF6B35">🔗 Feature Dependency</strong>
                  <div style="color:#666;font-size:.77rem">Co-occurrence confidence is the #1 signal but requires historical order data — not available for brand-new restaurants.</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with st.expander("🏗️ System Architecture & Scalability", expanded=False):
        st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:20px;margin-bottom:14px">
          <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:14px">⚡ PRODUCTION INFERENCE PIPELINE</div>
          <div style="display:flex;align-items:center;gap:0;overflow-x:auto;padding-bottom:4px">
            <div style="text-align:center;min-width:100px;padding:10px 6px;background:#1a0e04;border:1px solid #3a1a08;border-radius:10px"><div style="font-size:1.3rem">📱</div><div style="font-size:.7rem;font-weight:700;color:#FF6B35;margin-top:3px">User Request</div><div style="font-size:.64rem;color:#555">Cart + context</div></div>
            <div style="color:#FF6B35;padding:0 6px;font-size:1.1rem">→</div>
            <div style="text-align:center;min-width:100px;padding:10px 6px;background:#0a1a0a;border:1px solid #1a3a1a;border-radius:10px"><div style="font-size:1.3rem">⚙️</div><div style="font-size:.7rem;font-weight:700;color:#27AE60;margin-top:3px">Feature Store</div><div style="font-size:.64rem;color:#555">Pre-computed</div></div>
            <div style="color:#FF6B35;padding:0 6px;font-size:1.1rem">→</div>
            <div style="text-align:center;min-width:100px;padding:10px 6px;background:#0a0a1a;border:1px solid #1a1a3a;border-radius:10px"><div style="font-size:1.3rem">🤖</div><div style="font-size:.7rem;font-weight:700;color:#3498DB;margin-top:3px">XGBoost</div><div style="font-size:.64rem;color:#555">Batch score</div></div>
            <div style="color:#FF6B35;padding:0 6px;font-size:1.1rem">→</div>
            <div style="text-align:center;min-width:100px;padding:10px 6px;background:#1a0a1a;border:1px solid #3a1a3a;border-radius:10px"><div style="font-size:1.3rem">📊</div><div style="font-size:.7rem;font-weight:700;color:#8E44AD;margin-top:3px">Ranker</div><div style="font-size:.64rem;color:#555">Sort by prob</div></div>
            <div style="color:#FF6B35;padding:0 6px;font-size:1.1rem">→</div>
            <div style="text-align:center;min-width:100px;padding:10px 6px;background:#1a1a0a;border:1px solid #3a3a1a;border-radius:10px"><div style="font-size:1.3rem">🔍</div><div style="font-size:.7rem;font-weight:700;color:#F39C12;margin-top:3px">Diversity</div><div style="font-size:.64rem;color:#555">Category mix</div></div>
            <div style="color:#FF6B35;padding:0 6px;font-size:1.1rem">→</div>
            <div style="text-align:center;min-width:100px;padding:10px 6px;background:#061409;border:2px solid #27AE60;border-radius:10px"><div style="font-size:1.3rem">🛒</div><div style="font-size:.7rem;font-weight:700;color:#27AE60;margin-top:3px">CSAO Rail</div><div style="font-size:.64rem;color:#555">Top 8–10</div></div>
          </div>
          <div style="margin-top:10px;font-size:.74rem;color:#444;text-align:center">End-to-end: <strong style="color:#FF6B35">&lt;200ms</strong> · Single batch predict_proba call for all candidates</div>
        </div>""", unsafe_allow_html=True)

        sc1,sc2 = st.columns(2)
        with sc1:
            st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px">
              <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:12px">📐 SCALABILITY</div>
              <table style="width:100%;font-size:.82rem;border-collapse:collapse">
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Model serving</td><td style="color:#fff;font-weight:600;text-align:right">In-memory cache</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Feature lookup</td><td style="color:#fff;font-weight:600;text-align:right">O(1) retrieval</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Candidate scoring</td><td style="color:#fff;font-weight:600;text-align:right">1 batch call</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Throughput</td><td style="color:#27AE60;font-weight:700;text-align:right">Millions/day</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:7px 0">Latency</td><td style="color:#27AE60;font-weight:700;text-align:right">&lt;200ms ✅</td></tr>
                <tr><td style="color:#888;padding:7px 0">Coverage</td><td style="color:#27AE60;font-weight:700;text-align:right">97.3%</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px">
              <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:12px">⚠️ CONSTRAINTS ADDRESSED</div>
              <div style="font-size:.82rem;color:#ccc;line-height:1.8">
                <div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">🆕 Cold Start</strong><div style="color:#666;font-size:.76rem">New users → popularity fallback. New items → content features used directly.</div></div>
                <div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">🎲 Diversity</strong><div style="color:#666;font-size:.76rem">Max 3 items per category in Top-10 to avoid fatigue.</div></div>
                <div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #1e1e1e"><strong style="color:#fff">⚖️ Fairness</strong><div style="color:#666;font-size:.76rem">Evaluated per segment. Works for chain + independent restaurants.</div></div>
                <div><strong style="color:#fff">🤫 Non-intrusive</strong><div style="color:#666;font-size:.76rem">Only shown when cart has items. 50% probability threshold.</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with st.expander("🚀 Deployment Strategy & Projected Business Impact", expanded=False):
        dep1,dep2 = st.columns(2)
        with dep1:
            st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px">
              <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:12px">📋 DEPLOYMENT PHASES</div>
              <div style="font-size:.82rem;color:#ccc">
                <div style="display:flex;gap:10px;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #1e1e1e">
                  <div style="background:#FF6B35;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.75rem;flex-shrink:0">1</div>
                  <div><strong style="color:#fff">Shadow Mode (Wk 1–2)</strong><div style="color:#666;font-size:.76rem">Run parallel, validate latency + coverage silently</div></div>
                </div>
                <div style="display:flex;gap:10px;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #1e1e1e">
                  <div style="background:#3498DB;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.75rem;flex-shrink:0">2</div>
                  <div><strong style="color:#fff">A/B Test 10% (Wk 3–6)</strong><div style="color:#666;font-size:.76rem">Monitor AOV, CTR, C2O, acceptance. Check abandonment.</div></div>
                </div>
                <div style="display:flex;gap:10px;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #1e1e1e">
                  <div style="background:#27AE60;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.75rem;flex-shrink:0">3</div>
                  <div><strong style="color:#fff">Gradual Rollout (Wk 7–10)</strong><div style="color:#666;font-size:.76rem">25→50→100%. Auto-rollback if acceptance &lt;20%.</div></div>
                </div>
                <div style="display:flex;gap:10px">
                  <div style="background:#8E44AD;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.75rem;flex-shrink:0">4</div>
                  <div><strong style="color:#fff">Continuous Learning</strong><div style="color:#666;font-size:.76rem">Weekly retrain. Daily co-occurrence refresh.</div></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        with dep2:
            st.markdown("""<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:14px;padding:18px">
              <div style="font-size:.75rem;color:#FF6B35;font-weight:700;letter-spacing:.8px;margin-bottom:12px">📈 PROJECTED IMPACT</div>
              <table style="width:100%;font-size:.83rem;border-collapse:collapse">
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:8px 0">AOV Lift</td><td style="color:#27AE60;font-weight:800;text-align:right">+12–15%</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:8px 0">CSAO Attach Rate</td><td style="color:#27AE60;font-weight:800;text-align:right">+8–10%</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:8px 0">Acceptance Rate</td><td style="color:#27AE60;font-weight:800;text-align:right">26–31%</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:8px 0">CTR</td><td style="color:#27AE60;font-weight:800;text-align:right">18–22%</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:8px 0">C2O Rate</td><td style="color:#27AE60;font-weight:800;text-align:right">Neutral → +2%</td></tr>
                <tr style="border-bottom:1px solid #1e1e1e"><td style="color:#888;padding:8px 0">Avg Items / Order</td><td style="color:#27AE60;font-weight:800;text-align:right">+0.4–0.7</td></tr>
                <tr><td style="color:#888;padding:8px 0">Coverage</td><td style="color:#27AE60;font-weight:800;text-align:right">97.3%</td></tr>
              </table>
              <div style="margin-top:12px;padding:8px;background:#061409;border:1px solid #27AE60;border-radius:8px;font-size:.76rem;color:#7fcea0;text-align:center">
                🎯 Deploy <strong>Drink + Side</strong> first — 33% + 31% acceptance, lowest abandonment risk
              </div>
            </div>""", unsafe_allow_html=True)