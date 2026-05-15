"""
RELIANCE Industries — AttentionGRU Forecast Dashboard
Best fold: Fold 2 (balanced fit — T=0.2589, V=0.3241, gap=0.0652)
Trained up to: 2006-11-21
All data and images embedded — no file uploads needed
Streamlit Cloud compatible (Altair only)
"""

import os, warnings, base64
from io import StringIO
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDED FORECAST DATA  (from future_forecast_30days.csv)
# ─────────────────────────────────────────────────────────────────────────────
FORECAST_CSV = """Date,Predicted_Close,Pct_vs_last
2026-03-10,1407.17,0.55
2026-03-11,1409.57,0.72
2026-03-12,1411.97,0.89
2026-03-13,1414.37,1.06
2026-03-16,1416.77,1.23
2026-03-17,1419.17,1.41
2026-03-18,1421.60,1.58
2026-03-19,1424.04,1.75
2026-03-20,1426.47,1.93
2026-03-23,1428.93,2.10
2026-03-24,1431.41,2.28
2026-03-25,1433.85,2.45
2026-03-26,1436.31,2.63
2026-03-27,1438.77,2.81
2026-03-30,1441.26,2.98
2026-03-31,1443.74,3.16
2026-04-01,1446.25,3.34
2026-04-02,1448.80,3.52
2026-04-03,1451.35,3.70
2026-04-06,1454.06,3.90
2026-04-07,1456.79,4.09
2026-04-08,1459.55,4.29
2026-04-09,1462.33,4.49
2026-04-10,1465.14,4.69
2026-04-13,1468.03,4.90
2026-04-14,1470.92,5.10
2026-04-15,1473.81,5.31
2026-04-16,1476.77,5.52
2026-04-17,1479.95,5.75
2026-04-20,1483.27,5.99
"""

# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDED METRICS  (from metrics_summary.csv)
# ─────────────────────────────────────────────────────────────────────────────
METRICS_DATA = [
    {"Fold": "Fold 1", "MAPE_pct": 5.174, "R2": 0.9516, "DirAcc": 48.95, "best": False},
    {"Fold": "Fold 2", "MAPE_pct": 5.912, "R2": 0.8652, "DirAcc": 51.05, "best": True},
    {"Fold": "Fold 3", "MAPE_pct": 3.670, "R2": 0.8086, "DirAcc": 52.43, "best": False},
    {"Fold": "Fold 4", "MAPE_pct": 4.846, "R2": 0.9534, "DirAcc": 52.11, "best": False},
    {"Fold": "TEST",   "MAPE_pct": 3.675, "R2": 0.8463, "DirAcc": 51.70, "best": False},
]

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL CONFIG  (from model_config.json)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "model_name": "AttentionGRU_v2",
    "sequence_len": 60,
    "n_features": 13,
    "feature_cols": ["Open","High","Low","Close","Volume","Return","HL_Ratio",
                     "OC_Ratio","Momentum_5","Momentum_10","Vol_10","MA_Ratio","Log_Volume"],
    "target_col": "Log_Return",
    "trained_on_fold": 2,
    "train_end_date": "2006-11-21",
    "architecture": {"gru1":160,"gru2":96,"attn_units":64,"dense1":64,"dense2":32,
                     "dropout":0.15,"l2":5e-6,"recurrent_dropout":0.05},
    "training": {"loss":"huber","optimizer":"adam","init_lr":5e-5,"peak_lr":0.0008,
                 "warmup_epochs":5,"batch_size":16,"max_epochs":150,"early_stopping_patience":35},
    "best_fold_selection": "min(|train_loss − val_loss|) — most balanced fit",
}

# ─────────────────────────────────────────────────────────────────────────────
#  EMBED ALL 9 PLOT IMAGES AS BASE64
# ─────────────────────────────────────────────────────────────────────────────
def _load_b64(filename):
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "plots", filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        f"/mnt/user-data/uploads/{filename}",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

PLOT_B64 = {fn: _load_b64(fn) for fn in [
    "01_eda_overview.png",
    "04_fold_training_curves.png",
    "06_test_actual_vs_predicted.png",
    "07_residuals.png",
    "08_scatter.png",
    "09_val_vs_test.png",
    "10_attention_weights.png",
    "11_metrics_summary.png",
    "12_future_forecast_30d.png",
]}

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RELIANCE · AI Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (cinematic dark stock-market aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
NAVY     = "#05071A"
NAVY2    = "#080C24"
NAVY3    = "#0D1230"
NAVY4    = "#111840"
GLOW_TOP = "#FF8C00"
GLOW_MID = "#E05500"
LIME_GRN = "#00FF88"
SOFT_GRN = "#00CC66"
TICKER_R = "#FF3B5C"
SOFT_RED = "#CC2244"
ICE_BLUE = "#4FC3F7"
SILVER   = "#C8D6E5"
SILVER2  = "#8A9BC0"
DIM_BLUE = "#2A3560"
FAINT    = "#1A2245"
AMBER    = "#FFB347"
WHITE    = "#FFFFFF"

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

html,body,[class*="css"]{{
    background:{NAVY}; color:{SILVER};
    font-family:'Rajdhani',sans-serif; font-size:15px;
}}
.block-container{{padding:0!important;max-width:100%!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stSidebar"]{{display:none;}}
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{NAVY};}}
::-webkit-scrollbar-thumb{{background:{DIM_BLUE};border-radius:2px;}}

/* ── Hero ── */
.hero{{
    background:linear-gradient(135deg,{NAVY} 0%,{NAVY2} 30%,#0A0E28 55%,#160820 75%,#1A0A10 88%,#221006 100%);
    border-bottom:1px solid {DIM_BLUE}; padding:2rem 2.5rem 1.75rem;
    position:relative; overflow:hidden;
}}
.hero::before{{
    content:''; position:absolute; top:-80px; right:-60px;
    width:420px; height:420px;
    background:radial-gradient(circle,{GLOW_TOP}55 0%,{GLOW_MID}33 30%,{GLOW_TOP}11 55%,transparent 70%);
    pointer-events:none; animation:glow-pulse 4s ease-in-out infinite;
}}
.hero::after{{
    content:''; position:absolute; bottom:-40px; left:20%;
    width:300px; height:200px;
    background:radial-gradient(ellipse,{ICE_BLUE}18 0%,transparent 70%);
    pointer-events:none;
}}
@keyframes glow-pulse{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:.75;transform:scale(1.08);}}}}

/* ── Ticker tape ── */
.ticker-wrap{{overflow:hidden;background:{NAVY2};border-bottom:1px solid {DIM_BLUE};
  border-top:1px solid {DIM_BLUE};padding:.35rem 0;white-space:nowrap;}}
.ticker-inner{{display:inline-block;animation:ticker-scroll 45s linear infinite;
  font-family:'Share Tech Mono',monospace;font-size:.78rem;letter-spacing:.04em;}}
@keyframes ticker-scroll{{0%{{transform:translateX(0);}}100%{{transform:translateX(-50%);}}}}
.tick-item{{display:inline-block;margin:0 2.5rem;color:{SILVER2};}}
.tick-item .sym{{color:{AMBER};font-weight:700;margin-right:.4rem;}}
.tick-item .val{{color:{SILVER};}}
.tick-item .chg.up{{color:{LIME_GRN};}}
.tick-item .chg.dn{{color:{TICKER_R};}}

/* ── Brand ── */
.brand-tag{{font-family:'Orbitron',monospace;font-size:.65rem;color:{AMBER};
  letter-spacing:.35em;text-transform:uppercase;margin-bottom:.5rem;
  display:flex;align-items:center;gap:.6rem;}}
.brand-tag .live-dot{{width:7px;height:7px;background:{LIME_GRN};border-radius:50%;
  box-shadow:0 0 8px {LIME_GRN};animation:live-pulse 1.8s ease-in-out infinite;}}
@keyframes live-pulse{{0%,100%{{box-shadow:0 0 4px {LIME_GRN};}}50%{{box-shadow:0 0 14px {LIME_GRN},0 0 28px {LIME_GRN}66;}}}}
.hero-title{{font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;
  color:{WHITE};letter-spacing:.06em;text-transform:uppercase;margin:0 0 .3rem;
  text-shadow:0 0 30px {GLOW_TOP}88;}}
.hero-title span{{background:linear-gradient(90deg,{GLOW_TOP},{AMBER},{LIME_GRN});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.hero-sub{{font-family:'Rajdhani',sans-serif;font-size:.9rem;color:{SILVER2};
  letter-spacing:.12em;text-transform:uppercase;}}
.model-badge{{display:inline-flex;align-items:center;gap:.4rem;background:{NAVY4};
  border:1px solid {AMBER}55;color:{AMBER};font-family:'Share Tech Mono',monospace;
  font-size:.68rem;padding:.22rem .7rem;border-radius:3px;margin-left:1rem;
  vertical-align:middle;letter-spacing:.06em;}}
.fold-badge{{display:inline-flex;align-items:center;gap:.4rem;background:{LIME_GRN}11;
  border:1px solid {LIME_GRN}55;color:{LIME_GRN};font-family:'Share Tech Mono',monospace;
  font-size:.68rem;padding:.22rem .7rem;border-radius:3px;margin-left:.5rem;
  vertical-align:middle;letter-spacing:.06em;}}
.hero-stats{{display:flex;gap:2.5rem;margin-top:1.25rem;flex-wrap:wrap;}}
.hstat{{display:flex;flex-direction:column;gap:.15rem;}}
.hstat .hl{{font-family:'Share Tech Mono',monospace;font-size:.6rem;color:{SILVER2};
  letter-spacing:.12em;text-transform:uppercase;}}
.hstat .hv{{font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:700;color:{WHITE};}}
.hstat .hv.g{{color:{LIME_GRN};text-shadow:0 0 10px {LIME_GRN}66;}}
.hstat .hv.a{{color:{AMBER};text-shadow:0 0 10px {AMBER}55;}}
.hstat .hv.r{{color:{TICKER_R};text-shadow:0 0 10px {TICKER_R}55;}}
.hstat .hv.b{{color:{ICE_BLUE};text-shadow:0 0 10px {ICE_BLUE}55;}}

/* ── Section title ── */
.sec-title{{font-family:'Orbitron',monospace;font-size:.72rem;font-weight:700;
  color:{SILVER2};letter-spacing:.2em;text-transform:uppercase;
  margin:1.25rem 0 .8rem;display:flex;align-items:center;gap:.75rem;}}
.sec-title::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,{DIM_BLUE},{NAVY});}}
.sec-title.amber{{color:{AMBER};}}
.sec-title.green{{color:{LIME_GRN};}}

/* ── Metric cards ── */
.mc{{background:linear-gradient(145deg,{NAVY3},{NAVY4});border:1px solid {DIM_BLUE};
  border-radius:6px;padding:1rem 1.2rem;position:relative;overflow:hidden;}}
.mc::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,{AMBER}66,transparent);}}
.mc .lbl{{font-family:'Share Tech Mono',monospace;font-size:.58rem;color:{SILVER2};
  letter-spacing:.14em;text-transform:uppercase;margin-bottom:.4rem;}}
.mc .val{{font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:700;color:{WHITE};line-height:1;}}
.mc .val.g{{color:{LIME_GRN};text-shadow:0 0 12px {LIME_GRN}55;}}
.mc .val.a{{color:{AMBER};text-shadow:0 0 12px {AMBER}55;}}
.mc .val.r{{color:{TICKER_R};text-shadow:0 0 12px {TICKER_R}55;}}
.mc .val.b{{color:{ICE_BLUE};text-shadow:0 0 12px {ICE_BLUE}55;}}
.mc .sub{{font-size:.68rem;color:{SILVER2};margin-top:.3rem;font-weight:500;}}
.mc .sub.g{{color:{SOFT_GRN};}} .mc .sub.r{{color:{SOFT_RED};}}

/* ── Day toggle ── */
.day-toggle-wrap{{background:{NAVY2};border:1px solid {DIM_BLUE};
  border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;}}
.day-toggle-label{{font-family:'Share Tech Mono',monospace;font-size:.62rem;
  color:{SILVER2};letter-spacing:.15em;text-transform:uppercase;
  margin-bottom:.75rem;display:flex;align-items:center;gap:.6rem;}}
.day-toggle-label .badge{{background:{AMBER}22;border:1px solid {AMBER}55;
  color:{AMBER};border-radius:3px;padding:.1rem .4rem;font-size:.6rem;}}
.prog-wrap{{height:4px;background:{NAVY3};border-radius:2px;margin:.6rem 0;overflow:hidden;}}
.prog-fill{{height:4px;border-radius:2px;
  background:linear-gradient(90deg,{ICE_BLUE},{AMBER},{LIME_GRN});transition:width .4s ease;}}

/* ── Reveal card ── */
.reveal-card{{background:linear-gradient(135deg,{NAVY3},{NAVY4});
  border:1px solid {AMBER}55;border-radius:8px;padding:1.5rem 2rem;text-align:center;
  position:relative;overflow:hidden;margin-bottom:1rem;}}
.reveal-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,{AMBER},transparent);}}
.reveal-card .rc-day{{font-family:'Share Tech Mono',monospace;font-size:.65rem;
  color:{SILVER2};letter-spacing:.15em;text-transform:uppercase;margin-bottom:.5rem;}}
.reveal-card .rc-date{{font-family:'Orbitron',monospace;font-size:.8rem;color:{SILVER2};
  margin-bottom:.75rem;letter-spacing:.08em;}}
.reveal-card .rc-price{{font-family:'Orbitron',monospace;font-size:2.8rem;
  font-weight:900;line-height:1;margin-bottom:.5rem;}}
.reveal-card .rc-price.up{{color:{LIME_GRN};text-shadow:0 0 20px {LIME_GRN}66;}}
.reveal-card .rc-chg{{font-family:'Share Tech Mono',monospace;font-size:1.1rem;
  font-weight:700;margin-bottom:.75rem;}}
.reveal-card .rc-chg.up{{color:{LIME_GRN};}}
.reveal-card .rc-from{{font-family:'Share Tech Mono',monospace;font-size:.65rem;
  color:{SILVER2};letter-spacing:.08em;}}

/* ── Forecast table ── */
.fc-wrap{{background:linear-gradient(180deg,{NAVY3},{NAVY2});
  border:1px solid {DIM_BLUE};border-radius:6px;overflow:hidden;}}
.fc-hdr{{display:grid;grid-template-columns:0.4fr 1.2fr 1fr 1fr;padding:.5rem .85rem;
  background:{NAVY4};border-bottom:1px solid {DIM_BLUE};
  font-family:'Share Tech Mono',monospace;font-size:.58rem;color:{SILVER2};
  letter-spacing:.12em;text-transform:uppercase;}}
.fc-row{{display:grid;grid-template-columns:0.4fr 1.2fr 1fr 1fr;padding:.42rem .85rem;
  border-bottom:1px solid {FAINT};font-size:.78rem;font-weight:500;transition:background .12s;}}
.fc-row:hover{{background:{NAVY4};}}
.fc-row:last-child{{border-bottom:none;}}
.fc-row.highlighted{{background:{AMBER}11;border-left:2px solid {AMBER};}}
.fc-row .num{{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:{DIM_BLUE};}}
.fc-row .date{{font-family:'Share Tech Mono',monospace;font-size:.72rem;color:{SILVER2};}}
.fc-row .price{{text-align:right;font-family:'Share Tech Mono',monospace;
  color:{SILVER};font-size:.78rem;}}
.fc-row .pct{{text-align:right;font-family:'Share Tech Mono',monospace;font-size:.74rem;}}
.fc-row .pct.up{{color:{LIME_GRN};text-shadow:0 0 6px {LIME_GRN}55;}}
.fc-row .pct.dim{{color:{DIM_BLUE};}}

/* ── Plot frame ── */
.plot-frame{{background:{NAVY3};border:1px solid {DIM_BLUE};
  border-radius:6px;padding:.5rem;overflow:hidden;margin-bottom:.9rem;}}
.plot-cap{{font-family:'Share Tech Mono',monospace;font-size:.57rem;color:{SILVER2};
  letter-spacing:.1em;text-transform:uppercase;text-align:center;margin-top:.35rem;}}

/* ── Architecture ── */
.arch-row{{display:flex;align-items:center;gap:.75rem;margin-bottom:3px;}}
.arch-box{{font-family:'Share Tech Mono',monospace;font-size:.65rem;font-weight:700;
  padding:.3rem .75rem;border-radius:3px;min-width:105px;text-align:center;
  border:1px solid;letter-spacing:.05em;}}
.arch-desc{{font-size:.75rem;font-weight:500;color:{SILVER2};}}
.arch-arrow{{font-size:.85rem;color:{DIM_BLUE};margin-left:50px;line-height:.5;margin-bottom:1px;}}

/* ── HP table ── */
.hp-table{{background:{NAVY3};border:1px solid {DIM_BLUE};border-radius:6px;overflow:hidden;}}
.hp-row{{display:flex;justify-content:space-between;align-items:center;
  padding:.4rem .9rem;border-bottom:1px solid {FAINT};font-size:.78rem;font-weight:500;}}
.hp-row:last-child{{border-bottom:none;}}
.hp-k{{color:{SILVER2};letter-spacing:.05em;}}
.hp-v{{font-family:'Share Tech Mono',monospace;color:{AMBER};font-size:.75rem;}}
.hp-v.green{{color:{LIME_GRN};}}

/* ── Pills ── */
.pill{{display:inline-block;background:{NAVY4};border:1px solid {DIM_BLUE};
  border-radius:3px;padding:.2rem .55rem;font-family:'Share Tech Mono',monospace;
  font-size:.65rem;color:{SILVER2};margin:.15rem;letter-spacing:.04em;}}

/* ── Disclaimer ── */
.disc{{background:{GLOW_TOP}0D;border:1px solid {GLOW_TOP}33;border-radius:4px;
  padding:.6rem .9rem;font-size:.72rem;color:{AMBER};margin-top:1rem;
  font-weight:500;letter-spacing:.04em;line-height:1.6;}}

/* ── Streamlit overrides ── */
div[data-testid="stButton"]>button{{
    background:linear-gradient(135deg,{NAVY3},{NAVY4});border:1px solid {DIM_BLUE};
    color:{SILVER};font-family:'Rajdhani',sans-serif;font-weight:600;
    letter-spacing:.1em;text-transform:uppercase;transition:.2s;
}}
div[data-testid="stButton"]>button:hover{{border-color:{AMBER}88;color:{AMBER};}}
div[data-testid="stButton"]>button[kind="primary"]{{
    border-color:{AMBER}88;color:{AMBER};
    background:linear-gradient(135deg,{NAVY3},{NAVY4});
}}
div[data-testid="stToggle"] label{{
    font-family:'Share Tech Mono',monospace!important;
    font-size:.75rem!important;letter-spacing:.08em!important;
    color:{SILVER2}!important;text-transform:uppercase!important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR  = os.path.join(_APP_DIR, "outputs", "plots")
MODEL_DIR = os.path.join(_APP_DIR, "outputs", "models")


def plot_img(filename, cap=""):
    b = PLOT_B64.get(filename)
    if b is None:
        path = os.path.join(PLOT_DIR, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                b = base64.b64encode(f.read()).decode()
    if b:
        cap_html = f"<div class='plot-cap'>{cap}</div>" if cap else ""
        st.markdown(
            f"<div class='plot-frame'>"
            f"<img src='data:image/png;base64,{b}' "
            f"style='width:100%;border-radius:3px;display:block;'>"
            f"{cap_html}</div>",
            unsafe_allow_html=True,
        )


def dark_alt(chart, h=300):
    return (
        chart.properties(height=h)
        .configure(background="transparent", view=alt.ViewConfig(stroke=DIM_BLUE))
        .configure_axis(gridColor=FAINT, domainColor=DIM_BLUE,
                        labelColor=SILVER2, titleColor=SILVER2,
                        labelFont="Share Tech Mono, monospace",
                        titleFont="Share Tech Mono, monospace",
                        labelFontSize=10, titleFontSize=10)
        .configure_legend(labelColor=SILVER2, titleColor=SILVER2,
                          labelFont="Share Tech Mono, monospace",
                          titleFont="Share Tech Mono, monospace",
                          strokeColor=DIM_BLUE)
        .configure_title(color=SILVER2, font="Share Tech Mono, monospace", fontSize=10)
    )


def mc(col, lbl, val, sub="", vcls="", scls=""):
    col.markdown(
        f"<div class='mc'><div class='lbl'>{lbl}</div>"
        f"<div class='val {vcls}'>{val}</div>"
        f"{'<div class=sub '+scls+'>'+sub+'</div>' if sub else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_forecast():
    df = pd.read_csv(StringIO(FORECAST_CSV), parse_dates=["Date"])
    df["direction"] = df["Pct_vs_last"].apply(lambda x: "UP" if x >= 0 else "DN")
    df["Day"] = df.index + 1
    return df

@st.cache_data
def load_reliance():
    paths = [
        os.path.join(_APP_DIR, "RELIANCE.csv"),
        os.path.join(_APP_DIR, "data", "RELIANCE.csv"),
        os.path.join(_APP_DIR, "outputs", "data", "RELIANCE.csv"),
        "RELIANCE.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df = df[cols].apply(pd.to_numeric, errors="coerce")
            df["Volume"] = df["Volume"].replace(0, np.nan)
            df.dropna(inplace=True)
            return df
    return None

@st.cache_resource(show_spinner="Initialising AttentionGRU model…")
def load_model():
    try:
        import tensorflow as tf
        class BahdanauAttention(tf.keras.layers.Layer):
            def __init__(self, units, **kw):
                super().__init__(**kw); self.units = units
                self.W = tf.keras.layers.Dense(units)
                self.V = tf.keras.layers.Dense(1)
            def call(self, h):
                s = self.V(tf.nn.tanh(self.W(h)))
                s -= tf.reduce_max(s, axis=1, keepdims=True)
                a  = tf.nn.softmax(s, axis=1)
                return tf.reduce_sum(a * h, axis=1), a
            def get_config(self):
                c = super().get_config(); c["units"] = self.units; return c
        mdl = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "final_model.keras"),
            custom_objects={"BahdanauAttention": BahdanauAttention})
        fsc = joblib.load(os.path.join(MODEL_DIR, "final_f_scaler.pkl"))
        tsc = joblib.load(os.path.join(MODEL_DIR, "final_t_scaler.pkl"))
        return mdl, fsc, tsc, True
    except Exception:
        return None, None, None, False


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────
fc_df   = load_forecast()
df_raw  = load_reliance()
mdl, fsc, tsc, model_ok = load_model()

LAST_CLOSE = float(df_raw["Close"].iloc[-1]) if df_raw is not None else 1399.50
LAST_DATE  = df_raw.index[-1].strftime("%d %b %Y") if df_raw is not None else "09 Mar 2026"
PRED_30    = float(fc_df["Predicted_Close"].iloc[-1])
PCT_30     = float(fc_df["Pct_vs_last"].iloc[-1])
DAYS_UP    = int((fc_df["direction"] == "UP").sum())
DAYS_DN    = 30 - DAYS_UP

# Session state
if "tab"          not in st.session_state: st.session_state.tab = "forecast"
if "sel_day"      not in st.session_state: st.session_state.sel_day = 1
if "show_all_fc"  not in st.session_state: st.session_state.show_all_fc = False

# ─────────────────────────────────────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
arr30 = "▲" if PCT_30 >= 0 else "▼"
st.markdown(f"""
<div class='hero'>
  <div class='brand-tag'><span class='live-dot'></span>
    NSE · RELIANCE INDUSTRIES LIMITED · AI FORECAST ENGINE
  </div>
  <div style='display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap;'>
    <h1 class='hero-title'>RELIANCE <span>STOCK FORECAST</span></h1>
    <span class='model-badge'>AttentionGRU v4</span>
    <span class='fold-badge'>★ FOLD 2 — BALANCED FIT</span>
  </div>
  <div class='hero-sub'>Walk-Forward · Bahdanau Attention · Huber Loss · min|train−val| selection</div>
  <div class='hero-stats'>
    <div class='hstat'><span class='hl'>Last Close</span>
      <span class='hv'>₹{LAST_CLOSE:,.2f}</span></div>
    <div class='hstat'><span class='hl'>30-Day Target</span>
      <span class='hv {"g" if PCT_30>=0 else "r"}'>{arr30} ₹{PRED_30:,.2f}</span></div>
    <div class='hstat'><span class='hl'>Expected Return</span>
      <span class='hv {"g" if PCT_30>=0 else "r"}'>{arr30} {abs(PCT_30):.2f}%</span></div>
    <div class='hstat'><span class='hl'>Bullish / Bearish</span>
      <span class='hv'>
        <span style='color:{LIME_GRN}'>{DAYS_UP}▲</span>
        <span style='color:{SILVER2}'> · </span>
        <span style='color:{TICKER_R}'>{DAYS_DN}▼</span>
      </span></div>
    <div class='hstat'><span class='hl'>Test MAPE</span>
      <span class='hv g'>3.675%</span></div>
    <div class='hstat'><span class='hl'>Test R²</span>
      <span class='hv b'>0.8463</span></div>
    <div class='hstat'><span class='hl'>Dir. Accuracy</span>
      <span class='hv a'>51.7%</span></div>
    <div class='hstat'><span class='hl'>Best Fold</span>
      <span class='hv a'>#2 ★ Balanced</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Ticker tape ───────────────────────────────────────────────────────────────
tickers = [
    ("RELIANCE",   f"₹{LAST_CLOSE:,.2f}",  "▲ +0.89%",  "up"),
    ("NIFTY50",    "22,147",               "▲ +0.44%",   "up"),
    ("SENSEX",     "72,988",               "▲ +0.38%",   "up"),
    ("30D TARGET", f"₹{PRED_30:,.2f}",     f"{arr30} {PCT_30:.2f}%", "up" if PCT_30>=0 else "dn"),
    ("TEST MAPE",  "3.675%",               "Fold 2 ★",   "up"),
    ("R²",         "0.8463",               "Test set",    "up"),
    ("DIR ACC",    "51.7%",                "+1.7%",       "up"),
    ("BEST FOLD",  "Fold 2",               "Balanced ✓",  "up"),
    ("TRAIN END",  "2006-11-21",           "Fold #2",     "up"),
    ("SEQ LEN",    "60 days",              "13 features", "up"),
    ("GRU",        "160/96",               "Bahdanau",    "up"),
]
tape = " &nbsp;|&nbsp; ".join(
    f"<span class='tick-item'>"
    f"<span class='sym'>{t[0]}</span>"
    f"<span class='val'>{t[1]}</span> "
    f"<span class='chg {t[3]}'>{t[2]}</span>"
    f"</span>" for t in tickers
)
st.markdown(f"""
<div class='ticker-wrap'>
  <div class='ticker-inner'>{tape}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{tape}</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB NAV  (4 tabs — no Analysis Plots tab)
# ─────────────────────────────────────────────────────────────────────────────
TABS = [
    ("forecast", "📈  Forecast"),
    ("history",  "📉  Price History"),
    ("metrics",  "📊  Metrics"),
    ("arch",     "⚙   Architecture"),
]
tab_cols = st.columns(len(TABS) + 7)
for i, (key, label) in enumerate(TABS):
    active = st.session_state.tab == key
    if tab_cols[i].button(label, key=f"tb_{key}",
                          type="primary" if active else "secondary"):
        st.session_state.tab = key
        st.rerun()

st.markdown(f"<hr style='margin:0;border:none;border-top:1px solid {DIM_BLUE};'>",
            unsafe_allow_html=True)

TAB = st.session_state.tab


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — FORECAST  (day-by-day toggle)
# ══════════════════════════════════════════════════════════════════════════════
if TAB == "forecast":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)

    sel     = st.session_state.sel_day
    row     = fc_df[fc_df["Day"] == sel].iloc[0]
    sel_price = float(row["Predicted_Close"])
    sel_pct   = float(row["Pct_vs_last"])
    sel_dir   = row["direction"]
    sel_date  = pd.to_datetime(row["Date"]).strftime("%A, %d %b %Y")
    sel_arr   = "▲" if sel_dir == "UP" else "▼"
    prc_cls   = "up" if sel_dir == "UP" else "dn"

    # ── Summary cards for selected day ───────────────────────────────────────
    st.markdown(f"<div class='sec-title amber'>Day {sel} Forecast Snapshot</div>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    mc(c1, "Last Close",    f"₹{LAST_CLOSE:,.2f}", LAST_DATE)
    days_up_so = int((fc_df[fc_df["Day"] <= sel]["direction"] == "UP").sum())
    c2.markdown(f"""<div class='mc'><div class='lbl'>Day {sel} Price</div>
      <div class='val {prc_cls}'>₹{sel_price:,.2f}</div>
      <div class='sub {prc_cls}'>{sel_arr} {abs(sel_pct):.2f}% vs today</div></div>""",
      unsafe_allow_html=True)
    mc(c3, f"Bullish Days 1–{sel}", str(days_up_so), f"of {sel} days revealed", "g")
    mc(c4, f"Bearish Days 1–{sel}", str(sel - days_up_so), f"of {sel} days revealed", "r")
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    # ── Day selector toggle ───────────────────────────────────────────────────
    st.markdown(f"""<div class='day-toggle-wrap'>
      <div class='day-toggle-label'>
        Select Forecast Day
        <span class='badge'>Day {sel} / 30 selected</span>
      </div>
      <div class='prog-wrap'>
        <div class='prog-fill' style='width:{sel/30*100:.1f}%'></div>
      </div>
    </div>""", unsafe_allow_html=True)

    for row_start in range(0, 30, 10):
        btn_cols = st.columns(10)
        for j, day_num in enumerate(range(row_start + 1, row_start + 11)):
            if day_num > 30: break
            is_active = (day_num == sel)
            if btn_cols[j].button(str(day_num), key=f"day_{day_num}",
                                  type="primary" if is_active else "secondary"):
                st.session_state.sel_day = day_num
                st.rerun()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Reveal card ───────────────────────────────────────────────────────────
    st.markdown(f"""<div class='reveal-card'>
      <div class='rc-day'>Day {sel} of 30</div>
      <div class='rc-date'>{sel_date}</div>
      <div class='rc-price up'>₹{sel_price:,.2f}</div>
      <div class='rc-chg up'>{sel_arr} {abs(sel_pct):.2f}% vs ₹{LAST_CLOSE:,.2f}</div>
      <div class='rc-from'>
        Base close · {LAST_DATE} · AttentionGRU v4 · Fold #2 (Balanced Fit)
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Chart + Table ─────────────────────────────────────────────────────────
    left, right = st.columns([3, 1], gap="medium")

    with left:
        show_all = st.toggle(
            "Show complete 30-day forecast on chart",
            value=st.session_state.show_all_fc, key="toggle_show_all"
        )
        st.session_state.show_all_fc = show_all
        visible_n = 30 if show_all else sel

        st.markdown(
            f"<div class='sec-title'>Price Trajectory — Historical + "
            f"{'All 30 Days' if show_all else f'Days 1–{sel}'}</div>",
            unsafe_allow_html=True,
        )

        chart_rows = []
        if df_raw is not None:
            tail = df_raw["Close"].iloc[-90:].reset_index()
            tail.columns = ["Date","Price"]
            tail["Series"] = "Historical"; tail["Dir"] = "—"
            chart_rows.append(tail)

        fc_visible = fc_df[fc_df["Day"] <= visible_n].copy()
        fc_plot = fc_visible[["Date","Predicted_Close","direction"]].copy()
        fc_plot.columns = ["Date","Price","Dir"]; fc_plot["Series"] = "Forecast"
        chart_rows.append(fc_plot)

        pdf = pd.concat(chart_rows, ignore_index=True)
        pdf["Date"] = pd.to_datetime(pdf["Date"])

        hist_line = alt.Chart(pdf[pdf["Series"]=="Historical"]).mark_line(
            color=ICE_BLUE, strokeWidth=1.6, opacity=.85
        ).encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", title="")),
            y=alt.Y("Price:Q", axis=alt.Axis(title="Price (₹)", format=",.0f"),
                    scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Price:Q", format=",.2f", title="₹")]
        )
        fc_band_df = fc_visible.copy()
        fc_band_df["upper"] = fc_band_df["Predicted_Close"] * 1.012
        fc_band_df["lower"] = fc_band_df["Predicted_Close"] * 0.988
        band = alt.Chart(pd.concat([
            fc_band_df[["Date","upper"]].rename(columns={"upper":"Price"}),
            fc_band_df[["Date","lower"]].rename(columns={"lower":"Price"}).iloc[::-1]
        ])).mark_area(color=AMBER, opacity=.07).encode(x="Date:T", y="Price:Q")

        fc_line = alt.Chart(pdf[pdf["Series"]=="Forecast"]).mark_line(
            color=AMBER, strokeWidth=2.2, strokeDash=[5,2]
        ).encode(x="Date:T", y="Price:Q")

        fc_dots = alt.Chart(pdf[pdf["Series"]=="Forecast"]).mark_circle(size=60).encode(
            x="Date:T", y="Price:Q",
            color=alt.Color("Dir:N",
                scale=alt.Scale(domain=["UP","DN"], range=[LIME_GRN, TICKER_R]),
                legend=alt.Legend(title="Direction")),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Price:Q", format=",.2f", title="₹"), "Dir:N"]
        )
        sel_df = fc_df[fc_df["Day"] == sel][["Date","Predicted_Close"]].copy()
        sel_df.columns = ["Date","Price"]
        sel_dot = alt.Chart(sel_df).mark_circle(size=200, color=AMBER, opacity=1).encode(
            x="Date:T", y="Price:Q",
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Price:Q", format=",.2f", title="Selected Day ₹")]
        )
        st.altair_chart(dark_alt(band + hist_line + fc_line + fc_dots + sel_dot, h=340),
                        use_container_width=True)

        # Pct change bars
        st.markdown(f"<div class='sec-title'>Cumulative % Change — Days 1–{visible_n}</div>",
                    unsafe_allow_html=True)
        pct_df = fc_visible[["Date","Pct_vs_last","direction","Day"]].copy()
        pct_bars = alt.Chart(pct_df).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%d %b", title="")),
            y=alt.Y("Pct_vs_last:Q", axis=alt.Axis(title="% vs last close")),
            color=alt.Color("direction:N",
                scale=alt.Scale(domain=["UP","DN"], range=[LIME_GRN, TICKER_R]),
                legend=None),
            opacity=alt.condition(
                alt.datum.Day == sel, alt.value(1.0), alt.value(0.55)),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Pct_vs_last:Q", format=".2f", title="Chg%"),
                     alt.Tooltip("Day:Q", title="Day")]
        )
        zero_r = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(
            color=SILVER2, strokeDash=[4,3], opacity=.4
        ).encode(y="y:Q")
        st.altair_chart(dark_alt(pct_bars + zero_r, h=170), use_container_width=True)

        # Static forecast plot
        plot_img("12_future_forecast_30d.png", "30-Day Forecast — Training Run Output")

        st.markdown(f"""<div class='disc'>
          ⚠ NOT FINANCIAL ADVICE · Research prototype only.
          Reliability degrades rapidly beyond day 5.
          Best fold selected by min|train−val loss| — Fold 2 (Good Fit: T=0.2589, V=0.3241).
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"<div class='sec-title'>30-Day Table</div>", unsafe_allow_html=True)
        st.markdown("<div class='fc-wrap'>", unsafe_allow_html=True)
        st.markdown("""<div class='fc-hdr'>
          <span>#</span><span>Date</span>
          <span style='text-align:right;display:block'>₹ Price</span>
          <span style='text-align:right;display:block'>Chg%</span>
        </div>""", unsafe_allow_html=True)

        for _, r in fc_df.iterrows():
            day_num = int(r["Day"])
            d   = pd.to_datetime(r["Date"]).strftime("%d %b")
            p   = f"{float(r['Predicted_Close']):,.2f}"
            pv  = float(r["Pct_vs_last"])
            ps  = f"+{pv:.2f}%" if pv >= 0 else f"{pv:.2f}%"
            arr_sym = "▲" if r["direction"] == "UP" else "▼"
            is_sel  = (day_num == sel)
            is_hide = (day_num > sel) and not show_all
            row_cls = "fc-row highlighted" if is_sel else "fc-row"
            p_disp  = "—" if is_hide else p
            ps_disp = "—" if is_hide else f"{arr_sym} {ps}"
            pc_cls  = "dim" if is_hide else "up"
            st.markdown(f"""<div class='{row_cls}'>
              <span class='num'>{day_num}</span>
              <span class='date'>{d}</span>
              <span class='price'>{p_disp}</span>
              <span class='pct {pc_cls}'>{ps_disp}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PRICE HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif TAB == "history":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title amber'>Market Data — RELIANCE Industries (NSE)</div>",
                unsafe_allow_html=True)

    if df_raw is not None:
        last_1y  = df_raw["Close"].iloc[-252] if len(df_raw) >= 252 else df_raw["Close"].iloc[0]
        ret_1y   = (LAST_CLOSE / last_1y - 1) * 100
        h1,h2,h3,h4 = st.columns(4)
        mc(h1,"Current Price",   f"₹{LAST_CLOSE:,.2f}", LAST_DATE)
        mc(h2,"52-Week High",
           f"₹{df_raw['High'].iloc[-252:].max():,.2f}", "52-week high", "a")
        mc(h3,"52-Week Low",
           f"₹{df_raw['Low'].iloc[-252:].min():,.2f}",  "52-week low",  "r")
        mc(h4,"1-Year Return",
           f"{'▲' if ret_1y>=0 else '▼'} {abs(ret_1y):.1f}%",
           "vs 1 year ago", "g" if ret_1y>=0 else "r")

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        WIN_MAP    = {"1M":22,"3M":66,"6M":130,"1Y":252,"3Y":756,"5Y":1260,"ALL":len(df_raw)}
        WIN_LABELS = list(WIN_MAP.keys())
        sel_label  = st.select_slider("Chart window", options=WIN_LABELS, value="1Y")
        sel_win_n  = WIN_MAP[sel_label]

        hist = df_raw.iloc[-sel_win_n:].copy().reset_index()
        hist.columns = ["Date"] + list(hist.columns[1:])
        hist["Date"]  = pd.to_datetime(hist["Date"])
        hist["Dir"]   = (hist["Close"] >= hist["Open"]).map({True:"UP",False:"DN"})
        hist["MA20"]  = hist["Close"].rolling(20).mean()
        hist["MA50"]  = hist["Close"].rolling(50).mean()
        hist["Ret"]   = hist["Close"].pct_change() * 100
        hist["Series"]= "Historical"

        # Forecast extension toggle
        show_fc_ext = st.toggle(
            "Show 30-day forecast extension on chart", value=True, key="hist_fc_toggle"
        )

        st.markdown(f"<div class='sec-title'>Closing Price & Moving Averages</div>",
                    unsafe_allow_html=True)

        close_line = alt.Chart(hist).mark_line(color=ICE_BLUE, strokeWidth=1.5).encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", title="")),
            y=alt.Y("Close:Q", axis=alt.Axis(title="Price (₹)", format=",.0f"),
                    scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Close:Q", format=",.2f", title="₹")]
        )
        ma20 = alt.Chart(hist.dropna(subset=["MA20"])).mark_line(
            color=AMBER, strokeWidth=1.2, strokeDash=[4,2], opacity=.8
        ).encode(x="Date:T", y=alt.Y("MA20:Q"))
        ma50 = alt.Chart(hist.dropna(subset=["MA50"])).mark_line(
            color=TICKER_R, strokeWidth=1, strokeDash=[5,3], opacity=.7
        ).encode(x="Date:T", y=alt.Y("MA50:Q"))

        chart_layers = [close_line, ma20, ma50]

        if show_fc_ext:
            fc_ext = fc_df.copy()
            fc_ext["Date"]  = pd.to_datetime(fc_ext["Date"])
            fc_ext["upper"] = fc_ext["Predicted_Close"] * 1.015
            fc_ext["lower"] = fc_ext["Predicted_Close"] * 0.985
            band_data = pd.concat([
                fc_ext[["Date","upper"]].rename(columns={"upper":"Close"}),
                fc_ext[["Date","lower"]].rename(columns={"lower":"Close"}).iloc[::-1]
            ])
            fc_band_c = alt.Chart(band_data).mark_area(
                color=AMBER, opacity=.08
            ).encode(x="Date:T", y="Close:Q")
            fc_line_c = alt.Chart(fc_ext).mark_line(
                color=AMBER, strokeWidth=2, strokeDash=[5,2]
            ).encode(x="Date:T", y=alt.Y("Predicted_Close:Q"),
                     tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                              alt.Tooltip("Predicted_Close:Q", format=",.2f", title="Forecast ₹")])
            chart_layers = [fc_band_c] + chart_layers + [fc_line_c]

        price_chart = chart_layers[0]
        for layer in chart_layers[1:]:
            price_chart = price_chart + layer
        st.altair_chart(dark_alt(price_chart, h=340), use_container_width=True)

        lc1, lc2 = st.columns(2, gap="medium")
        with lc1:
            st.markdown(f"<div class='sec-title'>Volume</div>", unsafe_allow_html=True)
            vb = alt.Chart(hist[hist["Volume"] > 0]).mark_bar(opacity=.78).encode(
                x=alt.X("Date:T", axis=alt.Axis(format="%b %y", title="")),
                y=alt.Y("Volume:Q", axis=alt.Axis(title="Volume")),
                color=alt.Color("Dir:N",
                    scale=alt.Scale(domain=["UP","DN"], range=[SOFT_GRN, SOFT_RED]),
                    legend=None),
                tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                         alt.Tooltip("Volume:Q", format=",.0f")]
            )
            st.altair_chart(dark_alt(vb, h=220), use_container_width=True)

        with lc2:
            st.markdown(f"<div class='sec-title'>Daily Return Distribution</div>",
                        unsafe_allow_html=True)
            ret_data = hist.dropna(subset=["Ret"])
            rh = alt.Chart(ret_data).mark_bar(color=ICE_BLUE, opacity=.75).encode(
                x=alt.X("Ret:Q", bin=alt.Bin(maxbins=55),
                         axis=alt.Axis(title="Daily Return (%)")),
                y=alt.Y("count()", axis=alt.Axis(title="Count")),
                tooltip=[alt.Tooltip("Ret:Q", bin=True, title="Return %"), "count()"]
            )
            mv = ret_data["Ret"].mean()
            mr = alt.Chart(pd.DataFrame({"x":[mv]})).mark_rule(
                color=AMBER, strokeDash=[4,3]
            ).encode(x="x:Q")
            st.altair_chart(dark_alt(rh + mr, h=220), use_container_width=True)

        # Combined historical + forecast table
        st.markdown(f"<div class='sec-title'>Recent Prices + Forecast Extension</div>",
                    unsafe_allow_html=True)
        last_hist = df_raw.iloc[-8:].reset_index()[["Date","Close","High","Low","Volume"]]

        tbl_cols = st.columns([2,1.5,1.5,1.5,2,1.5,1.5])
        for col_w, h in zip(tbl_cols, ["Date","Close ₹","High ₹","Low ₹","Volume","Chg%","Type"]):
            col_w.markdown(
                f"<div style='font-family:Share Tech Mono,monospace;font-size:.6rem;"
                f"color:{SILVER2};letter-spacing:.1em;text-transform:uppercase;"
                f"padding:.3rem 0;border-bottom:1px solid {DIM_BLUE};'>{h}</div>",
                unsafe_allow_html=True
            )
        for _, row_h in last_hist.iterrows():
            vals = [
                row_h["Date"].strftime("%d %b %Y") if hasattr(row_h["Date"],"strftime") else str(row_h["Date"])[:10],
                f"₹{float(row_h['Close']):,.2f}", f"₹{float(row_h['High']):,.2f}",
                f"₹{float(row_h['Low']):,.2f}", f"{int(row_h['Volume']):,}", "—", "Historical"
            ]
            clrs = [SILVER2, SILVER, SILVER2, SILVER2, SILVER2, SILVER2, ICE_BLUE]
            for col_w, v, clr in zip(tbl_cols, vals, clrs):
                col_w.markdown(
                    f"<div style='font-family:Share Tech Mono,monospace;font-size:.72rem;"
                    f"color:{clr};padding:.25rem 0;border-bottom:1px solid {FAINT};'>{v}</div>",
                    unsafe_allow_html=True
                )
        for _, row_f in fc_df.iterrows():
            pv  = float(row_f["Pct_vs_last"])
            ps  = f"+{pv:.2f}%" if pv >= 0 else f"{pv:.2f}%"
            vals = [pd.to_datetime(row_f["Date"]).strftime("%d %b %Y"),
                    f"₹{float(row_f['Predicted_Close']):,.2f}", "—","—","—",
                    f"{'▲' if pv>=0 else '▼'} {ps}", "Forecast"]
            clrs = [SILVER2, AMBER, SILVER2, SILVER2, SILVER2,
                    LIME_GRN if pv>=0 else TICKER_R, AMBER]
            for col_w, v, clr in zip(tbl_cols, vals, clrs):
                col_w.markdown(
                    f"<div style='font-family:Share Tech Mono,monospace;font-size:.72rem;"
                    f"color:{clr};padding:.25rem 0;border-bottom:1px solid {FAINT};'>{v}</div>",
                    unsafe_allow_html=True
                )

        st.markdown(f"<div class='sec-title' style='margin-top:1.5rem'>Summary Statistics</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            hist[["Open","High","Low","Close","Volume"]].describe().T.style.format("{:.2f}"),
            use_container_width=True
        )
    else:
        st.warning("RELIANCE.csv not found. Place it in the same folder as app.py.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif TAB == "metrics":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title amber'>Hold-Out Test Set Performance</div>",
                unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.columns(5)
    mc(t1,"Test MAPE",    "3.675%", "Mean Absolute % Error",  "g")
    mc(t2,"Test R²",      "0.8463", "Variance explained",     "b")
    mc(t3,"Directional",  "51.7%",  "vs 50% random baseline", "a")
    mc(t4,"Edge",         "+1.7%",  "over random baseline",   "g")
    mc(t5,"Best Fold",    "#2 ★",   "Balanced fit selection",  "a")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # Fold selection explanation
    st.markdown(f"""
    <div style='background:{LIME_GRN}0A;border:1px solid {LIME_GRN}33;border-radius:6px;
         padding:.8rem 1.1rem;font-family:Share Tech Mono,monospace;font-size:.72rem;
         color:{LIME_GRN};margin-bottom:1rem;line-height:1.8;'>
      ★ FOLD 2 SELECTED — BALANCED FIT CRITERION<br>
      <span style='color:{SILVER2}'>
      Selection method: min(|train_loss − val_loss|) across all folds.<br>
      Fold 2: T=0.2589  V=0.3241  gap=+0.0652  [Good Fit ✓]<br>
      vs Fold 1: T=0.2594  V=0.1447  gap=−0.1147  [UNDERFITTING] ·
      Fold 3: T=0.2903  V=0.1653  gap=−0.1250  [UNDERFITTING] ·
      Fold 4: T=0.2989  V=0.2323  gap=−0.0666  [UNDERFITTING]
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='sec-title'>Walk-Forward Fold Comparison</div>",
                unsafe_allow_html=True)

    mdf     = pd.DataFrame(METRICS_DATA)
    fold_df = mdf[mdf["Fold"] != "TEST"].copy()
    fold_df["Color"] = fold_df["best"].apply(lambda x: LIME_GRN if x else ICE_BLUE)

    ch1, ch2, ch3 = st.columns(3, gap="medium")

    def make_bar(data, y_col, title, y_title, domain=None):
        opts = {"cornerRadiusTopLeft":4,"cornerRadiusTopRight":4}
        enc = dict(
            x=alt.X("Fold:N", axis=alt.Axis(title="")),
            y=alt.Y(f"{y_col}:Q",
                    axis=alt.Axis(title=y_title),
                    scale=alt.Scale(domain=domain) if domain else alt.Scale()),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Fold:N", alt.Tooltip(f"{y_col}:Q", format=".3f")]
        )
        return alt.Chart(data).mark_bar(**opts).encode(**enc).properties(title=title)

    with ch1:
        st.altair_chart(dark_alt(make_bar(fold_df,"MAPE_pct","MAPE %","MAPE %"), h=240),
                        use_container_width=True)
    with ch2:
        st.altair_chart(dark_alt(make_bar(fold_df,"R2","R² Score","R²",[0,1]), h=240),
                        use_container_width=True)
    with ch3:
        r50 = alt.Chart(pd.DataFrame({"y":[50]})).mark_rule(
            color=TICKER_R, strokeDash=[5,3], opacity=.7
        ).encode(y="y:Q")
        b3 = make_bar(fold_df,"DirAcc","Directional Accuracy %","Dir. Acc %",[45,56])
        st.altair_chart(dark_alt(b3 + r50, h=240), use_container_width=True)

    st.markdown(f"<div class='sec-title'>Full Metrics Table</div>", unsafe_allow_html=True)
    disp = mdf[["Fold","MAPE_pct","R2","DirAcc"]].copy()
    disp.columns = ["Fold","MAPE %","R²","Dir. Acc %"]
    st.dataframe(
        disp.style.format({"MAPE %":"{:.3f}","R²":"{:.4f}","Dir. Acc %":"{:.2f}"}),
        use_container_width=True, height=210
    )

    # All analysis plots
    st.markdown(f"<div class='sec-title' style='margin-top:1.5rem'>Analysis Plots</div>",
                unsafe_allow_html=True)
    plot_img("04_fold_training_curves.png",    "Walk-Forward Training Curves — All 4 Folds")
    plot_img("11_metrics_summary.png",          "Model Performance Summary — MAPE & Dir. Accuracy")
    plot_img("06_test_actual_vs_predicted.png", "Actual vs Predicted — Unseen Test Set")
    c1p, c2p = st.columns(2, gap="small")
    with c1p: plot_img("07_residuals.png",     "Residuals — Test Set")
    with c2p: plot_img("09_val_vs_test.png",   "Validation vs Test Comparison")
    c3p, c4p = st.columns(2, gap="small")
    with c3p: plot_img("08_scatter.png",       "Actual vs Predicted Scatter (₹)")
    with c4p: plot_img("10_attention_weights.png", "Attention Weights — Recency Decay (Fold 2)")
    plot_img("01_eda_overview.png",             "EDA Overview — Price · Volume · Distribution · Correlation")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif TAB == "arch":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)

    arch = MODEL_CONFIG["architecture"]
    tr   = MODEL_CONFIG["training"]
    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown(f"<div class='sec-title amber'>Model Architecture</div>",
                    unsafe_allow_html=True)
        LAYERS = [
            ("INPUT",     f"(60 × 13) — OHLCV + 8 engineered features", ICE_BLUE),
            ("GRU 1",     f"{arch['gru1']} units · return_sequences=True · rec_drop=0.05", SOFT_GRN),
            ("LAYERNORM", "Normalise activations across sequence", SILVER2),
            ("GRU 2",     f"{arch['gru2']} units · return_sequences=True · rec_drop=0.05", SOFT_GRN),
            ("LAYERNORM", "Normalise activations across sequence", SILVER2),
            ("ATTENTION", f"Bahdanau additive · {arch['attn_units']} units → context vector", AMBER),
            ("DENSE 1",   f"{arch['dense1']} units · ReLU · Dropout({arch['dropout']})", ICE_BLUE),
            ("DENSE 2",   f"{arch['dense2']} units · ReLU · Dropout({arch['dropout']*0.5:.3f})", ICE_BLUE),
            ("OUTPUT",    "1 unit → Log-Return → inverse_transform → ₹", LIME_GRN),
        ]
        for i, (name, desc, color) in enumerate(LAYERS):
            st.markdown(f"""
            <div class='arch-row'>
              <div class='arch-box' style='color:{color};border-color:{color}44;
                   background:{color}0D;'>{name}</div>
              <div class='arch-desc'>{desc}</div>
            </div>
            {"<div class='arch-arrow'>↓</div>" if i < len(LAYERS)-1 else ""}
            """, unsafe_allow_html=True)

        # Attention insight callout
        st.markdown(f"""
        <div style='background:{AMBER}0A;border:1px solid {AMBER}33;border-radius:6px;
             padding:.8rem 1rem;margin-top:1rem;font-family:Share Tech Mono,monospace;
             font-size:.68rem;color:{AMBER};line-height:1.8;'>
          ★ FOLD 2 ATTENTION INSIGHT<br>
          <span style='color:{SILVER2}'>
          Unlike Fold 1 (near-uniform weights ≈0.0163 across all 60 days),<br>
          Fold 2 shows a clear exponential recency decay — days 1–5 dominate<br>
          (weight ≈0.022), falling to near-zero at day 60 (weight ≈0.001).<br>
          This suggests the balanced-fit model learned a meaningful temporal<br>
          focus: recent momentum drives prediction far more than old history.
          </span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='sec-title' style='margin-top:1.5rem'>Input Features</div>",
                    unsafe_allow_html=True)
        pills = "".join([f"<span class='pill'>{f}</span>"
                         for f in MODEL_CONFIG["feature_cols"]])
        st.markdown(f"<div style='margin-bottom:.5rem'>{pills}</div>", unsafe_allow_html=True)
        st.markdown(f"""<div style='font-family:Share Tech Mono,monospace;font-size:.65rem;
            color:{SILVER2};line-height:1.9;margin-top:.5rem;'>
          TARGET&nbsp;&nbsp;&nbsp;: <span style='color:{AMBER}'>Log_Return</span><br>
          FEAT SC  : RobustScaler (fit on train fold only)<br>
          TGT SC   : StandardScaler (fit on train fold only)<br>
          BEST FOLD: <span style='color:{LIME_GRN}'>#2 ★ Balanced Fit</span>
          — selected by min|train−val loss|<br>
          TRAIN END: {MODEL_CONFIG['train_end_date']}
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='sec-title amber'>Hyperparameters</div>",
                    unsafe_allow_html=True)
        HP = [
            ("GRU Units",            f"{arch['gru1']} / {arch['gru2']}"),
            ("Attention Units",      str(arch["attn_units"])),
            ("Dense Units",          f"{arch['dense1']} / {arch['dense2']}"),
            ("Dropout",              str(arch["dropout"])),
            ("Recurrent Dropout",    str(arch["recurrent_dropout"])),
            ("L2 Regularisation",    str(arch["l2"])),
            ("Kernel Init",          "glorot_uniform"),
            ("Loss Function",        tr["loss"].upper()),
            ("Optimiser",            "Adam · clipnorm=1.0"),
            ("Init LR",              str(tr["init_lr"])),
            ("Peak LR",              str(tr["peak_lr"])),
            ("Warmup Epochs",        str(tr["warmup_epochs"])),
            ("Max Epochs",           str(tr["max_epochs"])),
            ("Batch Size",           str(tr["batch_size"])),
            ("Early Stop Patience",  str(tr["early_stopping_patience"])),
            ("Sequence Length",      f"{MODEL_CONFIG['sequence_len']} days"),
            ("N Features",           str(MODEL_CONFIG["n_features"])),
            ("Train End Date",       MODEL_CONFIG["train_end_date"]),
        ]
        st.markdown(f"<div class='hp-table'>", unsafe_allow_html=True)
        for k, v in HP:
            st.markdown(f"""<div class='hp-row'>
              <span class='hp-k'>{k}</span>
              <span class='hp-v'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='sec-title' style='margin-top:1.5rem'>Best Fold Selection Logic</div>",
                    unsafe_allow_html=True)
        FOLD_SCORES = [
            ("Fold 1", "0.2594", "0.1447", "0.1147", "UNDERFITTING"),
            ("Fold 2", "0.2589", "0.3241", "0.0652", "Good Fit ✓ ★"),
            ("Fold 3", "0.2903", "0.1653", "0.1250", "UNDERFITTING"),
            ("Fold 4", "0.2989", "0.2323", "0.0666", "UNDERFITTING"),
        ]
        st.markdown(f"""<div style='background:{NAVY3};border:1px solid {DIM_BLUE};
            border-radius:6px;overflow:hidden;'>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1.5fr;
               padding:.38rem .85rem;background:{NAVY4};font-size:.58rem;
               color:{SILVER2};letter-spacing:.1em;text-transform:uppercase;
               border-bottom:1px solid {DIM_BLUE};'>
            <span>Fold</span><span>Min Train</span><span>Min Val</span>
            <span>|Gap|</span><span>Status</span>
          </div>""", unsafe_allow_html=True)
        for fold, mt, mv, gap, status in FOLD_SCORES:
            is_best = "★" in status
            clr = LIME_GRN if is_best else SILVER2
            bg  = f"background:{LIME_GRN}0A;" if is_best else ""
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1.5fr;
                 padding:.38rem .85rem;border-bottom:1px solid {FAINT};
                 font-size:.7rem;font-family:Share Tech Mono,monospace;{bg}'>
              <span style='color:{clr};font-weight:{"700" if is_best else "400"}'>{fold}</span>
              <span style='color:{SILVER2}'>{mt}</span>
              <span style='color:{SILVER2}'>{mv}</span>
              <span style='color:{"#22C55E" if is_best else TICKER_R}'>{gap}</span>
              <span style='color:{clr}'>{status}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""<div class='disc'>
          ⚠ NOT FINANCIAL ADVICE · Research prototype only.<br>
          Best fold selected by min|train_loss − val_loss|.<br>
          Past model performance does not guarantee future results.</div>""",
          unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
