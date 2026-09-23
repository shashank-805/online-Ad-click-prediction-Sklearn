import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Online Ad Click Predictor",
    page_icon="🖱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom AdTech Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #31104B 50%, #4C1D95 100%);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(167, 139, 250, 0.15);
        color: #C4B5FD;
        border: 1px solid rgba(167, 139, 250, 0.35);
        margin-bottom: 10px;
    }

    .click-card-yes {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .click-card-no {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .ctr-value {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 6px 0;
    }

    .ad-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #8B5CF6;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Online_Ad_Click_Prediction_Sklearn" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("online_ad_click_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">AdTech & Programmatic CTR Optimization AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">🖱️ Online Ad Click Predictor</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast user click-through propensity (CTR) based on browsing engagement, ad placement slot, device form factor, and historical click affinity.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Impression CTR Predictor", "📁 Batch Impressions Screening (CSV)", "📊 Feature Importance & Diagnostics"])

# --- TAB 1: Impression CTR Predictor ---
with tabs[0]:
    st.subheader("User Engagement & Ad Placement Parameters")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        load_high = st.button("🚀 High-Intent Clicker", width="stretch")
    with p_cols[1]:
        load_low = st.button("🧊 Low-Engagement User", width="stretch")
    with p_cols[2]:
        load_sample = st.button("📋 Sample Impression", width="stretch")

    if load_high:
        st.session_state["age"] = 38
        st.session_state["time"] = 145.0
        st.session_state["pages"] = 18
        st.session_state["income"] = 72000.0
        st.session_state["internet"] = 8.5
        st.session_state["gender"] = "Female"
        st.session_state["device"] = "Mobile"
        st.session_state["pos"] = "Top"
        st.session_state["prev"] = 8
    elif load_low:
        st.session_state["age"] = 22
        st.session_state["time"] = 12.0
        st.session_state["pages"] = 2
        st.session_state["income"] = 28000.0
        st.session_state["internet"] = 1.5
        st.session_state["gender"] = "Male"
        st.session_state["device"] = "Desktop"
        st.session_state["pos"] = "Bottom"
        st.session_state["prev"] = 0
    elif load_sample:
        st.session_state["age"] = 42
        st.session_state["time"] = 92.0
        st.session_state["pages"] = 10
        st.session_state["income"] = 55000.0
        st.session_state["internet"] = 5.5
        st.session_state["gender"] = "Male"
        st.session_state["device"] = "Mobile"
        st.session_state["pos"] = "Top"
        st.session_state["prev"] = 5

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### ⏱️ On-Site User Behavior")
        daily_time = st.slider(
            "Daily Time on Site (Minutes)", 1.0, 180.0,
            value=float(st.session_state.get("time", 92.0)), step=1.0,
            key="input_time", help="Active dwell time logged by user across sessions."
        )
        pages_visited = st.slider(
            "Pages Visited in Session", 1, 30,
            value=int(st.session_state.get("pages", 10)), step=1,
            key="input_pages"
        )
        internet_usage = st.slider(
            "Daily Internet Usage (Hours)", 0.5, 15.0,
            value=float(st.session_state.get("internet", 5.5)), step=0.5,
            key="input_internet"
        )
        prev_clicks = st.slider(
            "Historical Ad Clicks Recorded", 0, 10,
            value=int(st.session_state.get("prev", 5)), step=1,
            key="input_prev", help="Prior ads clicked by user across campaign network."
        )

    with c_right:
        st.markdown("#### 📱 Device, Demographics & Ad Slot")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            age = st.slider(
                "User Age", 18, 65,
                value=int(st.session_state.get("age", 42)), step=1,
                key="input_age"
            )
        with col_d2:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=["Male", "Female"].index(st.session_state.get("gender", "Male")),
                key="input_gender"
            )

        annual_income = st.slider(
            "Annual Income ($)", 15000.0, 120000.0,
            value=float(st.session_state.get("income", 55000.0)), step=2500.0,
            key="input_income"
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            device = st.selectbox(
                "User Device Type",
                ["Mobile", "Desktop", "Tablet"],
                index=["Mobile", "Desktop", "Tablet"].index(st.session_state.get("device", "Mobile")),
                key="input_device"
            )
        with col_p2:
            ad_position = st.selectbox(
                "Ad Slot Position",
                ["Top", "Middle", "Bottom"],
                index=["Top", "Middle", "Bottom"].index(st.session_state.get("pos", "Top")),
                key="input_pos"
            )

    st.markdown("---")
    eval_btn = st.button("🚀 Predict Ad Click Likelihood", type="primary", width="stretch")

    impression_df = pd.DataFrame([{
        "age": age,
        "daily_time_on_site": daily_time,
        "pages_visited": pages_visited,
        "annual_income": annual_income,
        "internet_usage_hours": internet_usage,
        "gender": gender,
        "device": device,
        "ad_position": ad_position,
        "previous_clicks": prev_clicks
    }])

    pred = model.predict(impression_df)[0]
    probs = model.predict_proba(impression_df)[0]
    classes = list(model.classes_)
    click_idx = classes.index(1) if 1 in classes else 1
    click_prob = probs[click_idx] * 100

    st.markdown("### 📋 Impression CTR Prediction")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        if pred == 1:
            st.markdown(f"""
            <div class="click-card-yes">
                <span style="font-size: 2.8rem;">🎯</span>
                <div style="color: #34D399; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    User Likely to CLICK
                </div>
                <div class="ctr-value" style="color: #10B981;">
                    {click_prob:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Predicted Click-Through Likelihood</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="click-card-no">
                <span style="font-size: 2.8rem;">❌</span>
                <div style="color: #F87171; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    User Likely to NOT Click
                </div>
                <div class="ctr-value" style="color: #EF4444;">
                    {click_prob:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Predicted Click-Through Likelihood</p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(float(click_prob / 100.0))

    with r2:
        st.markdown("#### 🔍 Engagement Drivers & Ad Slot Signals")
        signals = []
        if ad_position == "Top":
            signals.append(("Prime Above-the-Fold Slot", "Top placement maximizes visual attention and initial dwell gaze.", "info"))
        elif ad_position == "Bottom":
            signals.append(("Footer / Below-the-Fold Slot", "Bottom slot suffers from scroll fatigue and lower conversion.", "warn"))

        if prev_clicks >= 5:
            signals.append(("High Past Click Affinity", f"{prev_clicks} previous ad clicks indicates an active, responsive consumer.", "info"))
        elif prev_clicks == 0:
            signals.append(("Zero Click History", "User exhibits ad blindness; requires high-relevance creative hooks.", "warn"))

        if daily_time >= 90.0:
            signals.append(("Extended Session Dwell Time", f"{daily_time:.0f} mins daily browsing increases impression recall.", "info"))

        if signals:
            for title, desc, kind in signals:
                if kind == "info":
                    st.success(f"**{title}**: {desc}")
                else:
                    st.warning(f"**{title}**: {desc}")
        else:
            st.info("User parameters match average baseline engagement patterns.")

        st.markdown(f"""
        <div class="ad-box">
            <strong style="color: #C4B5FD;">Programmatic Bidding Recommendation:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'AGGRESSIVE BID: Increase real-time eCPM bid by 20-30% to secure this high-converting impression.' if pred == 1 else 'CONSERVATIVE BID: Lower bid ceiling to prevent wasted ad budget on low-probability impression.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Features"):
        st.dataframe(impression_df, width="stretch")

# --- TAB 2: Batch Impressions Screening ---
with tabs[1]:
    st.subheader("Batch Impression Log Simulation")
    st.write("Upload an ad server log CSV or screen against the 700-impression benchmark dataset.")

    csv_file = st.file_uploader("Upload Impressions CSV", type=["csv"], key="ad_csv")
    df_ads = None

    if csv_file is not None:
        df_ads = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_ads)} impression records from file.")
    else:
        sample_path = get_asset_path("data/online_ad_clicks.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline ad clicks dataset (`data/online_ad_clicks.csv`)", value=True):
                df_ads = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_ads)} records from baseline dataset.")

    if df_ads is not None:
        req_cols = ["age", "daily_time_on_site", "pages_visited", "annual_income", "internet_usage_hours", "gender", "device", "ad_position", "previous_clicks"]
        missing = [c for c in req_cols if c not in df_ads.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Score Entire Batch Impressions", type="primary"):
                with st.spinner("Calculating CTR probabilities..."):
                    preds = model.predict(df_ads[req_cols])
                    probs = model.predict_proba(df_ads[req_cols])
                    c_idx = list(model.classes_).index(1) if 1 in list(model.classes_) else 1
                    click_probs = probs[:, c_idx] * 100

                    res_df = df_ads.copy()
                    res_df["Predicted_Click"] = ["Clicked" if p == 1 else "Not Clicked" for p in preds]
                    res_df["Click_Probability_%"] = np.round(click_probs, 1)

                    clicks_c = sum(preds == 1)
                    no_clicks_c = sum(preds == 0)
                    overall_ctr = (clicks_c / len(res_df)) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Impressions", len(res_df))
                    m2.metric("Predicted Clicks", clicks_c, delta=f"{overall_ctr:.1f}% CTR")
                    m3.metric("Non-Click Impressions", no_clicks_c)
                    m4.metric("Avg Click Propensity", f"{np.mean(click_probs):.1f}%")

                    f1, f2 = st.columns(2)
                    with f1:
                        f_res = st.selectbox("Filter Result:", ["All", "Clicked", "Not Clicked"])
                    with f2:
                        f_pos = st.selectbox("Filter Position:", ["All"] + list(df_ads["ad_position"].unique()))

                    view = res_df
                    if f_res != "All":
                        view = view[view["Predicted_Click"] == f_res]
                    if f_pos != "All":
                        view = view[view["ad_position"] == f_pos]

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Impressions as CSV",
                        data=csv_export,
                        file_name="ad_clicks_scored_predictions.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Feature Importance & Diagnostics ---
with tabs[2]:
    st.subheader("Model Architecture & Feature Importance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 CTR Prediction Model Architecture
        - **Algorithm**: `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`
        - **Pre-processing**:
            - `OneHotEncoder` on `gender`, `device`, `ad_position`
            - Passthrough on engagement & behavioral numerics
        - **Evaluation Benchmark**:
            - **Accuracy**: **90.71%** on Stratified Test Split
            - **Precision (Clicked)**: ~0.94
            - **Recall (Clicked)**: ~0.96
        """)

    with c2:
        img_path = get_asset_path("feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Top Features Influencing Ad Clicks", width="stretch")
        else:
            st.info("Feature importance chart not found.")

st.caption("AdTech & Digital Marketing Intelligence Suite • Scikit-learn & Streamlit")
