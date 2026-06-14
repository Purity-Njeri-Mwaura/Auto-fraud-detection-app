import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme & CSS ───────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg       = "#0d1117"
        surface  = "#161b22"
        surface2 = "#21262d"
        border   = "#30363d"
        text     = "#e6edf3"
        subtext  = "#8b949e"
        accent   = "#f78166"
        accent2  = "#58a6ff"
        success  = "#3fb950"
        warning  = "#d29922"
        danger   = "#f85149"
        card_bg  = "#161b22"
    else:
        bg       = "#f0f4f8"
        surface  = "#ffffff"
        surface2 = "#f6f8fa"
        border   = "#d0d7de"
        text     = "#1c2128"
        subtext  = "#57606a"
        accent   = "#cf222e"
        accent2  = "#0969da"
        success  = "#1a7f37"
        warning  = "#9a6700"
        danger   = "#cf222e"
        card_bg  = "#ffffff"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg: {bg}; --surface: {surface}; --surface2: {surface2};
        --border: {border}; --text: {text}; --subtext: {subtext};
        --accent: {accent}; --accent2: {accent2}; --success: {success};
        --warning: {warning}; --danger: {danger}; --card-bg: {card_bg};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }}
    [data-testid="stSidebar"] * {{ color: var(--text) !important; }}
    .stButton > button {{
        background: var(--accent2) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        transition: opacity .15s !important;
    }}
    .stButton > button:hover {{ opacity: .85 !important; }}
    .stSelectbox > div, .stNumberInput > div, .stSlider > div {{
        background: var(--surface2) !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }}
    .metric-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }}
    .metric-card .val {{ font-size: 2.2rem; font-weight: 800; }}
    .metric-card .lbl {{ font-size: 0.78rem; color: var(--subtext); font-weight: 500; text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; }}
    .verdict-box {{
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin: 1rem 0;
        border: 2px solid;
    }}
    .verdict-fraud   {{ background: {'#2d1b1b' if dark else '#fff0f0'}; border-color: var(--danger); }}
    .verdict-legit   {{ background: {'#1b2d1b' if dark else '#f0fff4'}; border-color: var(--success); }}
    .verdict-title   {{ font-size: 1.7rem; font-weight: 800; margin-bottom: .3rem; }}
    .verdict-sub     {{ color: var(--subtext); font-size: .9rem; }}
    .section-header  {{ font-size: 1.1rem; font-weight: 700; color: var(--text); border-bottom: 2px solid var(--accent2); padding-bottom: .4rem; margin: 1.4rem 0 1rem; display: inline-block; }}
    .tag {{
        display: inline-block;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: .2rem .7rem;
        font-size: .75rem;
        font-weight: 600;
        color: var(--subtext);
        margin: .15rem .1rem;
    }}
    .sidebar-logo {{ font-size: 1.5rem; font-weight: 800; letter-spacing: -.5px; }}
    .sidebar-tagline {{ font-size: .78rem; color: var(--subtext); margin-top: -4px; }}
    div[data-testid="stTab"] button {{ font-weight: 600 !important; }}
    .stTabs [data-baseweb="tab-list"] {{ background: var(--surface2) !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px !important; color: var(--subtext) !important; }}
    .stTabs [aria-selected="true"] {{ background: var(--surface) !important; color: var(--text) !important; }}
    hr {{ border-color: var(--border) !important; }}
    .stAlert {{ border-radius: 10px !important; }}
    [data-testid="stExpander"] {{ background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }}
    .risk-bar-wrap {{ background: var(--surface2); border-radius: 99px; height: 12px; overflow: hidden; margin: .5rem 0 1.2rem; }}
    .risk-bar-fill {{ height: 12px; border-radius: 99px; transition: width .6s ease; }}
    </style>
    """, unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "ratings" not in st.session_state:
    st.session_state.ratings = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "input_df" not in st.session_state:
    st.session_state.input_df = None

dark = st.session_state.dark_mode
inject_css(dark)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("fraud_lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_cols = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛡️ FraudShield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Auto Insurance Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio(
        "Navigate",
        ["🏠  Home", "📋  Input Claim", "🔬  Analysis", "⚙️  Settings", "⭐  Rate App"],
        label_visibility="collapsed",
    )
    page = nav.split("  ")[1].strip()

    st.markdown("---")
    st.markdown('<p style="font-size:.75rem;color:var(--subtext)">Model Performance</p>', unsafe_allow_html=True)
    cols_m = st.columns(2)
    with cols_m[0]:
        st.markdown('<div class="metric-card"><div class="val" style="color:var(--accent2);font-size:1.3rem">0.978</div><div class="lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
    with cols_m[1]:
        st.markdown('<div class="metric-card"><div class="val" style="color:var(--success);font-size:1.3rem">94.6%</div><div class="lbl">Recall</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:.5rem" class="metric-card"><div class="val" style="color:var(--warning);font-size:1.3rem">0.770</div><div class="lbl">F1-Score</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Quick theme toggle in sidebar too
    if st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="sidebar_dark"):
        st.session_state.dark_mode = True
    else:
        st.session_state.dark_mode = False
    if dark != st.session_state.dark_mode:
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div style="padding:2.5rem 0 1rem">
        <div style="font-size:3rem;font-weight:900;letter-spacing:-1.5px;line-height:1.1">
            🛡️ FraudShield
        </div>
        <div style="font-size:1.15rem;color:var(--subtext);margin-top:.6rem;max-width:560px">
            XAI-driven fraud detection for auto insurance claims — powered by 
            LightGBM, SHAP, and LIME.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("0.978", "ROC-AUC", "var(--accent2)"),
        ("94.6%", "Fraud Recall", "var(--success)"),
        ("0.770", "F1-Score", "var(--warning)"),
        ("10,000", "Claims Trained On", "var(--subtext)"),
    ]
    for col, (val, lbl, color) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="val" style="color:{color}">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<span class="section-header">What is FraudShield?</span>', unsafe_allow_html=True)
        st.markdown("""
        FraudShield is an **explainable AI system** built to detect fraudulent auto insurance 
        claims. Unlike black-box detectors, every prediction comes with transparent reasoning 
        via **SHAP** (global) and **LIME** (local) explanations — so investigators always 
        know *why* a claim was flagged.

        Built as part of a Bachelor of Science in Data Science research project at JKUAT, 
        the system was evaluated against four ensemble classifiers. **LightGBM emerged as the 
        top performer**, achieving a fraud recall of 94.6% and ROC-AUC of 0.978.
        """)

        st.markdown('<span class="section-header">How to use it</span>', unsafe_allow_html=True)
        st.markdown("""
        1. **Go to Input Claim** — fill in the policyholder and incident details  
        2. **Run the prediction** — get an instant fraud probability score  
        3. **Explore Analysis** — understand *why* via SHAP & LIME explanations  
        4. **Share findings** — download a summary report
        """)

    with col_r:
        st.markdown('<span class="section-header">Key Fraud Signals</span>', unsafe_allow_html=True)
        signals = [
            ("🔴", "Incident Severity", "Highest predictive power"),
            ("🟠", "Policy Year", "Older policies — higher risk"),
            ("🟠", "Insured Hobbies", "Skydiving, polo correlate with fraud"),
            ("🟡", "Bodily Injuries", "Inflated injury counts flagged"),
            ("🟡", "Witnesses", "Zero witnesses increases risk"),
            ("🟢", "Authorities Contacted", "Police filing reduces fraud odds"),
        ]
        for icon, feat, desc in signals:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:.7rem;margin-bottom:.75rem;
                        background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.7rem 1rem">
                <div style="font-size:1.1rem;margin-top:1px">{icon}</div>
                <div>
                    <div style="font-weight:600;font-size:.9rem">{feat}</div>
                    <div style="font-size:.78rem;color:var(--subtext)">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:var(--subtext);font-size:.8rem;padding:.5rem 0">
        Built with ❤️ · LightGBM + SHAP + LIME · Purity Njeri Mwaura, JKUAT 2026
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INPUT CLAIM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Input Claim":
    st.markdown("## 📋 Input Claim Details")
    st.markdown('<p style="color:var(--subtext);margin-top:-.5rem">Fill in the claim information below. All fields are required for an accurate prediction.</p>', unsafe_allow_html=True)

    with st.form("claim_form"):
        # ── Section 1: Policyholder ──────────────────────────────────────────
        st.markdown('<span class="section-header">👤 Policyholder Details</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            months_as_customer = st.number_input("Months as Customer", 0, 600, 120)
            age = st.number_input("Age", 16, 90, 35)
            policy_number = st.number_input("Policy Number", 100000, 999999, 500000)
        with c2:
            insured_sex = st.selectbox("Sex", ["FEMALE", "MALE"])
            insured_education_level = st.selectbox("Education Level",
                ["Associate", "College", "High School", "JD", "MD", "Masters", "PhD"])
            insured_occupation = st.selectbox("Occupation", [
                "adm-clerical", "armed-forces", "craft-repair", "exec-managerial",
                "farming-fishing", "handlers-cleaners", "machine-op-inspct", "other-service",
                "priv-house-serv", "prof-specialty", "protective-serv", "sales",
                "tech-support", "transport-moving"])
        with c3:
            insured_hobbies = st.selectbox("Hobbies", [
                "basketball", "board-games", "bungie-jumping", "camping", "chess",
                "cross-fit", "dancing", "exercise", "golf", "hiking", "kayaking",
                "movies", "paintball", "polo", "reading", "skydiving", "sleeping",
                "video-games", "yachting"])
            insured_relationship = st.selectbox("Relationship", [
                "husband", "not-in-family", "other-relative", "own-child", "unmarried", "wife"])
            insured_zip = st.number_input("ZIP Code", 10000, 99999, 46082)

        # ── Section 2: Policy ────────────────────────────────────────────────
        st.markdown('<span class="section-header">📄 Policy Details</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            policy_state = st.selectbox("Policy State", ["IL", "IN", "OH"])
            policy_csl = st.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
            policy_deductable = st.selectbox("Deductible ($)", [500, 1000, 2000])
        with c2:
            policy_annual_premium = st.number_input("Annual Premium ($)", 500.0, 3000.0, 1200.0, step=50.0)
            umbrella_limit = st.number_input("Umbrella Limit ($M)", -1000000, 10000000, 0, step=1000000)
            capital_gains = st.number_input("Capital Gains ($)", 0, 100000, 0, step=1000)
        with c3:
            capital_loss = st.number_input("Capital Loss ($)", -100000, 0, 0, step=1000)
            policy_bind_year = st.number_input("Policy Bind Year", 1990, 2024, 2010)
            policy_bind_month = st.number_input("Policy Bind Month", 1, 12, 6)

        # ── Section 3: Incident ──────────────────────────────────────────────
        st.markdown('<span class="section-header">🚗 Incident Details</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            incident_type = st.selectbox("Incident Type", [
                "Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"])
            incident_severity = st.selectbox("Incident Severity", [
                "Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"])
            collision_type = st.selectbox("Collision Type", [
                "Front Collision", "Rear Collision", "Side Collision", "?"])
        with c2:
            authorities_contacted = st.selectbox("Authorities Contacted", [
                "Ambulance", "Fire", "None", "Other", "Police"])
            police_report_available = st.selectbox("Police Report Available", ["YES", "NO"])
            property_damage = st.selectbox("Property Damage", ["YES", "NO"])
        with c3:
            incident_hour_of_the_day = st.slider("Incident Hour (0–23)", 0, 23, 14)
            number_of_vehicles_involved = st.number_input("Vehicles Involved", 1, 4, 1)
            bodily_injuries = st.number_input("Bodily Injuries", 0, 2, 0)
            witnesses = st.number_input("Witnesses", 0, 3, 1)

        c1, c2 = st.columns(2)
        with c1:
            incident_state = st.selectbox("Incident State", ["IL", "NY", "OH", "PA", "SC", "VA", "WV"])
            incident_city = st.selectbox("Incident City", [
                "Arlington", "Columbus", "Hillsdale", "Northbend", "Northbrook", "Riverwood", "Springfield"])
        with c2:
            incident_year = st.number_input("Incident Year", 2000, 2026, 2015)
            incident_month = st.number_input("Incident Month", 1, 12, 3)

        # ── Section 4: Vehicle ───────────────────────────────────────────────
        st.markdown('<span class="section-header">🚙 Vehicle Details</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            auto_make = st.selectbox("Auto Make", [
                "Accura", "Audi", "BMW", "Chevrolet", "Dodge", "Ford",
                "Honda", "Jeep", "Mercedes", "Nissan", "Saab", "Suburu", "Toyota", "Volkswagen"])
        with c2:
            auto_model = st.selectbox("Auto Model", [
                "95", "3 Series", "92x", "A3", "A5", "Accord", "C300", "CRV",
                "Camry", "Civic", "Corolla", "E400", "Escape", "F150", "Forrestor",
                "Fusion", "Grand Cherokee", "Highlander", "Impreza", "Jetta",
                "Legacy", "M5", "MDX", "ML350", "Malibu", "Maxima", "Neon",
                "Passat", "Pathfinder", "RAM", "RSX", "Silverado", "TL",
                "Tahoe", "Ultima", "Wrangler", "X5", "X6"])
        with c3:
            auto_year = st.number_input("Auto Year", 1995, 2015, 2010)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Analyse Claim", use_container_width=True)

    if submitted:
        # Build raw input dict
        raw = {
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_number": policy_number,
            "policy_deductable": policy_deductable,
            "policy_annual_premium": policy_annual_premium,
            "umbrella_limit": umbrella_limit,
            "insured_zip": insured_zip,
            "capital-gains": capital_gains,
            "capital-loss": capital_loss,
            "incident_hour_of_the_day": incident_hour_of_the_day,
            "number_of_vehicles_involved": number_of_vehicles_involved,
            "bodily_injuries": bodily_injuries,
            "witnesses": witnesses,
            "auto_year": auto_year,
            "policy_year": policy_bind_year,
            "policy_month": policy_bind_month,
            "incident_year": incident_year,
            "incident_month": incident_month,
        }

        # One-hot encodings
        ohe_map = {
            "policy_state_IN": policy_state == "IN",
            "policy_state_OH": policy_state == "OH",
            "policy_csl_250/500": policy_csl == "250/500",
            "policy_csl_500/1000": policy_csl == "500/1000",
            "insured_sex_MALE": insured_sex == "MALE",
            "insured_education_level_College": insured_education_level == "College",
            "insured_education_level_High School": insured_education_level == "High School",
            "insured_education_level_JD": insured_education_level == "JD",
            "insured_education_level_MD": insured_education_level == "MD",
            "insured_education_level_Masters": insured_education_level == "Masters",
            "insured_education_level_PhD": insured_education_level == "PhD",
            "insured_occupation_armed-forces": insured_occupation == "armed-forces",
            "insured_occupation_craft-repair": insured_occupation == "craft-repair",
            "insured_occupation_exec-managerial": insured_occupation == "exec-managerial",
            "insured_occupation_farming-fishing": insured_occupation == "farming-fishing",
            "insured_occupation_handlers-cleaners": insured_occupation == "handlers-cleaners",
            "insured_occupation_machine-op-inspct": insured_occupation == "machine-op-inspct",
            "insured_occupation_other-service": insured_occupation == "other-service",
            "insured_occupation_priv-house-serv": insured_occupation == "priv-house-serv",
            "insured_occupation_prof-specialty": insured_occupation == "prof-specialty",
            "insured_occupation_protective-serv": insured_occupation == "protective-serv",
            "insured_occupation_sales": insured_occupation == "sales",
            "insured_occupation_tech-support": insured_occupation == "tech-support",
            "insured_occupation_transport-moving": insured_occupation == "transport-moving",
            "insured_hobbies_basketball": insured_hobbies == "basketball",
            "insured_hobbies_board-games": insured_hobbies == "board-games",
            "insured_hobbies_bungie-jumping": insured_hobbies == "bungie-jumping",
            "insured_hobbies_camping": insured_hobbies == "camping",
            "insured_hobbies_chess": insured_hobbies == "chess",
            "insured_hobbies_cross-fit": insured_hobbies == "cross-fit",
            "insured_hobbies_dancing": insured_hobbies == "dancing",
            "insured_hobbies_exercise": insured_hobbies == "exercise",
            "insured_hobbies_golf": insured_hobbies == "golf",
            "insured_hobbies_hiking": insured_hobbies == "hiking",
            "insured_hobbies_kayaking": insured_hobbies == "kayaking",
            "insured_hobbies_movies": insured_hobbies == "movies",
            "insured_hobbies_paintball": insured_hobbies == "paintball",
            "insured_hobbies_polo": insured_hobbies == "polo",
            "insured_hobbies_reading": insured_hobbies == "reading",
            "insured_hobbies_skydiving": insured_hobbies == "skydiving",
            "insured_hobbies_sleeping": insured_hobbies == "sleeping",
            "insured_hobbies_video-games": insured_hobbies == "video-games",
            "insured_hobbies_yachting": insured_hobbies == "yachting",
            "insured_relationship_not-in-family": insured_relationship == "not-in-family",
            "insured_relationship_other-relative": insured_relationship == "other-relative",
            "insured_relationship_own-child": insured_relationship == "own-child",
            "insured_relationship_unmarried": insured_relationship == "unmarried",
            "insured_relationship_wife": insured_relationship == "wife",
            "incident_type_Parked Car": incident_type == "Parked Car",
            "incident_type_Single Vehicle Collision": incident_type == "Single Vehicle Collision",
            "incident_type_Vehicle Theft": incident_type == "Vehicle Theft",
            "collision_type_Front Collision": collision_type == "Front Collision",
            "collision_type_Rear Collision": collision_type == "Rear Collision",
            "collision_type_Side Collision": collision_type == "Side Collision",
            "incident_severity_Minor Damage": incident_severity == "Minor Damage",
            "incident_severity_Total Loss": incident_severity == "Total Loss",
            "incident_severity_Trivial Damage": incident_severity == "Trivial Damage",
            "authorities_contacted_Fire": authorities_contacted == "Fire",
            "authorities_contacted_Other": authorities_contacted == "Other",
            "authorities_contacted_Police": authorities_contacted == "Police",
            "incident_state_NY": incident_state == "NY",
            "incident_state_OH": incident_state == "OH",
            "incident_state_PA": incident_state == "PA",
            "incident_state_SC": incident_state == "SC",
            "incident_state_VA": incident_state == "VA",
            "incident_state_WV": incident_state == "WV",
            "incident_city_Columbus": incident_city == "Columbus",
            "incident_city_Hillsdale": incident_city == "Hillsdale",
            "incident_city_Northbend": incident_city == "Northbend",
            "incident_city_Northbrook": incident_city == "Northbrook",
            "incident_city_Riverwood": incident_city == "Riverwood",
            "incident_city_Springfield": incident_city == "Springfield",
            "property_damage_NO": property_damage == "NO",
            "property_damage_YES": property_damage == "YES",
            "police_report_available_NO": police_report_available == "NO",
            "police_report_available_YES": police_report_available == "YES",
            "auto_make_Audi": auto_make == "Audi",
            "auto_make_BMW": auto_make == "BMW",
            "auto_make_Chevrolet": auto_make == "Chevrolet",
            "auto_make_Dodge": auto_make == "Dodge",
            "auto_make_Ford": auto_make == "Ford",
            "auto_make_Honda": auto_make == "Honda",
            "auto_make_Jeep": auto_make == "Jeep",
            "auto_make_Mercedes": auto_make == "Mercedes",
            "auto_make_Nissan": auto_make == "Nissan",
            "auto_make_Saab": auto_make == "Saab",
            "auto_make_Suburu": auto_make == "Suburu",
            "auto_make_Toyota": auto_make == "Toyota",
            "auto_make_Volkswagen": auto_make == "Volkswagen",
            "auto_model_95": auto_model == "95",
            "auto_model_3 Series": auto_model == "3 Series",
            "auto_model_92x": auto_model == "92x",
            "auto_model_A3": auto_model == "A3",
            "auto_model_A5": auto_model == "A5",
            "auto_model_Accord": auto_model == "Accord",
            "auto_model_C300": auto_model == "C300",
            "auto_model_CRV": auto_model == "CRV",
            "auto_model_Camry": auto_model == "Camry",
            "auto_model_Civic": auto_model == "Civic",
            "auto_model_Corolla": auto_model == "Corolla",
            "auto_model_E400": auto_model == "E400",
            "auto_model_Escape": auto_model == "Escape",
            "auto_model_F150": auto_model == "F150",
            "auto_model_Forrestor": auto_model == "Forrestor",
            "auto_model_Fusion": auto_model == "Fusion",
            "auto_model_Grand Cherokee": auto_model == "Grand Cherokee",
            "auto_model_Highlander": auto_model == "Highlander",
            "auto_model_Impreza": auto_model == "Impreza",
            "auto_model_Jetta": auto_model == "Jetta",
            "auto_model_Legacy": auto_model == "Legacy",
            "auto_model_M5": auto_model == "M5",
            "auto_model_MDX": auto_model == "MDX",
            "auto_model_ML350": auto_model == "ML350",
            "auto_model_Malibu": auto_model == "Malibu",
            "auto_model_Maxima": auto_model == "Maxima",
            "auto_model_Neon": auto_model == "Neon",
            "auto_model_Passat": auto_model == "Passat",
            "auto_model_Pathfinder": auto_model == "Pathfinder",
            "auto_model_RAM": auto_model == "RAM",
            "auto_model_RSX": auto_model == "RSX",
            "auto_model_Silverado": auto_model == "Silverado",
            "auto_model_TL": auto_model == "TL",
            "auto_model_Tahoe": auto_model == "Tahoe",
            "auto_model_Ultima": auto_model == "Ultima",
            "auto_model_Wrangler": auto_model == "Wrangler",
            "auto_model_X5": auto_model == "X5",
            "auto_model_X6": auto_model == "X6",
        }

        for k, v in ohe_map.items():
            raw[k] = int(v)

        # Build aligned df
        input_df = pd.DataFrame([{col: raw.get(col, 0) for col in feature_cols}])
        input_df = input_df.astype(float)

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)

        st.session_state.prediction_result = {"prob": prob, "pred": pred, "input_df": input_df}
        st.session_state.input_df = input_df

        # ── Verdict box ──────────────────────────────────────────────────────
        if pred == 1:
            risk_color = "#f85149" if prob > 0.75 else "#d29922"
            st.markdown(f"""
            <div class="verdict-box verdict-fraud">
                <div class="verdict-title" style="color:var(--danger)">⚠️ Fraud Suspected</div>
                <div class="verdict-sub">Fraud probability: <strong style="color:var(--danger);font-size:1.1rem">{prob*100:.1f}%</strong></div>
                <div style="margin-top:.8rem;font-size:.88rem;color:var(--subtext)">
                    This claim has been flagged as likely fraudulent. Proceed to the <strong>Analysis</strong> tab 
                    to understand the key factors driving this decision.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box verdict-legit">
                <div class="verdict-title" style="color:var(--success)">✅ Claim Appears Legitimate</div>
                <div class="verdict-sub">Fraud probability: <strong style="color:var(--success);font-size:1.1rem">{prob*100:.1f}%</strong></div>
                <div style="margin-top:.8rem;font-size:.88rem;color:var(--subtext)">
                    This claim is classified as legitimate. Check the <strong>Analysis</strong> tab for full explanation.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk bar
        bar_color = "#f85149" if prob > 0.75 else ("#d29922" if prob > 0.5 else "#3fb950")
        st.markdown(f"""
        <div style="margin:1rem 0 .3rem;font-size:.82rem;font-weight:600;color:var(--subtext)">
            FRAUD RISK LEVEL
        </div>
        <div class="risk-bar-wrap">
            <div class="risk-bar-fill" style="width:{prob*100:.1f}%;background:{bar_color}"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.73rem;color:var(--subtext)">
            <span>0% — No Risk</span><span>{prob*100:.1f}%</span><span>100% — Certain Fraud</span>
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 Navigate to **Analysis** in the sidebar to see SHAP & LIME explanations.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analysis":
    st.markdown("## 🔬 Prediction Analysis")

    if st.session_state.prediction_result is None:
        st.warning("⚠️ No prediction yet. Go to **Input Claim** first and run an analysis.")
        st.stop()

    result   = st.session_state.prediction_result
    prob     = result["prob"]
    pred     = result["pred"]
    input_df = result["input_df"]

    # Summary bar
    bar_color = "#f85149" if prob > 0.75 else ("#d29922" if prob > 0.5 else "#3fb950")
    verdict_text = "Fraud Suspected" if pred == 1 else "Legitimate"
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;
                padding:1.2rem 1.6rem;display:flex;align-items:center;gap:2rem;margin-bottom:1.5rem;flex-wrap:wrap">
        <div>
            <div style="font-size:.75rem;color:var(--subtext);font-weight:600;text-transform:uppercase">Verdict</div>
            <div style="font-size:1.3rem;font-weight:800;color:{'var(--danger)' if pred==1 else 'var(--success)'}">
                {'⚠️ ' if pred==1 else '✅ '}{verdict_text}
            </div>
        </div>
        <div>
            <div style="font-size:.75rem;color:var(--subtext);font-weight:600;text-transform:uppercase">Fraud Probability</div>
            <div style="font-size:1.3rem;font-weight:800;color:{bar_color}">{prob*100:.1f}%</div>
        </div>
        <div style="flex:1;min-width:200px">
            <div class="risk-bar-wrap" style="margin:0">
                <div class="risk-bar-fill" style="width:{prob*100:.1f}%;background:{bar_color}"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 SHAP Explanation", "🔍 LIME Explanation", "📈 Feature Impact", "📋 Claim Summary"])

    # ── TAB 1: SHAP ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("**SHAP** (SHapley Additive exPlanations) shows how each feature contributed to pushing the prediction toward or away from fraud.")
        
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            feat_names = feature_cols
            shap_df = pd.DataFrame({"feature": feat_names, "shap": sv})
            shap_df["abs"] = shap_df["shap"].abs()
            shap_df = shap_df.sort_values("abs", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(9, 5.5))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            colors = ["#f85149" if v > 0 else "#58a6ff" for v in shap_df["shap"]]
            bars = ax.barh(range(len(shap_df)), shap_df["shap"].values[::-1] if False else shap_df["shap"].values,
                           color=colors, edgecolor="none", height=0.7)
            ax.set_yticks(range(len(shap_df)))
            ax.set_yticklabels(shap_df["feature"].values, fontsize=9,
                               color="#e6edf3" if dark else "#1c2128")
            ax.set_xlabel("SHAP Value (impact on fraud probability)", fontsize=9,
                          color="#8b949e")
            ax.axvline(0, color="#30363d", linewidth=1)
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.set_title("Top 15 Features by SHAP Impact", fontsize=11, fontweight="bold",
                         color="#e6edf3" if dark else "#1c2128", pad=12)

            red_patch  = mpatches.Patch(color="#f85149", label="Increases fraud risk")
            blue_patch = mpatches.Patch(color="#58a6ff", label="Decreases fraud risk")
            ax.legend(handles=[red_patch, blue_patch], frameon=False,
                      labelcolor="#8b949e", fontsize=8, loc="lower right")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Top 3 risk drivers
            st.markdown('<span class="section-header">Top Risk Drivers</span>', unsafe_allow_html=True)
            top3 = shap_df[shap_df["shap"] > 0].head(3)
            top3_mit = shap_df[shap_df["shap"] < 0].head(3)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔴 Increasing Fraud Risk**")
                for _, row in top3.iterrows():
                    val = input_df[row["feature"]].values[0]
                    st.markdown(f"""
                    <div style="background:var(--surface);border-left:3px solid var(--danger);
                                border-radius:0 8px 8px 0;padding:.6rem 1rem;margin:.4rem 0">
                        <div style="font-weight:600;font-size:.88rem">{row['feature']}</div>
                        <div style="font-size:.78rem;color:var(--subtext)">Value: {val:.2f} · SHAP: +{row['shap']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            with c2:
                st.markdown("**🔵 Reducing Fraud Risk**")
                for _, row in top3_mit.iterrows():
                    val = input_df[row["feature"]].values[0]
                    st.markdown(f"""
                    <div style="background:var(--surface);border-left:3px solid var(--accent2);
                                border-radius:0 8px 8px 0;padding:.6rem 1rem;margin:.4rem 0">
                        <div style="font-weight:600;font-size:.88rem">{row['feature']}</div>
                        <div style="font-size:.78rem;color:var(--subtext)">Value: {val:.2f} · SHAP: {row['shap']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"SHAP computation error: {e}")
            st.info("Tip: Ensure `shap` is installed in your environment.")

    # ── TAB 2: LIME ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("**LIME** (Local Interpretable Model-agnostic Explanations) perturbs the input and fits a simple model locally to explain *this specific claim*.")

        try:
            import lime
            import lime.lime_tabular

            # Build a simple background dataset (random noise around input)
            np.random.seed(42)
            bg_data = np.random.randn(200, len(feature_cols))

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=bg_data,
                feature_names=feature_cols,
                class_names=["Legitimate", "Fraud"],
                mode="classification",
                discretize_continuous=True,
            )

            explanation = lime_explainer.explain_instance(
                input_df.values[0],
                model.predict_proba,
                num_features=15,
                top_labels=1,
            )

            lime_list = explanation.as_list(label=1)
            lime_df   = pd.DataFrame(lime_list, columns=["Feature Rule", "Weight"])
            lime_df   = lime_df.sort_values("Weight", key=abs, ascending=False)

            fig, ax = plt.subplots(figsize=(9, 5.5))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")
            colors = ["#f85149" if w > 0 else "#58a6ff" for w in lime_df["Weight"]]
            ax.barh(range(len(lime_df)), lime_df["Weight"].values, color=colors, edgecolor="none", height=0.7)
            ax.set_yticks(range(len(lime_df)))
            ax.set_yticklabels(lime_df["Feature Rule"].values, fontsize=8,
                               color="#e6edf3" if dark else "#1c2128")
            ax.set_xlabel("LIME Weight (local impact)", fontsize=9, color="#8b949e")
            ax.axvline(0, color="#30363d", linewidth=1)
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.set_title("LIME — Local Feature Explanation", fontsize=11, fontweight="bold",
                         color="#e6edf3" if dark else "#1c2128", pad=12)
            red_patch  = mpatches.Patch(color="#f85149", label="Pushes toward Fraud")
            blue_patch = mpatches.Patch(color="#58a6ff", label="Pushes toward Legitimate")
            ax.legend(handles=[red_patch, blue_patch], frameon=False, labelcolor="#8b949e", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 View Raw LIME Table"):
                st.dataframe(lime_df.reset_index(drop=True), use_container_width=True)

        except Exception as e:
            st.error(f"LIME computation error: {e}")
            st.info("Tip: Ensure `lime` is installed.")

    # ── TAB 3: Feature Impact ─────────────────────────────────────────────────
    with tab3:
        st.markdown("Comparing this claim's key numeric values against typical ranges in the training distribution.")

        key_numeric = {
            "months_as_customer": ("Months as Customer", 0, 500),
            "age": ("Age", 16, 90),
            "policy_annual_premium": ("Annual Premium ($)", 500, 3000),
            "incident_hour_of_the_day": ("Incident Hour", 0, 23),
            "bodily_injuries": ("Bodily Injuries", 0, 2),
            "witnesses": ("Witnesses", 0, 3),
            "number_of_vehicles_involved": ("Vehicles Involved", 1, 4),
        }

        for col_name, (label, lo, hi) in key_numeric.items():
            val = input_df[col_name].values[0]
            pct = min(max((val - lo) / (hi - lo), 0), 1)
            st.markdown(f"""
            <div style="margin-bottom:1rem">
                <div style="display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:.3rem">
                    <span style="font-weight:600">{label}</span>
                    <span style="color:var(--subtext)">{val:.0f}</span>
                </div>
                <div class="risk-bar-wrap">
                    <div class="risk-bar-fill" style="width:{pct*100:.1f}%;background:var(--accent2)"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<span class="section-header">Categorical Flags</span>', unsafe_allow_html=True)
        flags = []
        if input_df.get("incident_severity_Total Loss", pd.Series([0])).values[0]:
            flags.append(("🔴 Total Loss Severity", "High-risk severity category"))
        if input_df.get("insured_hobbies_skydiving", pd.Series([0])).values[0]:
            flags.append(("🟠 Skydiving Hobby", "Correlated with fraud in training data"))
        if input_df.get("insured_hobbies_polo", pd.Series([0])).values[0]:
            flags.append(("🟠 Polo Hobby", "Correlated with fraud in training data"))
        if input_df.get("police_report_available_NO", pd.Series([0])).values[0]:
            flags.append(("🟡 No Police Report", "Absence of police report elevates risk"))
        if input_df.get("property_damage_YES", pd.Series([0])).values[0]:
            flags.append(("🟡 Property Damage Reported", "Property damage flag active"))

        if flags:
            for flag, desc in flags:
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;
                            padding:.6rem 1rem;margin:.4rem 0;display:flex;gap:.8rem;align-items:center">
                    <div style="font-weight:600;font-size:.9rem">{flag}</div>
                    <div style="font-size:.78rem;color:var(--subtext)">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No high-risk categorical flags detected for this claim.")

    # ── TAB 4: Claim Summary ──────────────────────────────────────────────────
    with tab4:
        st.markdown("Full snapshot of the claim data submitted for this prediction.")
        display_df = input_df.T.reset_index()
        display_df.columns = ["Feature", "Value"]
        display_df = display_df[display_df["Value"] != 0]
        st.dataframe(display_df, use_container_width=True, height=400)

        csv = display_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Claim Data (CSV)", csv, "claim_data.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    st.markdown("## ⚙️ Settings")

    st.markdown('<span class="section-header">🎨 Appearance</span>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        new_dark = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="settings_dark")
        if new_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark
            st.rerun()
        st.markdown(f'<p style="color:var(--subtext);font-size:.85rem">Currently using: <strong>{"Dark" if dark else "Light"} theme</strong></p>', unsafe_allow_html=True)

    st.markdown('<span class="section-header">🤖 Model Information</span>', unsafe_allow_html=True)
    info_cols = st.columns(3)
    model_info = [
        ("Algorithm", "LightGBM"),
        ("Features", str(len(feature_cols))),
        ("Training Records", "10,000"),
        ("Class Imbalance Fix", "SMOTE"),
        ("Validation", "5-Fold CV"),
        ("Explainability", "SHAP + LIME"),
    ]
    for i, (k, v) in enumerate(model_info):
        with info_cols[i % 3]:
            st.markdown(f'<div class="metric-card" style="text-align:left;margin-bottom:.7rem"><div style="font-size:.75rem;color:var(--subtext);font-weight:600;text-transform:uppercase">{k}</div><div style="font-size:1rem;font-weight:700;margin-top:.2rem">{v}</div></div>', unsafe_allow_html=True)

    st.markdown('<span class="section-header">📊 Performance Thresholds</span>', unsafe_allow_html=True)
    threshold = st.slider("Fraud Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability above this threshold is classified as fraud. Lower = more sensitive.")
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem">
        <div style="font-size:.85rem;color:var(--subtext)">
            At threshold <strong style="color:var(--text)">{threshold:.2f}</strong>: 
            claims with fraud probability ≥ {threshold*100:.0f}% will be flagged. 
            Lowering this catches more fraud but increases false positives.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-header">ℹ️ About FraudShield</span>', unsafe_allow_html=True)
    st.markdown("""
    **FraudShield v1.0** is an XAI-driven ensemble system for detecting fraudulent auto insurance 
    claims developed as a BSc Data Science final project at JOMO KENYATTA UNIVERSITY OF AGRICULTURE 
    AND TECHNOLOGY (JKUAT), 2026.

    - **Author:** Purity Njeri Mwaura  
    - **Supervisor:** Mr. Daniel Njuguna  
    - **Model:** LightGBM with SMOTE  
    - **Explainability:** SHAP (global) + LIME (local)  
    - **Dataset:** Auto Insurance Claims — Zenodo (10,000 records)
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RATE APP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Rate App":
    st.markdown("## ⭐ Rate FraudShield")
    st.markdown('<p style="color:var(--subtext);margin-top:-.5rem">Your feedback helps improve the app. Ratings are visible to everyone who uses FraudShield.</p>', unsafe_allow_html=True)

    col_form, col_wall = st.columns([1, 1])

    with col_form:
        st.markdown('<span class="section-header">Leave Your Rating</span>', unsafe_allow_html=True)

        reviewer_name = st.text_input("Your name (optional)", placeholder="Anonymous")

        st.markdown("**Overall Rating**")
        stars = st.select_slider(
            "Stars",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "⭐" * x,
            label_visibility="collapsed",
        )

        aspects = st.multiselect(
            "What did you like? (optional)",
            ["Easy to use", "Clear explanations", "Fast predictions",
             "Good design", "Helpful SHAP charts", "Helpful LIME charts",
             "Mobile-friendly", "Would use professionally"],
        )

        comment = st.text_area("Comment (optional)", placeholder="What did you think of FraudShield?", max_chars=300)

        col_heart, col_btn = st.columns([1, 2])
        with col_heart:
            heart = st.checkbox("❤️  Give a heart")
        with col_btn:
            if st.button("Submit Rating", use_container_width=True):
                entry = {
                    "name": reviewer_name or "Anonymous",
                    "stars": stars,
                    "heart": heart,
                    "aspects": aspects,
                    "comment": comment,
                }
                st.session_state.ratings.append(entry)
                st.success("Thanks for your feedback! 🙌")
                st.balloons()

    with col_wall:
        st.markdown('<span class="section-header">Community Ratings</span>', unsafe_allow_html=True)

        all_ratings = st.session_state.ratings
        if all_ratings:
            avg  = np.mean([r["stars"] for r in all_ratings])
            hearts = sum(1 for r in all_ratings if r["heart"])
            st.markdown(f"""
            <div style="display:flex;gap:1.5rem;margin-bottom:1.2rem;flex-wrap:wrap">
                <div class="metric-card">
                    <div class="val" style="color:var(--warning)">{avg:.1f} ⭐</div>
                    <div class="lbl">Avg Rating</div>
                </div>
                <div class="metric-card">
                    <div class="val" style="color:var(--danger)">{hearts} ❤️</div>
                    <div class="lbl">Hearts</div>
                </div>
                <div class="metric-card">
                    <div class="val" style="color:var(--accent2)">{len(all_ratings)}</div>
                    <div class="lbl">Total Ratings</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for r in reversed(all_ratings[-10:]):
                aspects_html = "".join(f'<span class="tag">{a}</span>' for a in r.get("aspects", []))
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;
                            padding:.9rem 1.1rem;margin-bottom:.7rem">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div style="font-weight:600;font-size:.9rem">{r['name']} {'❤️' if r['heart'] else ''}</div>
                        <div style="font-size:1rem">{"⭐" * r['stars']}</div>
                    </div>
                    {f'<div style="font-size:.82rem;color:var(--subtext);margin:.4rem 0">{r["comment"]}</div>' if r['comment'] else ''}
                    <div style="margin-top:.4rem">{aspects_html}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:var(--subtext)">
                <div style="font-size:2.5rem;margin-bottom:.5rem">🌟</div>
                <div style="font-size:1rem;font-weight:600">No ratings yet</div>
                <div style="font-size:.85rem;margin-top:.3rem">Be the first to rate FraudShield!</div>
            </div>
            """, unsafe_allow_html=True)
