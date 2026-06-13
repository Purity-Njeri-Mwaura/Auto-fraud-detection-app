import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Insurance Fraud Detector",
    page_icon="🚗",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODEL + FEATURES  (plain .pkl files, loaded directly)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("fraud_lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, FEATURES = load_artifacts()

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# ─────────────────────────────────────────────────────────────────────────
# CATEGORY OPTIONS  (matched exactly to one-hot columns in model_features.pkl)
# NOTE: the first value in each list is the "dropped" baseline category —
#       it maps to all-zeros in the one-hot encoding, so it is still valid.
# ─────────────────────────────────────────────────────────────────────────
POLICY_STATE       = ["IL", "IN", "OH"]           # IL = baseline (no column)
POLICY_CSL         = ["100/300", "250/500", "500/1000"]
SEX                = ["FEMALE", "MALE"]
EDUCATION          = ["Associate", "College", "High School", "JD", "MD", "Masters", "PhD"]
OCCUPATION         = [
    "adm-clerical", "armed-forces", "craft-repair", "exec-managerial",
    "farming-fishing", "handlers-cleaners", "machine-op-inspct", "other-service",
    "priv-house-serv", "prof-specialty", "protective-serv", "sales",
    "tech-support", "transport-moving",
]
HOBBIES            = [
    "basketball", "board-games", "bungie-jumping", "camping", "chess",
    "cross-fit", "dancing", "exercise", "golf", "hiking", "kayaking", "movies",
    "paintball", "polo", "reading", "skydiving", "sleeping", "video-games", "yachting",
]
RELATIONSHIP       = ["husband", "not-in-family", "other-relative", "own-child", "unmarried", "wife"]
INCIDENT_TYPE      = ["Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"]
COLLISION_TYPE     = ["Unknown", "Front Collision", "Rear Collision", "Side Collision"]
INCIDENT_SEVERITY  = ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"]
AUTHORITIES        = ["Ambulance", "Fire", "Other", "Police", "Unknown"]
INCIDENT_STATE     = ["NC", "NY", "OH", "PA", "SC", "VA", "WV"]
INCIDENT_CITY      = ["Arlington", "Columbus", "Hillsdale", "Northbend", "Northbrook", "Riverwood", "Springfield"]
YES_NO_UNKNOWN     = ["Unknown", "NO", "YES"]
AUTO_MAKE          = [
    "Accura", "Audi", "BMW", "Chevrolet", "Dodge", "Ford", "Honda",
    "Jeep", "Mercedes", "Nissan", "Saab", "Suburu", "Toyota", "Volkswagen",
]
AUTO_MODEL         = [
    "95", "3 Series", "92x", "A3", "A5", "Accord", "C300", "CRV", "Camry",
    "Civic", "Corolla", "E400", "Escape", "F150", "Forrestor", "Fusion",
    "Grand Cherokee", "Highlander", "Impreza", "Jetta", "Legacy", "M5",
    "MDX", "ML350", "Malibu", "Maxima", "Neon", "Passat", "Pathfinder",
    "RAM", "RSX", "Silverado", "TL", "Tahoe", "Ultima", "Wrangler", "X5", "X6",
]

# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
st.title("🚗 Auto Insurance Fraud Detection System")
st.caption(
    "An XAI-driven ensemble system using LightGBM + SHAP — "
    "Final Year Project · Purity Njeri Mwaura (SCT213-C002-0021/2022) · JKUAT Karen"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Claim Details")

with st.form("claim_form"):

    with st.expander("👤 Policyholder Information", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age                    = st.number_input("Age", 18, 100, 35)
            months_as_customer     = st.number_input("Months as Customer", 0, 600, 120)
            insured_sex            = st.selectbox("Sex", SEX)
        with c2:
            insured_education_level = st.selectbox("Education Level", EDUCATION)
            insured_occupation      = st.selectbox("Occupation", OCCUPATION)
            insured_relationship    = st.selectbox("Relationship to Policyholder", RELATIONSHIP)
        with c3:
            insured_hobbies = st.selectbox("Hobby", HOBBIES)
            insured_zip     = st.number_input("ZIP Code", 0, 999999, 466132)

    with st.expander("📄 Policy Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            policy_state      = st.selectbox("Policy State", POLICY_STATE)
            policy_csl        = st.selectbox("Policy CSL", POLICY_CSL)
            policy_deductable = st.number_input("Policy Deductable ($)", 0, 5000, 1000)
        with c2:
            policy_annual_premium = st.number_input("Annual Premium ($)", 0.0, 5000.0, 1250.0)
            umbrella_limit        = st.number_input("Umbrella Limit ($)", 0, 10_000_000, 0)
            policy_year           = st.number_input("Policy Start Year", 1990, 2030, 2015)
        with c3:
            policy_month  = st.selectbox("Policy Start Month", list(range(1, 13)), index=0)
            capital_gains = st.number_input("Capital Gains ($)", 0, 200000, 0)
            capital_loss  = st.number_input("Capital Loss ($)", -200000, 0, 0)

    with st.expander("🚧 Incident Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            incident_type     = st.selectbox("Incident Type", INCIDENT_TYPE)
            collision_type    = st.selectbox("Collision Type", COLLISION_TYPE)
            incident_severity = st.selectbox("Incident Severity", INCIDENT_SEVERITY)
        with c2:
            incident_state            = st.selectbox("Incident State", INCIDENT_STATE)
            incident_city             = st.selectbox("Incident City", INCIDENT_CITY)
            incident_hour_of_the_day  = st.slider("Incident Hour of Day", 0, 23, 14)
        with c3:
            number_of_vehicles_involved = st.number_input("Vehicles Involved", 1, 10, 1)
            bodily_injuries             = st.number_input("Bodily Injuries", 0, 5, 0)
            witnesses                   = st.number_input("Witnesses", 0, 10, 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            authorities_contacted = st.selectbox("Authorities Contacted", AUTHORITIES)
        with c2:
            property_damage = st.selectbox("Property Damage?", YES_NO_UNKNOWN)
        with c3:
            police_report_available = st.selectbox("Police Report Available?", YES_NO_UNKNOWN)

        c1, c2 = st.columns(2)
        with c1:
            incident_year  = st.number_input("Incident Year", 1990, 2030, 2015)
        with c2:
            incident_month = st.selectbox("Incident Month", list(range(1, 13)), index=0)

    with st.expander("🚙 Vehicle Information", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            auto_make  = st.selectbox("Auto Make", AUTO_MAKE)
        with c2:
            auto_model = st.selectbox("Auto Model", AUTO_MODEL)
        with c3:
            auto_year  = st.number_input("Auto Year", 1990, 2030, 2010)

    submitted = st.form_submit_button(
        "🔍 Analyze Claim", use_container_width=True, type="primary"
    )

# ─────────────────────────────────────────────────────────────────────────
# BUILD FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────
def build_feature_vector():
    # start with all zeros (handles every one-hot column that isn't set)
    row = {f: 0 for f in FEATURES}

    # ── numeric fields ──────────────────────────────────────────────────
    numeric_map = {
        "months_as_customer":        months_as_customer,
        "age":                       age,
        "policy_number":             100000,   # placeholder — not predictive
        "policy_deductable":         policy_deductable,
        "policy_annual_premium":     policy_annual_premium,
        "umbrella_limit":            umbrella_limit,
        "insured_zip":               insured_zip,
        "capital-gains":             capital_gains,
        "capital-loss":              capital_loss,
        "incident_hour_of_the_day":  incident_hour_of_the_day,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "bodily_injuries":           bodily_injuries,
        "witnesses":                 witnesses,
        "auto_year":                 auto_year,
        "policy_year":               policy_year,
        "policy_month":              policy_month,
        "incident_year":             incident_year,
        "incident_month":            incident_month,
    }
    for k, v in numeric_map.items():
        if k in row:
            row[k] = v

    # ── one-hot helper ──────────────────────────────────────────────────
    # The model was trained with drop-first / baseline encoding,
    # so the FIRST category of each variable has NO column (all zeros = baseline).
    # We only set a column to 1 when a non-baseline category is selected.
    def set_onehot(prefix, value):
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1
        # if col not in row, it's the baseline category → leave as 0, which is correct

    set_onehot("policy_state",             policy_state)
    set_onehot("policy_csl",               policy_csl)
    set_onehot("insured_sex",              insured_sex)
    set_onehot("insured_education_level",  insured_education_level)
    set_onehot("insured_occupation",       insured_occupation)
    set_onehot("insured_hobbies",          insured_hobbies)
    set_onehot("insured_relationship",     insured_relationship)
    set_onehot("incident_type",            incident_type)
    set_onehot("collision_type",           collision_type)
    set_onehot("incident_severity",        incident_severity)
    set_onehot("authorities_contacted",    authorities_contacted)
    set_onehot("incident_state",           incident_state)
    set_onehot("incident_city",            incident_city)
    set_onehot("property_damage",          property_damage)
    set_onehot("police_report_available",  police_report_available)
    set_onehot("auto_make",                auto_make)
    set_onehot("auto_model",               auto_model)

    return pd.DataFrame([row], columns=FEATURES)

# ─────────────────────────────────────────────────────────────────────────
# PREDICTION + EXPLANATION
# ─────────────────────────────────────────────────────────────────────────
if submitted:
    X = build_feature_vector()

    proba      = model.predict_proba(X)[0][1]
    prediction = "FRAUD" if proba >= 0.5 else "LEGITIMATE"

    st.divider()
    st.subheader("🧾 Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == "FRAUD":
            st.error(f"### 🚨 {prediction}")
        else:
            st.success(f"### ✅ {prediction}")
        st.metric("Fraud Probability", f"{proba*100:.1f}%")

    with col2:
        if proba >= 0.7:
            risk_level, color = "HIGH",   "🔴"
        elif proba >= 0.4:
            risk_level, color = "MEDIUM", "🟠"
        else:
            risk_level, color = "LOW",    "🟢"
        st.metric("Risk Level", f"{color} {risk_level}")

    with col3:
        recommendation = "Investigate Further" if proba >= 0.4 else "Approve Claim"
        st.metric("Recommendation", recommendation)

    st.progress(float(np.clip(proba, 0.0, 1.0)))

    st.divider()

    # ── SHAP explanation ────────────────────────────────────────────────
    st.subheader("🔬 Why this decision? (SHAP Explanation)")

    shap_values = explainer.shap_values(X)

    # LightGBM binary classifier returns either:
    #   - a list [shap_class0, shap_class1]  (older shap versions)
    #   - a single 2-D array                 (newer shap versions)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    contrib = (
        pd.Series(sv, index=FEATURES)
        .sort_values(key=abs, ascending=False)
        .head(12)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in contrib.values]
    ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value  (impact on fraud probability)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 12 factors influencing this prediction")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "🔴 Red bars push the prediction **toward fraud**. "
        "🔵 Blue bars push the prediction **toward legitimate**."
    )

    st.markdown("**Top drivers for this claim:**")
    for feat, val in contrib.head(3).items():
        direction = "increased" if val > 0 else "decreased"
        st.markdown(
            f"- `{feat}` **{direction}** the fraud probability (SHAP impact: {val:+.3f})"
        )

st.divider()
st.caption(
    "Model: LightGBM — selected as best performer across Random Forest, XGBoost, "
    "LightGBM and CatBoost based on F1-score and ROC-AUC after SMOTE class-imbalance correction. "
    "Explainability via SHAP (TreeExplainer)."
)
