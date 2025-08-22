# app.py — Streamlit web app for AI Career Guidance

import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# ---------- Paths ----------
MODEL_PATH = "models/ai_career_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
JOBS_PATH = "data/ai_job_market_insights.csv"
SALARIES_PATH = "data/salaries.csv"
LOG_FILE = "recommendation_history.csv"

# ---------- Page config ----------
st.set_page_config(page_title="AI Career Guidance", page_icon="🧭", layout="wide")

st.title("🧭 AI Career Guidance — Future Trends Predictor")
st.caption("Pick your profile → get industry recommendation → log and review history.")

# ---------- Cached loaders ----------
@st.cache_resource(show_spinner=False)
def load_model_and_features():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Model or feature file not found. Train and save them first.")
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols

@st.cache_data(show_spinner=False)
def load_valid_options():
    if not os.path.exists(JOBS_PATH) or not os.path.exists(SALARIES_PATH):
        raise FileNotFoundError("Datasets missing in /data. Please ensure CSVs exist.")

    job_df = pd.read_csv(JOBS_PATH)
    sal_df = pd.read_csv(SALARIES_PATH)

    job_df = job_df.rename(columns={"Job_Title": "job_title"})
    df = pd.merge(job_df, sal_df, on="job_title", how="inner")

    opts = {
        "job_titles": sorted(df["job_title"].dropna().unique().tolist()),
        "experience_levels": sorted(df["experience_level"].dropna().unique().tolist()),
        "company_sizes": sorted(df["company_size"].dropna().unique().tolist()),
        "locations": sorted(df["company_location"].dropna().unique().tolist()),
    }
    return opts

# ---------- Helpers ----------
def encode_align_input(input_dict, feature_columns):
    df = pd.DataFrame([input_dict])
    enc = pd.get_dummies(df)
    enc = enc.reindex(columns=feature_columns, fill_value=0)
    return enc

def log_recommendation(row_dict):
    """Append to history with fixed schema (no dynamic column names)."""
    required_cols = [
        "job_title", "experience_level", "company_size", "company_location",
        "salary_in_usd", "remote_ratio", "recommended_industry",
        "top1", "top2", "top3", "timestamp"
    ]
    for col in required_cols:
        if col not in row_dict:
            row_dict[col] = ""
    row = pd.DataFrame([row_dict], columns=required_cols)
    write_header = not os.path.exists(LOG_FILE)
    row.to_csv(LOG_FILE, mode="a", header=write_header, index=False)

def load_history():
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE)
        except Exception:
            os.remove(LOG_FILE)  # reset corrupted file
            return pd.DataFrame()
    return pd.DataFrame()

# ---------- Load resources ----------
try:
    model, feature_columns = load_model_and_features()
    options = load_valid_options()
except Exception as e:
    st.error(f"❌ Setup issue: {e}")
    st.stop()

# ---------- Sidebar: History ----------
with st.sidebar:
    st.header("📒 History")
    hist_df = load_history()
    if hist_df.empty:
        st.info("No recommendations logged yet.")
    else:
        st.dataframe(hist_df.tail(10), use_container_width=True)
        st.download_button(
            "⬇️ Download full history (CSV)",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="recommendation_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if st.button("🗑️ Clear history", type="secondary", use_container_width=True):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("History cleared.")
            st.rerun()
        else:
            st.warning("No history file to clear.")

# ---------- Welcome card ----------
last_box = st.empty()
if not hist_df.empty:
    last = hist_df.tail(1).to_dict("records")[0]
    with last_box.container():
        st.success(
            f"👋 Welcome back! Last recommendation: **{last.get('recommended_industry','?')}** "
            f"for **{last.get('job_title','?')}** ({last.get('experience_level','?')}) "
            f"in **{last.get('company_location','?')}** at **${last.get('salary_in_usd','?')}**, "
            f"on {last.get('timestamp','?')}."
        )

# ---------- Input form ----------
st.subheader("Tell us about your profile")

with st.form("profile_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        job_title = st.selectbox("Job Title", options["job_titles"], index=0)
        company_size = st.selectbox("Company Size", options["company_sizes"])
    with col2:
        experience_level = st.selectbox("Experience Level", options["experience_levels"])
        company_location = st.selectbox("Company Location", options["locations"])
    with col3:
        salary_in_usd = st.number_input("Expected Salary (USD)", min_value=10000, max_value=500000, value=80000, step=1000)
        remote_ratio = st.select_slider("Remote Ratio", options=[0, 50, 100], value=50)

    submitted = st.form_submit_button("🔮 Get Recommendation", use_container_width=True)

# ---------- Predict & display ----------
if submitted:
    user_input = {
        "job_title": job_title,
        "experience_level": experience_level,
        "company_size": company_size,
        "company_location": company_location,
        "salary_in_usd": int(salary_in_usd),
        "remote_ratio": int(remote_ratio),
    }

    # Encode
    X = encode_align_input(user_input, feature_columns)

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    prob_df = pd.DataFrame({"Industry": classes, "Probability": (proba * 100).round(2)}).sort_values(
        "Probability", ascending=False
    )

    st.success(f"💡 **Recommended Industry:** {pred}")
    st.caption("Model confidence across industries:")
    st.bar_chart(prob_df.set_index("Industry"))
    st.dataframe(prob_df, use_container_width=True)

    # Log with fixed schema
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top3 = prob_df.head(3).to_dict("records")
    log_row = {
        **user_input,
        "recommended_industry": pred,
        "top1": f"{top3[0]['Industry']} ({top3[0]['Probability']}%)" if len(top3) > 0 else "",
        "top2": f"{top3[1]['Industry']} ({top3[1]['Probability']}%)" if len(top3) > 1 else "",
        "top3": f"{top3[2]['Industry']} ({top3[2]['Probability']}%)" if len(top3) > 2 else "",
        "timestamp": now,
    }
    log_recommendation(log_row)
    st.toast("📂 Recommendation saved to history.", icon="✅")

    last_box.success(
        f"👋 Latest recommendation: **{pred}** for **{job_title}** ({experience_level}) "
        f"in **{company_location}** at **${int(salary_in_usd)}**, on {now}."
    )
