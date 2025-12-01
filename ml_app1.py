import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model & scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_COLS = [
    "Monthly_Revenue", "Total_Revenue", "Tenure_Months",
    "Avg_Monthly_Usage", "Support_Tickets", "Last_Active_Days"
]

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# --------------- PREMIUM DASHBOARD THEME ----------------
st.markdown("""
<style>
body { background-color:#f0f4ff; }
.main { background-color:#f0f4ff; }
h1 { font-weight:900; }
.sidebar .sidebar-content { background-color:#082032; color:white; }
footer {visibility: hidden;}
.reportview-container .main footer {visibility: hidden;}

.box-title {
    background:#001D4A; padding:15px; color:white;
    font-size:26px; font-weight:700; text-align:center;
    border-radius:10px; margin-bottom:18px;
}
.card {
    background:white; padding:18px; border-radius:14px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.10);
    margin-bottom:20px;
}
.result-banner {
    padding:18px; border-radius:12px; font-size:24px;
    font-weight:700; text-align:center; color:white;
}
.footer {
    text-align:center; padding:12px; margin-top:30px;
    font-size:14px; font-weight:600; color:#003566;
}
</style>
""", unsafe_allow_html=True)

# ---------------- APP TITLE ----------------
st.markdown("<div class='box-title'>🔍 CUSTOMER SEGMENTATION & CHURN RISK PREDICTOR</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 Navigation Menu")
section = st.sidebar.radio("Choose section", ["Single Prediction", "Batch Prediction (CSV)", "Prediction History"])

# ---------------- SINGLE PREDICTION ----------------
if section == "Single Prediction":
    st.subheader("👤 Predict a Single Customer")

    if "sample_clicked" not in st.session_state:
        st.session_state.sample_clicked = False

    if st.button("🧪 Use Sample Input"):
        st.session_state.sample_clicked = True

    sample = {
        "Monthly_Revenue": 2350.50,
        "Total_Revenue": 18890,
        "Tenure_Months": 26,
        "Avg_Monthly_Usage": 15.4,
        "Support_Tickets": 1,
        "Last_Active_Days": 10
    } if st.session_state.sample_clicked else None

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_revenue = st.number_input("💰 Monthly Revenue", value=float(sample["Monthly_Revenue"]) if sample else 0.0)
        tenure_months = st.number_input("📅 Tenure (Months)", value=int(sample["Tenure_Months"]) if sample else 0)

    with col2:
        total_revenue = st.number_input("💵 Total Revenue", value=float(sample["Total_Revenue"]) if sample else 0.0)
        avg_monthly_usage = st.number_input("📊 Average Monthly Usage", value=float(sample["Avg_Monthly_Usage"]) if sample else 0.0)

    with col3:
        support_tickets = st.number_input("🎫 Support Tickets", value=int(sample["Support_Tickets"]) if sample else 0)
        last_active_days = st.number_input("📌 Last Active Days", value=int(sample["Last_Active_Days"]) if sample else 0)

    if st.button("🚀 Predict Segment"):
        data = np.array([[monthly_revenue, total_revenue, tenure_months,
                          avg_monthly_usage, support_tickets, last_active_days]])
        scaled = scaler.transform(data)
        cluster = int(kmeans.predict(scaled)[0])
        distances = kmeans.transform(scaled)
        confidence = (1 - (distances[0][cluster] / distances[0].sum())) * 100

        personas = {
            0: ("💎 Loyal Premium Customer", "#2ecc71"),
            1: ("⚠️ High Churn Risk Customer", "#e0a800"),
            2: ("📉 Low Usage / Low Value Customer", "#d9534f")
        }
        persona, color = personas[cluster]

        st.markdown(f"<div class='result-banner' style='background:{color};'>{persona}</div>", unsafe_allow_html=True)
        st.metric("🔮 Confidence Score", f"{confidence:.2f}%")
        st.progress(confidence / 100)

        if cluster == 0 and support_tickets > 3:
            st.warning("⚠ High-value customer but frustrated — fix customer support urgently.")
        if cluster == 1 and total_revenue > 15000:
            st.info("🎯 Recoverable churn risk — customer is high revenue.")
        if cluster == 2 and tenure_months < 6:
            st.info("📌 New customer with low usage — onboarding strategy recommended.")

        record = {
            "Monthly_Revenue": monthly_revenue,
            "Total_Revenue": total_revenue,
            "Tenure_Months": tenure_months,
            "Avg_Monthly_Usage": avg_monthly_usage,
            "Support_Tickets": support_tickets,
            "Last_Active_Days": last_active_days,
            "Predicted_Cluster": cluster,
            "Confidence": round(confidence, 2)
        }

        try:
            df = pd.read_csv("prediction_logs.csv")
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame([record])

        df.to_csv("prediction_logs.csv", index=False)
        st.success("📌 Prediction logged successfully ✔")

# ---------------- BATCH PREDICTION ----------------
elif section == "Batch Prediction (CSV)":
    st.subheader("📂 Predict Churn Risk for Multiple Customers (Upload CSV)")
    st.code(", ".join(FEATURE_COLS))
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        missing = [c for c in FEATURE_COLS if c not in df.columns]

        if missing:
            st.error(f"❌ Missing columns in uploaded CSV: {missing}")
        else:
            X = df[FEATURE_COLS].values
            X_scaled = scaler.transform(X)
            clusters = kmeans.predict(X_scaled)
            distances = kmeans.transform(X_scaled)
            conf = [(1 - (distances[i][clusters[i]] / distances[i].sum())) * 100 for i in range(len(clusters))]
            df["Predicted_Cluster"] = clusters
            df["Confidence"] = [round(c, 2) for c in conf]

            st.success("🟢 Batch Prediction Complete!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "batch_predictions.csv", "text/csv")

# ---------------- PREDICTION HISTORY ----------------
else:
    st.subheader("📜 Prediction History")
    try:
        df = pd.read_csv("prediction_logs.csv")
        st.dataframe(df)
        st.success("🔁 Loaded Prediction History")
    except:
        st.info("No prediction logs found — try first prediction.")

# ---------------- FOOTER ----------------
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    background: #0d1117;
    padding: 10px 0;
    font-size: 16px;
    color: #9fb3ff;
    border-top: 1px solid #1f2633;
    font-weight: 500;
}
.footer b {
    color: #dbe4ff;
}
.footer:hover {
    color: #ffffff;
    transition: 0.3s;
}
.footer .line2 {
    font-size: 14px;
    opacity: 0.8;
}
.footer .line3 {
    font-size: 13px;
    opacity: 0.6;
    font-style: italic;
}
</style>

<div class="footer">
    🚀 Built with ❤️ by <a href="https://www.linkedin.com/in/disha-tarlekar" target="_blank"><b>Disha Tarlekar</b></a><br>
    📌 Customer Churn Machine Learning Segmentation App<br>
    🔍 Empowering businesses with data-driven customer retention
</div>
""", unsafe_allow_html=True)


