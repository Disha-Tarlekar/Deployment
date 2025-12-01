import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ================= LOAD MODEL & SCALER =================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body { background-color:#f5f7fa; }
.main { background-color:#f5f7fa; }
h1 { font-weight:900; }
.input-card {
    background:white; padding:20px; border-radius:14px;
    box-shadow:0px 1px 8px rgba(0,0,0,0.12); margin-bottom:18px;
}
.output-card {
    padding:20px; border-radius:14px; text-align:center; font-size:26px;
    font-weight:700; color:white; margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ================= PAGE TITLE =================
st.markdown("<h1 style='text-align:center; color:#003566;'>🔍 ML-Based Customer Segmentation App</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center; font-size:18px;'>Enter customer details to identify segment & recommended retention strategy</p>", unsafe_allow_html=True)
st.write("")

# ================= SAMPLE DATA LOGIC =================
if "sample_clicked" not in st.session_state:
    st.session_state.sample_clicked = False

if st.button("🧪 Use Sample Data"):
    st.session_state.sample_clicked = True

sample = {
    "Monthly_Revenue": 2350.50,
    "Total_Revenue": 18890.00,
    "Tenure_Months": 26,
    "Avg_Monthly_Usage": 15.4,
    "Support_Tickets": 1,
    "Last_Active_Days": 10
} if st.session_state.sample_clicked else None

# ================= INPUT FORM =================
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

monthly_revenue = st.number_input(
    "💰 Monthly Revenue",
    min_value=0.0,
    value=float(sample["Monthly_Revenue"]) if sample else 0.0
)

total_revenue = st.number_input(
    "💵 Total Revenue",
    min_value=0.0,
    value=float(sample["Total_Revenue"]) if sample else 0.0
)

tenure_months = st.number_input(
    "📅 Tenure (Months)",
    min_value=0,
    value=int(sample["Tenure_Months"]) if sample else 0
)

avg_monthly_usage = st.number_input(
    "📊 Average Monthly Usage",
    min_value=0.0,
    value=float(sample["Avg_Monthly_Usage"]) if sample else 0.0
)

support_tickets = st.number_input(
    "🎫 Support Tickets",
    min_value=0,
    value=int(sample["Support_Tickets"]) if sample else 0
)

last_active_days = st.number_input(
    "📌 Last Active Days",
    min_value=0,
    value=int(sample["Last_Active_Days"]) if sample else 0
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= PREDICTION BUTTON =================
predict_clicked = st.button("🚀 Predict Customer Segment")

# ================= RESULTS =================
if predict_clicked:

    data = np.array([[monthly_revenue, total_revenue, tenure_months,
                      avg_monthly_usage, support_tickets, last_active_days]])
    scaled = scaler.transform(data)
    cluster = int(kmeans.predict(scaled)[0])

    # Confidence score
    distances = kmeans.transform(scaled)
    confidence = (1 - (distances[0][cluster] / distances[0].sum())) * 100

    # Cluster mapping
    if cluster == 0:
        persona = "💎 Loyal Premium Customer"
        rec = "Offer loyalty rewards, personalized premium benefits & upsell."
        color = "#2ecc71"
    elif cluster == 1:
        persona = "⚠️ High Churn Risk Customer"
        rec = "Provide discounts, priority support & retention call."
        color = "#f1c40f"
    else:
        persona = "📉 Low Usage / Low Value Customer"
        rec = "Educate via onboarding tutorials & increase product awareness."
        color = "#e74c3c"

    # Display results
    st.markdown(f"<div class='output-card' style='background:{color};'>{persona}</div>", unsafe_allow_html=True)
    st.metric("🔮 Confidence Score", f"{confidence:.2f}%")
    st.info(f"📝 Recommended Business Strategy: **{rec}**")

    # Micro insights
    if cluster == 0 and support_tickets > 3:
        st.write("🔍 High-value but frustrated customer — fix support urgently.")
    if cluster == 1 and total_revenue > 15000:
        st.write("🔍 Recoverable churn risk — valuable customer worth retaining.")
    if cluster == 2 and tenure_months < 6:
        st.write("🔍 Onboarding gap — new customer may not understand the product yet.")

    # ====== LOGGING ======
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
    st.success("📁 Prediction saved successfully!")
    st.balloons()
    st.write("🌐 App running successfully on Render!")

