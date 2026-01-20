import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ===================== LOAD PIPELINE =====================
import joblib
import os

# Correct path to the model file
MODEL_PATH = os.path.join("scripts", "bike_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)

# ===================== PAGE TEMPLATE =====================
st.set_page_config(page_title="Bike Rental Prediction", page_icon="🚴", layout="wide")
st.title("🏍️ Bike Rental Demand Prediction ")
st.markdown("---")
st.subheader("📌 Project Group 5")

# ===================== SIDEBAR NAVIGATION =====================
st.sidebar.image(
    "https://th.bing.com/th/id/OIP.BRWSSyiMHebidKgb5Y5F6QHaHa?w=165&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
    width=150
)

st.sidebar.title("📍Navigation")
option = st.sidebar.radio(" **Choose Option** ", ["Single Prediction 🔮", "Upload CSV 📂", "About ℹ️"])
st.sidebar.markdown("---")

# ===================== FOOTER =====================
def footer():
    st.markdown("---")
    st.markdown("© 2026 Bike Rental Prediction App | Built with Streamlit")

# ===================== MAIN CONTENT =====================
if option == "Single Prediction 🔮":
    st.header("🔮 Single Prediction")

    # Mapping dictionaries for categorical features
    st.sidebar.header("📝 Select the Features")
    st.sidebar.markdown("Please choose categorical and numerical features below to generate predictions.")
    season_map = {"Spring": 0, "Summer": 1, "Fall": 2, "Winter": 3}
    year_map = {"2025": 0, "2026": 1}
    holiday_map = {"No Holiday": 0, "Holiday": 1}
    workingday_map = {"Non-working Day": 0, "Working Day": 1}
    weather_map = {"Clear": 1, "Mist/Cloudy": 2, "Light Rain": 3, "Heavy Rain/Snow": 4}

    # Sidebar inputs with labels
    season_label = st.sidebar.selectbox("Season", list(season_map.keys()))
    year_label = st.sidebar.selectbox("Year", list(year_map.keys()))
    holiday_label = st.sidebar.selectbox("Holiday", list(holiday_map.keys()))
    workingday_label = st.sidebar.selectbox("Working Day", list(workingday_map.keys()))
    weather_label = st.sidebar.selectbox("Weather Condition", list(weather_map.keys()))

    # Convert labels to numeric codes
    season = season_map[season_label]
    year = year_map[year_label]
    holiday = holiday_map[holiday_label]
    workingday = workingday_map[workingday_label]
    weather_condition = weather_map[weather_label]

    # Numeric inputs
    humidity = st.sidebar.number_input("Humidity (%)", value=60.0)
    windspeed = st.sidebar.number_input("Windspeed", value=0.0)
    hour = st.sidebar.slider("Hour of Day", 0, 23, 10)
    weekday = st.sidebar.slider("Weekday (0=Sun ... 6=Sat)", 0, 6, 2)
    month = st.sidebar.slider("Month", 1, 12, 6)
    temp = st.sidebar.number_input("Temperature (°C)", value=25.0)
    st.sidebar.markdown("---")

    # Derived features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    is_weekend = 1 if weekday in [0, 6] else 0
    comfort_index = temp - (humidity * 0.1)
    hour_type = 0 if hour < 6 else 1 if hour < 12 else 2 if hour < 18 else 3
    temp_feel_gap = temp - (humidity * 0.05)
    wind_temp_ratio = windspeed / (temp + 1)

    input_df = pd.DataFrame([{
        "season": season, "year": year, "holiday": holiday, "workingday": workingday,
        "weather_condition": weather_condition, "humidity": humidity, "windspeed": windspeed,
        "hour_sin": hour_sin, "hour_cos": hour_cos, "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos, "month_sin": month_sin, "month_cos": month_cos,
        "is_weekend": is_weekend, "comfort_index": comfort_index, "hour_type": hour_type,
        "temp_feel_gap": temp_feel_gap, "wind_temp_ratio": wind_temp_ratio
    }])

    # ===================== PREDICTION =====================
    if st.button("Predict"):
        prediction = pipeline.predict(input_df)[0]
        st.success(f"Predicted Rentals at {hour}:00 → {int(prediction)}")

        # Show selected parameters in categorical and numerical format
        st.markdown("### 📋 Selected Parameters")

        # Create categorical summary
        categorical_summary = pd.DataFrame({
            "Feature": ["Season", "Year", "Holiday", "Working Day", "Weather Condition"],
            "Selected Value": [season_label, year_label, holiday_label, workingday_label, weather_label]
        })

        # Create numerical summary
        numerical_summary = pd.DataFrame({
            "Feature": ["Temperature (°C)", "Humidity (%)", "Windspeed", "Hour", "Weekday", "Month"],
            "Selected Value": [temp, humidity, windspeed, hour, weekday, month]
        })

        st.subheader("🟦 Categorical Parameters")
        st.table(categorical_summary)

        st.subheader("🟩 Numerical Parameters")
        st.table(numerical_summary)

        # Graph
        hours_range = np.arange(0, 24)
        hourly_data = []
        for h in hours_range:
            hourly_data.append({
                **input_df.iloc[0].to_dict(),
                "hour_sin": np.sin(2 * np.pi * h / 24),
                "hour_cos": np.cos(2 * np.pi * h / 24),
                "hour_type": 0 if h < 6 else 1 if h < 12 else 2 if h < 18 else 3
            })
        hourly_df = pd.DataFrame(hourly_data)
        preds = pipeline.predict(hourly_df)

        plt.figure(figsize=(8,4))
        plt.plot(hours_range, preds, marker="o")
        plt.axvline(hour, color="red", linestyle="--", label="Selected Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Predicted Rentals")
        plt.title("Hourly Bike Rental Predictions")
        plt.legend()
        st.pyplot(plt)

    
        report_df = pd.DataFrame({"Hour": hours_range, "Predicted Rentals": preds.astype(int)})
        st.download_button("📥 Download Report", report_df.to_csv(index=False), "report.csv", "text/csv")

elif option == "Upload CSV 📂":
    st.header("📂 Upload CSV for Batch Prediction")

    
    st.markdown("### 📥 Download Sample Template")
    sample_template = pd.DataFrame([{
        "season": 1, "year": 0, "holiday": 0, "workingday": 1,
        "weather_condition": 2, "humidity": 55, "windspeed": 0.25,
        "hour_sin": 0.2588, "hour_cos": 0.9659,
        "weekday_sin": 0.7818, "weekday_cos": 0.6235,
        "month_sin": 0.5, "month_cos": 0.8660,
        "is_weekend": 0, "comfort_index": 22.5,
        "hour_type": 1, "temp_feel_gap": 23.0,
        "wind_temp_ratio": 0.01
    }])
    st.download_button(
        label="📥 Download CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="sample_template.csv",
        mime="text/csv"
    )


    uploaded_file = st.file_uploader("Upload CSV with features", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        preds = pipeline.predict(df)
        df["Predicted Rentals"] = preds.astype(int)

        st.write("📊 Predictions from uploaded file:")
        st.dataframe(df)

        
        if "cnt" in df.columns:
            r2 = r2_score(df["cnt"], preds)
            st.info(f"R² Score: {r2:.4f}")

        
        st.download_button(
            label="📥 Download Predictions",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
elif option == "About ℹ️":
    st.header("ℹ️ About This App")
    st.markdown("---")
    st.write("This application predicts bike rental demand using advanced machine learning models and engineered features derived from historical bike-sharing data.")
    st.write("The goal of this project is to help bike rental companies and city planners forecast demand accurately, optimize bike availability, and improve operational efficiency based on factors such as weather conditions, time, and seasonal trends.")
    st.write("**⚙️ Model & Feature Engineering**")
    st.write(" * Data preprocessing and feature engineering for improved prediction accuracy")

    st.write(" * Trained using multiple ML algorithms and evaluated using performance metrics")
    st.write(" * Designed for real-time, user-friendly predictions through an interactive interface")
    st.markdown("---")
    
    st.subheader("👨‍💻 Developed By")
    st.write(" * Project Group 5")

    st.subheader("📅 Project Generated")
    st.write(" * Year: 2025–2026")
    st.write(" * Domain: Data Science & Machine Learning")

    st.subheader("📧 Contact")

    st.write(" * For queries, feedback, or collaboration:")
    st.write("📩 Email: Sonipraveen220@gmail.com")

# Footer
footer()