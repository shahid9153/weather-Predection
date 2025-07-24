# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import requests
import plotly.express as px

st.set_page_config(page_title="🌦️ Weather Forecast", layout="wide")

# === 🌗 Dark / Light Mode Toggle ===
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("🌓 Choose Theme", ["Light Mode", "Dark Mode"], index=0)

# === 🆕 Features Sidebar ===
st.sidebar.markdown("---")
st.sidebar.markdown("**✨ Features:**")
st.sidebar.markdown("- 📍 Auto / Manual Location")
st.sidebar.markdown("- 🌡️ Live Temperature")
st.sidebar.markdown("- 🔮 ML Predictions")
st.sidebar.markdown("- 📊 Weather History")
st.sidebar.markdown("- 🌓 Theme Toggle")
refresh = st.sidebar.button("🔄 Refresh")
if refresh:
    try:
        st.experimental_rerun()
    except Exception:
        st.warning("Page is already reloading… please wait.")







# === 💅 CSS Styling ===
light_css = """
<style>
body {
    background-color: #f5f5f5;
}
.big-temp {
    font-size: 48px; font-weight: bold; color: #ff7300;
}
.info-box {
    background: linear-gradient(to right, #e0f7fa, #f1f8e9);
    padding: 12px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 16px;
    color: #333;
}
.rain-box {
    background-color: #e3f2fd;
    border-left: 6px solid #2196f3;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
    color: #333;
}
@media (max-width: 768px) {
    .big-temp { font-size: 36px; }
}
</style>
"""

dark_css = """
<style>
body {
    background-color: #121212;
    color: #eee;
}
.big-temp {
    font-size: 48px; font-weight: bold; color: #ffb347;
}
.info-box {
    background: linear-gradient(to right, #263238, #37474f);
    padding: 12px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 16px;
    color: #eee;
}
.rain-box {
    background-color: #1e3a5f;
    border-left: 6px solid #64b5f6;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
    color: #eee;
}
@media (max-width: 768px) {
    .big-temp { font-size: 36px; }
}
</style>
"""

st.markdown(dark_css if theme == "Dark Mode" else light_css, unsafe_allow_html=True)

st.title("🌦️ Weather Forecast")

# === 📦 Load Models ===
reg_model = joblib.load("model_regression.pkl")
clf_model = joblib.load("model_classification.pkl")

# === 📍 Location Detection ===
use_auto = st.toggle("📍 Use Automatic Location", value=True)

default_city = "Solapur"
default_country = "India"
default_lat, default_lon = 17.6599, 75.9064

if use_auto:
    try:
        geo = requests.get("https://ipinfo.io/json", timeout=5).json()
        loc = geo['loc'].split(',')
        lat, lon = float(loc[0]), float(loc[1])
        city = geo.get('city', default_city)
        country = geo.get('country', default_country)
    except:
        st.warning("Auto-location failed. Using default.")
        lat, lon, city, country = default_lat, default_lon, default_city, default_country
else:
    city_input = st.text_input("Enter a city name", "Solapur")

    if city_input:
        try:
            headers = {"User-Agent": "weather-app/1.0 (your_email@example.com)"}
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city_input}&format=json"
            response = requests.get(geo_url, headers=headers, timeout=5)

            if response.status_code == 200:
                geo_data = response.json()
                if geo_data:
                    lat = float(geo_data[0]["lat"])
                    lon = float(geo_data[0]["lon"])
                    city = geo_data[0]["display_name"].split(",")[0]
                    country = geo_data[0]["display_name"].split(",")[-1].strip()
                else:
                    st.warning("City not found. Using fallback.")
                    lat, lon, city, country = default_lat, default_lon, default_city, default_country
            else:
                st.error("Error fetching city. Using fallback.")
                lat, lon, city, country = default_lat, default_lon, default_city, default_country
        except Exception as e:
            st.error("Location fetch failed.")
            st.exception(e)
            lat, lon, city, country = default_lat, default_lon, default_city, default_country

# Show Resolved Location
st.markdown(f"### 📍 Location: **{city}, {country}**")

# === 🌐 Weather API Call ===
try:
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,"
        f"surface_pressure,wind_speed_10m&forecast_days=1&timezone=auto"
    )
    res = requests.get(url)
    data = res.json()["current"]

    temp = data["temperature_2m"]
    humidity = data["relative_humidity_2m"]
    pressure = data["surface_pressure"]
    wind = data["wind_speed_10m"]

    st.markdown(f"<div class='big-temp'>🌡️ {temp}°C</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='info-box'>
        💧 Humidity: {humidity}%<br>
        📈 Pressure: {pressure} hPa<br>
        💨 Wind Speed: {wind} km/h
        </div>
    """, unsafe_allow_html=True)

    # === 🔮 Prediction ===
    df_input = pd.DataFrame([[humidity, pressure, wind]], columns=["Humidity", "Pressure", "WindSpeed"])
    pred_temp = reg_model.predict(df_input)[0]
    rain_pred = clf_model.predict(df_input)[0]
    rain_conf = clf_model.predict_proba(df_input)[0][1]

    st.markdown("""
        <div class='rain-box'>
        🔮 <strong>Predicted Temp:</strong> {:.1f}°C<br>
        🌧️ <strong>Rain?</strong> {} ({:.1f}% confidence)
        </div>
    """.format(pred_temp, "Yes" if rain_pred else "No", rain_conf * 100), unsafe_allow_html=True)

except Exception as e:
    st.error("Weather fetch failed.")
    st.exception(e)

# === 📊 Past Weather Data ===
st.subheader("📊 Past Weather History")
df = pd.read_csv("data/weather.csv")
st.plotly_chart(px.line(df, y="Temperature", title="Temperature Over Time"), use_container_width=True)
st.plotly_chart(px.scatter(df, y="RainTomorrow", title="Rain Occurrence (1=Yes, 0=No)"), use_container_width=True)
