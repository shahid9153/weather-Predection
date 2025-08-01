# 🌦️ ClimaCast — AI-Powered Weather Forecast Web App

ClimaCast is a modern, responsive, and intelligent weather forecasting application built with Streamlit. It combines real-time weather APIs, machine learning predictions, and a sleek dark/light mode UI to deliver an intuitive user experience.

---

## ✨ Features

- 📍 **Auto / Manual Location Detection**  
  Automatically detects your location using IP, or lets you search any city worldwide.

- 🌡️ **Live Weather Data (via Open-Meteo API)**  
  Displays current temperature, humidity, wind speed, and pressure.

- 🤖 **Machine Learning Predictions**  
  - 🔮 Predicts future temperature using regression  
  - 🌧️ Rainfall likelihood using classification models (with confidence %)

- 📊 **Historical Weather Visualization**  
  Interactive charts using Plotly to show:  
  - Temperature trends  
  - Rain occurrence (yes/no)

- 🌓 **Light & Dark Mode Toggle**  
  Beautiful UI themes with smooth toggling.

- 🔄 **One-Click Refresh**  
  Quickly update data with a single button.

---

## 🚀 Tech Stack

- **Frontend/UI:** Streamlit + HTML/CSS (custom theming)
- **Backend:** Python (Pandas, Requests)
- **Visualization:** Plotly Express
- **ML Models:** Scikit-learn (joblib-serialized regression & classification)
- **APIs Used:**
  - Open-Meteo (Live weather)
  - Nominatim (City geocoding)
  - IPInfo (Auto-location)

---

## 📷 Screenshots

### 🔵 Dark Mode
![Dark Mode](screenshot_UI/dark_mode.png)

### ⚪ Light Mode
![Light Mode](screenshot_UI/light_mode.png)

---

## 📂 Folder Structure
📁 weather_app/
┣ 📄 app.py
┣ 📁 models/ # Trained regression/classification models
┣ 📁 utils/ # Helper functions
┣ 📁 screenshot_UI/ # UI screenshots
┗ 📄 README.md
