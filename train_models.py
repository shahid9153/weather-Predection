# train_models.py
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("data/weather.csv")
df = df.dropna()

X = df[["Humidity", "Pressure", "WindSpeed"]]
y_temp = df["Temperature"]

reg = LinearRegression()
reg.fit(X, y_temp)
joblib.dump(reg, "model_regression.pkl")

df["RainTomorrow"] = LabelEncoder().fit_transform(df["RainTomorrow"])
y_rain = df["RainTomorrow"]

clf = LogisticRegression()
clf.fit(X, y_rain)
joblib.dump(clf, "model_classification.pkl")

print("✅ Models trained and saved.")
