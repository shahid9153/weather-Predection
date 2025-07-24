import pandas as pd

def load_and_clean_data(path="data/weather.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    if 'RainTomorrow' in df.columns:
        df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    X = df[['Humidity', 'Pressure', 'WindSpeed']]
    y_temp = df['Temperature']
    y_rain = df['RainTomorrow']
    return X, y_temp, y_rain
