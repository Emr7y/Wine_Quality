# Projekt: Vorhersage der Weinqualität (Ordinal Classification)
# -----------------------------------------------------------------------------

# 1. Bibliotheken laden
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 2. Modell und LabelEncoder laden
model = joblib.load("wine_quality_model_xgb.pkl")
le = LabelEncoder()
y_sample = pd.read_csv("train.csv")["quality"].astype(int)
le.fit(y_sample)

# 3. Streamlit App
# ----------------------------------
# streamlit run w_qlty.py

st.set_page_config(page_title="Weinqualitäts-Vorhersage", layout="wide")
st.title("🍷 Weinqualitäts-Vorhersage")
st.write("Diese App sagt die **Qualitätsstufe** eines Weins anhand seiner chemischen Merkmale vorher.")

st.subheader("🧾 Eingabedaten")

# Beispiel-Merkmale mit Tooltips (per Hover über Info-Icon)
columns = {
    'fixed acidity': 'Gesamtsäuregehalt (z. B. Weinsäure, Äpfelsäure etc.)',
    'volatile acidity': 'Flüchtige Säuren (z. B. Essigsäure) – hoher Wert = schlechter',
    'citric acid': 'Zitronensäure – trägt zur Frische bei',
    'residual sugar': 'Unvergorener Zucker – süßer Geschmack bei höherem Wert',
    'chlorides': 'Salzgehalt – hoher Wert kann negativ sein',
    'free sulfur dioxide': 'Ungebundener Schwefel – schützt vor Oxidation',
    'total sulfur dioxide': 'Gesamtschwefel – zu viel kann unangenehm sein',
    'density': 'Dichte – kann Hinweise auf Zucker- oder Alkoholgehalt geben',
    'pH': 'Säuregrad – niedriger pH = saurer Wein',
    'sulphates': 'Sulfate – konservierend und geschmacklich relevant',
    'alcohol': 'Alkoholgehalt in Volumenprozent'
}

inputs = {}
cols = st.columns(len(columns))
example_data = pd.read_csv("train.csv")
max_values = example_data[list(columns.keys())].max().to_dict()

for i, (col, tooltip) in enumerate(columns.items()):
    max_val = float(max_values[col]) + 1
    label = f"{col}"
    with cols[i]:
        if "sugar" in col or "sulfur" in col or "chloride" in col:
            inputs[col] = st.number_input(label, min_value=0.0, max_value=max_val, step=0.01, value=float(example_data[col].median()), help=tooltip)
        elif "density" in col:
            inputs[col] = st.number_input(label, min_value=0.0, max_value=10.0, step=0.01, value=float(example_data[col].median()), help=tooltip)
        else:
            inputs[col] = st.slider(label, float(example_data[col].min()), float(max_val), float(example_data[col].median()), help=tooltip)

input_df = pd.DataFrame([inputs])

st.write("\n", input_df)

# Vorhersage
prediction = model.predict(input_df)
quality = le.inverse_transform(prediction)[0]

st.subheader("📊 Vorhergesagte Weinqualität")

if quality <= 4:
    st.error(f"🍷 Die vorhergesagte Qualität ist: {quality} (schlecht)")
elif quality >= 7:
    st.success(f"🍷 Die vorhergesagte Qualität ist: {quality} (gut)")
else:
    st.info(f"🍷 Die vorhergesagte Qualität ist: {quality} (mittelmäßig)")

# Footer
st.markdown("""
---
Made with ❤️ by Emr7y | Modell: XGBoost | Datenquelle: Kaggle
""")
