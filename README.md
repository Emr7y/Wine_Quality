# Wine_Quality
🍇 Weinqualitäts-Vorhersage mit XGBoost &amp; Streamlit – auf Basis chemischer Analyse

# 🧪 Projekt: Weinqualitäts-Vorhersage mit XGBoost & Keras (Ordinal Classification)

## 📌 Projektübersicht
Diese App sagt die **Qualitätsstufe eines Weins** (zwischen 3 und 8) anhand seiner **chemischen Eigenschaften** voraus. Dabei wird ein **XGBoost-Klassifikationsmodell** verwendet, das in einer Streamlit-App eingebunden ist. Zusätzlich wurden experimentelle Tests mit **Keras (MLP)** durchgeführt.

> 🔗 [Hugging Face Demo](https://huggingface.co/spaces/emr7y/Wein_Qualitat)

## 🧠 Verwendete Modelle
- ✅ **Hauptmodell**: XGBoost (Ordinal Klassifikation)
- 🔬 **Zusätzlich getestet**: Keras MLP (neuronales Netz)

## 📂 Projektstruktur
```bash
├── w_qlty.py                 # Streamlit-App
├── wine_quality_model_xgb.pkl # Gespeichertes Modell
├── train.csv                 # Für Medianwerte und LabelEncoder
├── requirements.txt          # Abhängigkeiten für Hugging Face / pip install
└── README.md                 # Dieses Dokument
```

## 📈 Genauigkeit
Die erzielte Genauigkeit lag bei ca. **54 %**, was für ordinale Qualitätsklassifikation akzeptabel ist, da die Werte **von 3 bis 8** stark unausgeglichen verteilt sind (Klasse 5 und 6 überwiegen). Eine stärkere Gruppierung (z. B. binär: gut/schlecht) wäre einfacher zu modellieren, wurde aber hier **bewusst nicht gewählt**, um die Herausforderung beizubehalten.

## ⚙️ Installation
```bash
pip install -r requirements.txt
streamlit run w_qlty.py
```

## 📊 Features (Eingabedaten)
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

💡 Alle Eingaben sind mit **Tooltip-Erklärungen** versehen, um auch Nicht-Experten eine Einschätzung zu ermöglichen.

## ✅ Beispielausgabe
![Beispiel](https://huggingface.co/spaces/emr7y/Wein_Qualitat/resolve/main/example.png)

---
### 👤 Autor
**Made with ❤️ by Emr7y**  
Modell: XGBoost & Keras  
Datenquelle: [Kaggle: Wine Quality Dataset](https://www.kaggle.com/competitions/wine-quality-prediction-ordinal-regression-challe)

