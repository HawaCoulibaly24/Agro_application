import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Chargement des modèles et encodeurs ---
model = joblib.load("models/gb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder_plante = joblib.load("models/encoder_plante.pkl")
encoder_type_sol = joblib.load("models/encoder_type_sol.pkl")
encoder_saison = joblib.load("models/encoder_saison.pkl")

st.set_page_config(page_title="Prédiction du besoin en eau", page_icon="💧")

st.title("💧 Prédiction du besoin en eau pour les cultures")

st.markdown("""
Ce formulaire permet d'estimer le **besoin en eau** pour une plante donnée en fonction des caractéristiques du **sol**, de la **plante** et des **conditions météorologiques**.
""")

# --- Données du sol ---
st.header(" Données du sol")
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Azote (N)", placeholder=50.39)
    temperature = st.number_input("Température (°C)", placeholder=25.59)
    ph = st.number_input("pH", placeholder=6.48)
with col2:
    P = st.number_input("Phosphore (P)", placeholder=53.44)
    humidity = st.number_input("Humidité (%)", placeholder=71.23)
    rainfall = st.number_input("Précipitations (mm)", placeholder=103.15)
with col3:
    K = st.number_input("Potassium (K)", placeholder=48.08)
    type_sol = st.selectbox("Type de sol", ['limoneux', 'sablo-limoneux', 'argilo-limoneux', 'sablé',
                                             'volcanique', 'argileux', 'calcaire', 'sablo-argileux', 'alluvial'])

# --- Données de la plante ---
st.header(" Données de la plante")
dictionnaire_plantes = {
    "Riz": "rice",
    "Pois chiche": "chickpea",
    "Lentille": "lentil",
    "Grenade": "pomegranate",
    "Banane": "banana",
    "Pomme": "apple",
    "Orange": "orange",
    "Papaye": "papaya",
    "Café": "coffee",
    "Maïs": "maize",
    "Haricot rouge": "kidneybeans",
    "Pois d'Angole": "pigeonpeas",
    "Haricot Moth": "mothbeans",
    "Haricot mungo": "mungbean",
    "Urd (haricot noir)": "blackgram",
    "Mangue": "mango",
    "Pastèque": "watermelon",
    "Melon": "muskmelon",
    "Coco": "coconut",
    "Coton": "cotton",
    "Jute": "jute",
    "Raisin": "grapes"
}

nom_plante_affiche = st.selectbox("Nom de la plante", list(dictionnaire_plantes.keys()))
nom_plante_code = dictionnaire_plantes[nom_plante_affiche]
saison = st.selectbox("Saison", ['Hiver', 'Printemps', 'Été', 'Automne'])

# --- Données météo ---
st.header("Données météorologiques")
col4, col5, col6 = st.columns(3)
with col4:
    RR = st.number_input("Précipitations RR (mm)", placeholder=4.99)
    TMAX = st.number_input("Température maximale (°C)", placeholder=35.47)
with col5:
    TMIN = st.number_input("Température minimale (°C)", placeholder=24.19)
    UMAX = st.number_input("Humidité maximale (%)", placeholder=74.02)
with col6:
    UMIN = st.number_input("Humidité minimale (%)", placeholder=47.01)

# --- Bouton de prédiction ---
if st.button("Prédire le besoin en eau"):
    try:
        # Encodage
        plante_encoded = encoder_plante.transform([nom_plante_code])[0]
        sol_encoded = encoder_type_sol.transform([type_sol])[0]
        saison_encoded = encoder_saison.transform([saison])[0]

        # Construction du DataFrame
        data = pd.DataFrame([{
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall,
            'nom_plante': plante_encoded,
            'type_sol': sol_encoded,
            'RR': RR,
            'TMAX': TMAX,
            'TMIN': TMIN,
            'UMAX': UMAX,
            'UMIN': UMIN,
            'Saison': saison_encoded
        }])

        # Normalisation
        data_scaled = scaler.transform(data)

        # Prédiction
        prediction = model.predict(data_scaled)[0]

        st.success(f" Besoin en eau estimé : **{prediction:.2f} mm/cycle**")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
