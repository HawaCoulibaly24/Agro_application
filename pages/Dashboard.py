import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des modèles et encoders
model = joblib.load("models/gb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder_plante = joblib.load("models/encoder_plante.pkl")
encoder_sol = joblib.load("models/encoder_type_sol.pkl")
encoder_saison = joblib.load("models/encoder_saison.pkl")

# Dictionnaires pour décoder les labels
decoder_plante = {v: k for k, v in zip(encoder_plante.classes_, range(len(encoder_plante.classes_)))}
decoder_sol = {v: k for k, v in zip(encoder_sol.classes_, range(len(encoder_sol.classes_)))}
decoder_saison = {v: k for k, v in zip(encoder_saison.classes_, range(len(encoder_saison.classes_)))}

# Configuration de la page
st.set_page_config(page_title="Dashboard d'analyse", layout="wide")
# --- Titre du dashboard ---
st.title("📊 Tableau de bord - Analyse des données agricoles")

#st.sidebar.title("📊 Tableau de bord interactif")

# === Upload de données ===
uploaded_file = st.sidebar.file_uploader(" Charger un fichier ", type=["json"])
df = None

if uploaded_file:
    try:
        df = pd.read_json(uploaded_file)

        # Encodage si nécessaire
        if df["nom_plante"].dtype == object:
            df["nom_plante"] = encoder_plante.transform(df["nom_plante"])
        if df["type_sol"].dtype == object:
            df["type_sol"] = encoder_sol.transform(df["type_sol"])
        if df["Saison"].dtype == object:
            df["Saison"] = encoder_saison.transform(df["Saison"])

        # Prédictions
        features = [
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
            'nom_plante', 'type_sol', 'RR', 'TMAX', 'TMIN', 'UMAX', 'UMIN', 'Saison'
        ]
        X = df[features]
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        df["Eau_moyenne_mm"] = y_pred

        # Décodage pour affichage lisible
        df["Plante_label"] = df["nom_plante"].map(decoder_plante)
        df["Sol_label"] = df["type_sol"].map(decoder_sol)
        df["Saison_label"] = df["Saison"].map(decoder_saison)
        

        st.sidebar.success(" Fichier chargé avec succès.")

        # Affichage conditionnel
        if st.sidebar.checkbox(" Afficher les données"):
            st.subheader("Aperçu du jeu de données")
            st.dataframe(df[features + ["Eau_moyenne_mm"]].head())

        # === KPIs ===
        st.markdown("###  Indicateurs clés (KPIs)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Nb d'observations", len(df))
        col2.metric("Température Moyenne (°C)", f"{df['temperature'].mean():.1f}")
        col3.metric("Humidité Moyenne (%)", f"{df['humidity'].mean():.1f}")
        col4.metric("pH Moyen", f"{df['ph'].mean():.2f}")
        col5.metric("Besoins en eau moyens (mm)", f"{df['Eau_moyenne_mm'].mean():.1f}")

        # === Performance du modèle ===
        if "Eau_moyenne_mm_reel" in df.columns:
            y_true = df["Eau_moyenne_mm_reel"]
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            st.markdown(f" **MSE** : `{mse:.2f}`")
            st.markdown(f" **R²** : `{r2:.2f}`")

        # === Graphiques ===
        st.sidebar.subheader(" Visualisation")
        graph_type = st.sidebar.selectbox("Type de graphique", ["Barres", "Boîte à moustaches", "Nuage de points"])
        variable_x = st.sidebar.selectbox("Variable X", ["Plante_label", "Sol_label", "Saison_label"] + features)
        variable_y = st.sidebar.selectbox("Variable Y", ["Eau_moyenne_mm"] + features)

        st.subheader("Graphiques")

        if graph_type == "Barres":
            fig, ax = plt.subplots()
            grouped = df.groupby(variable_x)[variable_y].mean().sort_values()
            grouped.plot(kind="bar", ax=ax)
            ax.set_ylabel(f"Moyenne de {variable_y}")
            ax.set_xlabel(variable_x)
            ax.set_title(f"Moyenne de {variable_y} selon {variable_x}")
            st.pyplot(fig)

        elif graph_type == "Boîte à moustaches":
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=variable_x, y=variable_y, ax=ax)
            ax.set_title(f"Distribution de {variable_y} par {variable_x}")
            st.pyplot(fig)

        elif graph_type == "Nuage de points":
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=variable_x, y=variable_y, ax=ax)
            ax.set_title(f"{variable_y} en fonction de {variable_x}")
            st.pyplot(fig)

    except Exception as e:
        st.error(f" Une erreur est survenue : {e}")

else:
    st.info("Veuillez charger un fichier à analyser.")
