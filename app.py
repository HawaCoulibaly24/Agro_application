import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Accueil - AgroApp", page_icon="🌾")

# --- Titre de la page ---
st.title("🌾 Bienvenue sur AgroApp")
st.markdown("Une application dédiée à l'agriculture intelligente et à la gestion de l'eau.")

st.header("Objectif du projet")
st.markdown("""
Ce projet a pour but de **prédire la quantité moyenne d’eau nécessaire aux plantes** en fonction de plusieurs facteurs :
- Les **caractéristiques du sol** (N, P, K, pH, humidité, température, etc.),
- Les **conditions météorologiques** (pluie, température maximale/minimale, humidité relative, etc.),
- Les **besoins spécifiques des plantes** et la saison de culture.

L’objectif est d’**optimiser l’irrigation** pour chaque plante selon son environnement.

""")

st.header("Datasets utilisés")
st.markdown("""
Nous avons utilisé et fusionné **3 jeux de données principaux** :
- **Plantes.csv** : informations sur les plantes, leur famille, cycle de vie, besoin en eau, température minimale/maximale, type de sol, et saison idéale.
- **Sol.csv** : valeurs de N, P, K, température, humidité, pH, et précipitations.
- **Meteo.csv** : données climatiques journalières : RR, Tmax, Tmin, Umax, Umin.

Ces données ont été nettoyées, combinées puis enrichies afin d’obtenir un **dataset final prêt à l’analyse et à la prédiction**.
""")

# Option de chargement des fichiers JSON ou CSV
st.sidebar.header(" Données disponibles")
show_plante = st.sidebar.checkbox("Plantes")
show_sol = st.sidebar.checkbox("Sol")
show_meteo = st.sidebar.checkbox("Météo")
show_final = st.sidebar.checkbox("Dataset final fusionné")

# Chargement conditionnel
if show_plante:
    df_plante = pd.read_csv("Data/plantes_dataset_final.csv")
    st.subheader(" Données Plantes")
    st.dataframe(df_plante)

if show_sol:
    df_sol = pd.read_csv("Data/Crop_recommendation.csv")
    st.subheader("Données Sol")
    st.dataframe(df_sol)

if show_meteo:
    df_meteo = pd.read_csv("data/meteo_wide_data(1).csv")
    st.subheader("Données Météo")
    st.dataframe(df_meteo)

if show_final:
    df_final = pd.read_csv("Data/df_final.csv")
    st.subheader("Données Fusionnées Finales")
    st.dataframe(df_final)
