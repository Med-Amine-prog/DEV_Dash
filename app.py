import os
import json
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import re
import time
from difflib import get_close_matches
from constants import nationalities

# Configuration pour accéder à Google Sheets
def load_google_sheet(sheet_url):
    # Charger les identifiants de Google depuis la variable d'environnement
    def load_google_credentials():
        # Lire la variable d'environnement
        credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')

        if not credentials_json:
            raise ValueError("La variable d'environnement 'GOOGLE_CREDENTIALS_JSON' n'est pas définie.")

        # Convertir la chaîne JSON en dictionnaire Python
        credentials_dict = json.loads(credentials_json)

        # Créer les identifiants à partir du dictionnaire
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict)

        return creds

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = load_google_credentials()  # Obtenez les identifiants en utilisant la fonction modifiée
    client = gspread.authorize(creds)

    # Retry logic
    retries = 3
    for i in range(retries):
        try:
            sheet = client.open_by_url(sheet_url).sheet1
            data = sheet.get_all_records()
            df = pd.DataFrame(data)

            # Nettoyer les noms de colonnes
            df.columns = clean_column_names(df.columns)
            return df
        except gspread.exceptions.APIError as e:
            st.warning(f"Tentative {i+1} échouée: {e}")
            if i < retries - 1:  # No delay after the last attempt
                time.sleep(2)  # Wait for 2 seconds before retrying

    st.error("Impossible de charger les données après plusieurs tentatives.")
    return None

def clean_column_names(columns):
    # Fonction pour nettoyer les noms de colonnes en supprimant les caractères spéciaux
    return [re.sub(r'[^\w\s]', '', col) for col in columns]

# Fonction pour normaliser les nationalités
def normalize_nationality(nationality, nationalities):
    # Trouver la nationalité la plus proche dans la liste
    matches = get_close_matches(nationality, nationalities, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    else:
        return nationality

# Fonction pour créer une carte choropleth
def make_choropleth(input_df, input_column, input_color_theme):
    if input_column in input_df.columns:
        # Filtrer les valeurs NA avant de compter les nationalités
        nationality_counts = input_df[input_column].dropna().value_counts().reset_index()
        nationality_counts.columns = ['Nationalité', 'Count']

        choropleth = px.choropleth(
            nationality_counts, 
            locations='Nationalité', 
            color='Count', 
            locationmode='country names',
            color_continuous_scale=input_color_theme,
            range_color=(0, nationality_counts['Count'].max()),
            labels={'Count': 'Nombre'},
            scope='world'
        )

        choropleth.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=500
        )
        return choropleth
    else:
        st.error(f"La colonne '{input_column}' n'existe pas dans les données.")
        return None

# Fonction pour créer un graphique
def create_graph(df, selected_column, graph_type, container):
    # Filtrer les valeurs NA avant de créer les graphiques
    filtered_df = df[selected_column].replace('', pd.NA).dropna()

    with container:
        if graph_type == "Bar":
            st.bar_chart(filtered_df.value_counts())
        elif graph_type == "Line":
            st.line_chart(filtered_df.value_counts().sort_index())
        elif graph_type == "Area":
            st.area_chart(filtered_df.value_counts().sort_index())
        elif graph_type == "Pie":
            # Utilisation de Plotly pour le graphique circulaire
            fig = px.pie(filtered_df, names=selected_column, title=f"Pie Chart of {selected_column}")
            st.plotly_chart(fig)
        elif graph_type == "Histogram":
            # Utilisation de Plotly pour le histogramme
            fig = px.histogram(filtered_df, x=selected_column, title=f"Histogram of {selected_column}")
            st.plotly_chart(fig)
        elif graph_type == "Map":
            fig = make_choropleth(df, selected_column, 'Viridis')
            st.plotly_chart(fig, use_container_width=True)

# Interface utilisateur Streamlit
def main():
    st.set_page_config(layout="wide")  # Mettre en mode "large"

    st.title("Analyse des données des start-ups")

    # Initialise une liste vide pour stocker les graphiques ajoutés
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = []

    # Entrée de l'utilisateur pour l'URL de la feuille Google
    sheet_url = st.text_input("Entrez le lien de votre Google Sheet", "")

    if sheet_url:
        try:
            df = load_google_sheet(sheet_url)
            if df is not None:
                st.success("Données chargées avec succès !")
                st.write(df.head())

                # Sélection des colonnes
                columns = df.columns.tolist()
                selected_column = st.selectbox("Choisissez une colonne à visualiser", columns)

                # Appliquer la normalisation des nationalités si applicable
                if selected_column in columns:
                    df[selected_column] = df[selected_column].apply(lambda x: normalize_nationality(str(x), nationalities))

                # Proposer des types de graphiques
                graph_type = st.selectbox("Choisissez un type de graphique", ["Bar", "Line", "Area", "Pie", "Histogram", "Map"])

                # Utiliser un conteneur vide pour le graphique temporaire
                temp_container = st.empty()

                # Générer le graphique temporaire selon le choix de l'utilisateur
                if selected_column and graph_type:
                    create_graph(df, selected_column, graph_type, temp_container)

                    # Bouton pour ajouter le graphique au tableau de bord
                    if st.button("Ajouter au Tableau de Bord"):
                        # Ajouter le type de graphique et la colonne choisie à la session
                        st.session_state.dashboard.append((selected_column, graph_type))
                        st.success("Graphique ajouté au tableau de bord !")
                        # Effacer le graphique temporaire après l'ajout
                        temp_container.empty()

        except Exception as e:
            st.error("Erreur lors du chargement des données: " + str(e))

    # Afficher le tableau de bord avec les graphiques ajoutés
    st.header("Tableau de Bord Personnalisé")

    # Boucle pour afficher chaque graphique ajouté en deux colonnes
    for index, (column, graph) in enumerate(st.session_state.dashboard):
        if index % 2 == 0:
            col1, col2 = st.columns(2)
        
        with (col1 if index % 2 == 0 else col2):
            st.subheader(f"{graph} Graphe pour : {column}")
            # Créer un nouveau conteneur pour chaque graphique du tableau de bord
            create_graph(df, column, graph, st.container())

            # Bouton pour supprimer le graphique
            if st.button("Supprimer", key=f"delete-{index}"):
                # Supprimer le graphique du tableau de bord
                st.session_state.dashboard.pop(index)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
