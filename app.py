import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import re
import time
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from Sauvegarde import create_db, load_dashboard_from_db, save_dashboard_to_db

from constants import nationalities

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DEV Dashboard",
    page_icon="logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration pour accéder à Google Sheets
def load_google_sheet(sheet_url):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("gsheetsessai-83f42d3fc4c0.json", scope)
    client = gspread.authorize(creds)

    retries = 3
    for i in range(retries):
        try:
            sheet = client.open_by_url(sheet_url).sheet1
            data = sheet.get_all_records()
            df = pd.DataFrame(data)

            df.columns = clean_column_names(df.columns)

            # Remplacer les chaînes vides par NaN
            df.replace('', pd.NA, inplace=True)

            # Supprimer les colonnes entièrement vides
            df.dropna(axis=1, how='all', inplace=True)

            return df
        except gspread.exceptions.APIError as e:
            st.warning(f"Tentative {i+1} échouée: {e}")
            if i < retries - 1:
                time.sleep(2)

    st.error("Impossible de charger les données après plusieurs tentatives.")
    return None

def load_multiple_sheets(sheet_urls):
    all_dfs = []
    for url in sheet_urls:
        url = url.strip()  # Nettoyer les espaces autour de l'URL
        if not url:
            continue  # Ignorer les URLs vides
        
        try:
            df = load_google_sheet(url)
            if df is not None:
                all_dfs.append(df)
        except Exception as e:
            st.warning(f"Erreur lors du chargement de la feuille à l'URL {url}: {e}")
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        return combined_df
    else:
        st.error("Aucune donnée à fusionner.")
        return None

def clean_column_names(columns):
    return [re.sub(r'[^\w\s]', '', col).strip().lower() for col in columns]


def get_column_embedding(column_name, tokenizer, model):
    inputs = tokenizer(column_name, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def find_similar_columns(columns1, columns2, tokenizer, model, threshold=0.8):
    similarities = []
    for col1 in columns1:
        embedding1 = get_column_embedding(col1, tokenizer, model)
        for col2 in columns2:
            embedding2 = get_column_embedding(col2, tokenizer, model)
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            if similarity > threshold:
                st.write(f"Colonnes similaires trouvées: {col1} et {col2} avec une similarité de {similarity}")
                similarities.append((col1, col2, similarity))
    return similarities

############################################################

def normalize_nationality(nationality, nationalities):
    matches = get_close_matches(nationality, nationalities, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    else:
        return nationality

############################################################

def make_choropleth(input_df, input_column, input_color_theme):
    if input_column in input_df.columns:
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

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def sentiment_analysis(texts):
    tokenizer, model = load_model_and_tokenizer()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    sentiments = ['Très négatif', 'Négatif', 'Neutre', 'Positif', 'Très positif']

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.last_hidden_state[:, 0, :], dim=-1)
    predicted_classes = torch.argmax(probabilities, dim=-1)

    sentiment_results = [sentiments[class_idx] for class_idx in predicted_classes]

    return sentiment_results, probabilities

def create_combined_sentiment_graph(df, sentiment_column, second_column):
    sentiment_results, probabilities = sentiment_analysis(df[sentiment_column].fillna('').tolist())
    df['Sentiment'] = sentiment_results

    combined_distribution = df.groupby(['Sentiment', second_column]).size().reset_index(name='Count')

    fig = px.bar(combined_distribution, x='Sentiment', y='Count', color=second_column, 
                 title=f"Répartition de {second_column} par Sentiment", barmode='group')

    st.plotly_chart(fig)

def create_graph(df, selected_column, graph_type, container, second_column=None):
    filtered_df = df[selected_column].replace('', pd.NA).dropna()

    with container:
        if graph_type == "Bar":
            st.bar_chart(filtered_df.value_counts())
        elif graph_type == "Line":
            st.line_chart(filtered_df.value_counts().sort_index())
        elif graph_type == "Area":
            st.area_chart(filtered_df.value_counts().sort_index())
        elif graph_type == "Pie":
            fig = px.pie(filtered_df, names=selected_column, title=f"Pie Chart of {selected_column}")
            st.plotly_chart(fig)
        elif graph_type == "Histogram":
            fig = px.histogram(filtered_df, x=selected_column, title=f"Histogram of {selected_column}")
            st.plotly_chart(fig)
        elif graph_type == "Map":
            fig = make_choropleth(df, selected_column, 'Viridis')
            st.plotly_chart(fig, use_container_width=True)
        elif graph_type == "Sentitment Analyser":
            if second_column:
                create_combined_sentiment_graph(df, selected_column, second_column)
            else:
                sentiment_results, probabilities = sentiment_analysis(df[selected_column].fillna('').tolist())
                df['Sentiment'] = sentiment_results

def main():
    st.title("DEV Dashboard App")

    if "dashboard" not in st.session_state:
        st.session_state.dashboard = []

    if "combined_df" not in st.session_state:
        st.session_state.combined_df = None  # Initialiser avec None

    st.sidebar.header("Ajouter des Feuilles Google")

    num_links = st.sidebar.number_input("Nombre de liens à ajouter", min_value=1, max_value=10, value=1)
    sheet_urls = [st.sidebar.text_input(f"URL de la feuille {i+1}") for i in range(num_links)]

    if st.sidebar.button("Charger et Combiner les Données"):
        combined_df = load_multiple_sheets(sheet_urls)
        if combined_df is not None:
            st.session_state.combined_df = combined_df  # Stocker le DataFrame dans st.session_state
        else:
            st.error("Aucune donnée combinée disponible pour l'affichage.")
            return

    if st.session_state.combined_df is not None:
        combined_df = st.session_state.combined_df  # Récupérer le DataFrame de st.session_state
        st.write("DataFrame combiné:")
        st.write(combined_df)

        columns = combined_df.columns.tolist()
        
        if len(columns) == 0:
            st.error("Aucune colonne disponible pour l'affichage.")
            return
        
        graph_type = st.sidebar.selectbox("Choisissez un type de graphique", ["Bar", "Line", "Area", "Pie", "Histogram", "Map", "Sentitment Analyser"])

        temp_container = st.empty()

        if graph_type:
            if graph_type == "Sentitment Analyser":
                sentiment_column = st.sidebar.selectbox("Choisissez une colonne pour l'analyse des sentiments", columns)
                second_column = st.sidebar.selectbox("Choisissez une deuxième colonne à visualiser", columns)
                if sentiment_column and second_column:
                    create_combined_sentiment_graph(combined_df, sentiment_column, second_column)

                    if st.sidebar.button("Ajouter au Tableau de Bord"):
                        st.session_state.dashboard.append((sentiment_column, graph_type, second_column))
                        st.success("Graphique ajouté au tableau de bord !")
                        temp_container.empty()
            else:
                selected_column = st.sidebar.selectbox("Choisissez une colonne à visualiser", columns)
                if selected_column:  # Vérifier si une colonne a été sélectionnée
                    create_graph(combined_df, selected_column, graph_type, temp_container)

                    if st.sidebar.button("Ajouter au Tableau de Bord"):
                        st.session_state.dashboard.append((selected_column, graph_type, None))
                        st.success("Graphique ajouté au tableau de bord !")
                        temp_container.empty()
                else:
                    st.error("Veuillez sélectionner une colonne pour créer un graphique.")

    st.header("Tableau de Bord Personnalisé")

    for index, (column, graph, second_column) in enumerate(st.session_state.dashboard):
        if index % 2 == 0:
            col1, col2 = st.columns(2)
        
        with (col1 if index % 2 == 0 else col2):
            st.subheader(f"{graph} Graphe pour : {column}")
            if st.session_state.combined_df is not None and column in st.session_state.combined_df.columns:
                create_graph(st.session_state.combined_df, column, graph, st.container(), second_column)
            else:
                st.error(f"Impossible de créer le graphique pour la colonne '{column}'.")

            if st.button("Supprimer", key=f"delete-{index}"):
                st.session_state.dashboard.pop(index)
                st.experimental_rerun()

    if st.sidebar.button("Sauvegarder le Tableau de Bord"):
        save_dashboard_to_db()
        st.success("Tableau de Bord sauvegardé dans la base de données !")

    if st.sidebar.button("Charger le Tableau de Bord"):
        load_dashboard_from_db()
        st.success("Tableau de Bord chargé depuis la base de données !")

if __name__ == "__main__":
    main()