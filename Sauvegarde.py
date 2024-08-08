import sqlite3
import streamlit as st


def create_db():
    conn = sqlite3.connect('dashboard.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS dashboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            column_name TEXT,
            graph_type TEXT,
            second_column TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_db()

def save_dashboard_to_db():
    conn = sqlite3.connect('dashboard.db')
    c = conn.cursor()
    
    # Vider la table avant de sauvegarder les nouvelles données
    c.execute('DELETE FROM dashboard')
    
    for column, graph_type, second_column in st.session_state.dashboard:
        c.execute('''
            INSERT INTO dashboard (column_name, graph_type, second_column)
            VALUES (?, ?, ?)
        ''', (column, graph_type, second_column))
    
    conn.commit()
    conn.close()

def load_dashboard_from_db():
    conn = sqlite3.connect('dashboard.db')
    c = conn.cursor()
    c.execute('SELECT column_name, graph_type, second_column FROM dashboard')
    
    st.session_state.dashboard = c.fetchall()
    conn.close()
