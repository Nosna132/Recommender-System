import streamlit as st
import pandas as pd
from joblib import load

# Load the data
@st.cache
def load_data():
    return pd.read_csv("tmdb_5000_movies.csv")

tmdb_data = load_data()

# Load the collaborative filtering model
collaborative_model = load('movie_recommender_model_collaborative.joblib')

# Load the content-based filtering model
content_based_model = load('movie_recommender_model_content-based.joblib')

# Title of the web app
st.title("Movie Recommender System")

# Sidebar for user input
movie_title = st.sidebar.selectbox("Select a Movie:", tmdb_data['title'])

# Button to trigger recommendation
if st.sidebar.button("Recommend"):
    # Collaborative Filtering
    st.subheader("Collaborative Filtering Recommendations")
    similar_movies_collaborative = collaborative_model[movie_title]
    if similar_movies_collaborative:
        for i, movie_idx in enumerate(similar_movies_collaborative):
            st.write(f"{i+1}. {tmdb_data.iloc[int(movie_idx)]['title']}")

    # Content-based Filtering
    st.subheader("Content-based Filtering Recommendations")
    similar_movies_content_based = content_based_model[movie_title]
    if similar_movies_content_based:
        for i, movie_idx in enumerate(similar_movies_content_based):
            st.write(f"{i+1}. {tmdb_data.iloc[int(movie_idx)]['title']}")
