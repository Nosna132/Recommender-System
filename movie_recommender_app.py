import streamlit as st
import pandas as pd
from joblib import load

# Load the TMDB dataset
tmdb_data = pd.read_csv('tmdb_5000_movies.csv')

# Function to perform collaborative filtering
def collaborative_filtering(movie_title):
    try:
        model = load('movie_recommender_model_collaborative.joblib')
    except FileNotFoundError:
        st.error("Error: Collaborative filtering model not found.")
        return []
    
    movie_index = tmdb_data[tmdb_data['title'] == movie_title].index.tolist()
    if movie_index:
        movie_index = movie_index[0]
        similar_movies = model[movie_index]
        top_indices = similar_movies.argsort()[-10:][::-1]
        return top_indices
    else:
        st.error("Error: Movie not found.")
        return []

# Function to perform content-based filtering
def content_based_filtering(movie_title):
    try:
        model = load('movie_recommender_model_content-based.joblib')
    except FileNotFoundError:
        st.error("Error: Content-based filtering model not found.")
        return []
    
    movie_index = tmdb_data[tmdb_data['title'] == movie_title].index.tolist()
    if movie_index:
        movie_index = movie_index[0]
        similar_movies = model[movie_index]
        top_indices = similar_movies.argsort()[-10:][::-1]
        return top_indices
    else:
        st.error("Error: Movie not found.")
        return []

# Streamlit UI
st.title("Movie Recommender System")

# Sidebar
st.sidebar.title("Choose a Movie")

movie_title = st.sidebar.selectbox(
    'Select a movie:',
    tmdb_data['title'].tolist()
)

if st.sidebar.button('Show Recommendations'):
    # Collaborative Filtering
    st.subheader("Collaborative Filtering Recommendations")
    top_indices_collaborative = collaborative_filtering(movie_title)
    if top_indices_collaborative:
        recommendations_collaborative = tmdb_data.iloc[top_indices_collaborative]['title']
        st.write(recommendations_collaborative)
    
    # Content-based Filtering
    st.subheader("Content-based Filtering Recommendations")
    top_indices_content_based = content_based_filtering(movie_title)
    if top_indices_content_based:
        recommendations_content_based = tmdb_data.iloc[top_indices_content_based]['title']
        st.write(recommendations_content_based)
