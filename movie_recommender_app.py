import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
@st.cache
def load_data():
    return pd.read_csv("tmdb_5000_movies.csv")

tmdb_data = load_data()

# Collaborative Filtering
def collaborative_filtering(movie_title, top_n=10):
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the movie that matches the title
    idx = indices[indices == movie_title].index[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top-n most similar movies
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    return movie_indices

# Content-Based Filtering
def content_based_filtering(movie_title, top_n=10):
    # Initialize the count vectorizer
    count = CountVectorizer(stop_words='english')

    # Fit and transform the data
    count_matrix = count.fit_transform(tmdb_data['overview'])

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Get the index of the movie that matches the title
    idx = indices[indices == movie_title].index[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top-n most similar movies
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    return movie_indices

# Title of the web app
st.title("Movie Recommender System")

# Sidebar for user input
movie_title = st.sidebar.selectbox("Select a Movie:", tmdb_data['title'])

# Collaborative Filtering
st.subheader("Collaborative Filtering Recommendations")
similar_movies_collaborative = collaborative_filtering(movie_title)
if similar_movies_collaborative:
    for i, movie_idx in enumerate(similar_movies_collaborative):
        st.write(f"{i+1}. {tmdb_data.iloc[int(movie_idx)]['title']}")

# Content-based Filtering
st.subheader("Content-based Filtering Recommendations")
similar_movies_content_based = content_based_filtering(movie_title)
if similar_movies_content_based:
    for i, movie_idx in enumerate(similar_movies_content_based):
        st.write(f"{i+1}. {tmdb_data.iloc[int(movie_idx)]['title']}")
