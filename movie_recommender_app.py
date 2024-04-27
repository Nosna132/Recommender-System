import streamlit as st
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load

# Load the data
@st.cache
def load_data():
    return pd.read_csv("tmdb_5000_movies.csv")

tmdb_data = load_data()

# Load collaborative filtering model
@st.cache(allow_output_mutation=True)
def load_collaborative_model():
    return load("movie_recommender_model_collaborative.joblib")

collaborative_model = load_collaborative_model()

# Load content-based filtering model
@st.cache(allow_output_mutation=True)
def load_content_based_model():
    return load("movie_recommender_model_content-based.joblib")

content_based_model = load_content_based_model()

# Function to handle errors and variations in user input
def find_closest_match(user_input):
    # List of movie titles from the dataset
    movie_titles = tmdb_data['title'].tolist()
    
    # Find closest match using difflib's get_close_matches function
    closest_matches = difflib.get_close_matches(user_input, movie_titles, n=1, cutoff=0.6)
    
    if closest_matches:
        # Return the index of the closest match
        return movie_titles.index(closest_matches[0])
    else:
        # Return None if no close match found
        return None

# Collaborative Filtering
def collaborative_filtering(movie_index):
    return collaborative_model[movie_index]

# Content-based Filtering
def content_based_filtering(movie_index):
    return content_based_model[movie_index]

# Title of the web app
st.title("Movie Recommender System")

# Sidebar for user input
movie_title = st.sidebar.selectbox("Select a Movie:", tmdb_data['title'])

# Recommend button
if st.sidebar.button("Recommend"):
    closest_match_index = find_closest_match(movie_title)
    if closest_match_index is not None:
        st.subheader("Collaborative Filtering Recommendations")
        collaborative_recommendations = collaborative_filtering(closest_match_index)
        for i, movie_idx in enumerate(collaborative_recommendations):
            st.write(f"{i+1}. {tmdb_data.iloc[movie_idx]['title']}")

        st.subheader("Content-based Filtering Recommendations")
        content_based_recommendations = content_based_filtering(closest_match_index)
        for i, movie_idx in enumerate(content_based_recommendations):
            st.write(f"{i+1}. {tmdb_data.iloc[movie_idx]['title']}")
    else:
        st.error("No close match found for the selected movie. Please try another movie title.")
