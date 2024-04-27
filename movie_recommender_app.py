import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load

# Load data
tmdb_data = pd.read_csv('tmdb_5000_movies.csv')

# Load movie recommender models
collab_model = load("movie_recommender_model_collaborative.joblib")
content_based_model = load("movie_recommender_model_content-based.joblib")

# Function to handle errors and variations in user input
def find_closest_match(user_input):
    movie_titles = tmdb_data['title'].tolist()
    closest_matches = df.get_close_matches(user_input, movie_titles, n=1, cutoff=0.6)
    if closest_matches:
        return closest_matches[0]
    else:
        return None

# Function to recommend movies based on collaborative filtering
def collaborative_filtering(movie_title):
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(numeric_data)
    # Find index of the input movie
    movie_index = tmdb_data[tmdb_data['title'] == movie_title].index[0]
    # Retrieve similar movies with their similarity scores
    similar_movies = list(enumerate(similarity_matrix[movie_index]))
    # Sort similar movies by similarity score in descending order
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    # Extract top similar movies excluding the input movie itself
    top_similar_movies = sorted_similar_movies[1:11]
    return top_similar_movies

# Function to recommend movies based on content-based filtering
def content_based_filtering(movie_title):
    cv = CountVectorizer()
    genres_matrix = cv.fit_transform(tmdb_data['genres'])
    similarity_scores = cosine_similarity(genres_matrix, genres_matrix)
    movie_index = tmdb_data[tmdb_data['title'] == movie_title].index[0]
    similar_movies = list(enumerate(similarity_scores[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    return sorted_similar_movies[1:11]

# Streamlit UI
st.title("Movie Recommender System")

# Input field for movie title
movie_title = st.text_input("Enter the title of the movie:")

# Recommend button
if st.button("Recommend"):
    closest_match = find_closest_match(movie_title)
    if closest_match:
        st.write("Closest match found:", closest_match)
        # Collaborative Filtering
        collab_filtering_result = collaborative_filtering(closest_match)
        st.write("\nTop 10 movies similar to", closest_match, "based on Collaborative Filtering:")
        for movie in collab_filtering_result:
            st.write("- Movie:", tmdb_data.iloc[movie[0]]['title'])
            st.write("  Similarity Score:", movie[1])
    else:
        st.write("There's no movie such as", movie_title, "Please enter another title")

    if closest_match:
        st.write("\nTop 10 movies similar to", closest_match, "based on Content-Based Filtering:")
        content_based_filtering_result = content_based_filtering(closest_match)
        for movie in content_based_filtering_result:
            st.write("- Movie:", tmdb_data.iloc[movie[0]]['title'])
            st.write("  Similarity Score:", movie[1])
