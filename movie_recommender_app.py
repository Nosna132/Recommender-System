import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load

# Load data
tmdb_data = pd.read_csv('tmdb_5000_movies.csv')

# Load collaborative filtering model
collab_model = load("movie_recommender_model_collaborative.joblib")

# Load content-based filtering model
content_based_model = load("movie_recommender_model_content-based.joblib")

# Collaborative Filtering
def collaborative_filtering(movie_title):
    # Calculate mean rating across all movies
    C = tmdb_data['vote_average'].mean()

    # Calculate the minimum number of votes required to be in the top percentile
    m = tmdb_data['vote_count'].quantile(0.90)

    # Filter out qualified movies
    q_movies = tmdb_data.copy().loc[tmdb_data['vote_count'] >= m]

    # Define numeric columns for collaborative filtering
    numeric_columns = ['budget', 'popularity', 'vote_average', 'vote_count']
    numeric_data = q_movies[numeric_columns].fillna(0)  # Fill missing values with 0
    
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
    
    # Return top similar movies
    return top_similar_movies

# Content-Based Filtering
def content_based_filtering(movie_title):
    # Initialize CountVectorizer
    cv = CountVectorizer()
    
    # Fit and transform genres into a matrix
    genres_matrix = cv.fit_transform(tmdb_data['genres'])
    
    # Compute cosine similarity between movies based on genres
    similarity_scores = cosine_similarity(genres_matrix)
    
    # Find index of the input movie
    movie_index = tmdb_data[tmdb_data['title'] == movie_title].index[0]
    
    # Retrieve similar movies with their similarity scores
    similar_movies = list(enumerate(similarity_scores[movie_index]))
    
    # Sort similar movies by similarity score in descending order
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    
    # Extract top similar movies excluding the input movie itself
    top_similar_movies = sorted_similar_movies[1:11]
    
    # Return top similar movies
    return top_similar_movies

# Main Streamlit app
st.set_page_config(layout="wide", page_title="Movie Recommender System", page_icon=":movie_camera:")

st.title("Movie Recommender System")

# Sidebar
filter_choice = st.sidebar.radio("Select Filter", ("Collaborative Filtering", "Content-Based Filtering"))

# Input for movie title
movie_title = st.text_input("Enter the title of the movie:")

# Button to trigger recommendation
if st.button("Recommend"):
    if filter_choice == "Collaborative Filtering":
        collab_filtering_result = collaborative_filtering(movie_title)
        st.subheader(f"Top 10 movies similar to {movie_title} based on Collaborative Filtering:")
        for i, movie in enumerate(collab_filtering_result, start=1):
            st.write(f"{i}. Movie: {tmdb_data.iloc[movie[0]]['title']}")
            st.write(f"   Similarity Score: {movie[1]}")

    elif filter_choice == "Content-Based Filtering":
        content_based_filtering_result = content_based_filtering(movie_title)
        st.subheader(f"Top 10 movies similar to {movie_title} based on Content-Based Filtering:")
        for i, movie in enumerate(content_based_filtering_result, start=1):
            st.write(f"{i}. Movie: {tmdb_data.iloc[movie[0]]['title']}")
            st.write(f"   Similarity Score: {movie[1]}")
