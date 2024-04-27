import streamlit as st
import pandas as pd
from joblib import load
import difflib as df

# Load data
try:
    tmdb_data = pd.read_csv('tmdb_5000_movies.csv')
except pd.errors.EmptyDataError:
    st.error("Error: No columns to parse from file. Please check if the dataset file is valid.")

# Function to handle errors and variations in user input
def find_closest_match(user_input):
    # List of movie titles from the dataset
    movie_titles = tmdb_data['title'].tolist()
    
    # Find closest match using difflib's get_close_matches function
    closest_matches = df.get_close_matches(user_input, movie_titles, n=1, cutoff=0.6)
    
    if closest_matches:
        # Return the closest match
        return closest_matches[0]
    else:
        # Return None if no close match found
        return None

# Function to perform collaborative filtering
def collaborative_filtering(movie_title):
    try:
        model = load('movie_recommender_model_collaborative.joblib')
    except FileNotFoundError:
        st.error("Error: Collaborative filtering model not found.")
        return []
    
    movie_index = tmdb_data.index[tmdb_data['title'] == movie_title].tolist()
    if movie_index:
        movie_index = movie_index[0]
        if movie_index < len(model):
            similar_movies = model[movie_index]
            return similar_movies
        else:
            st.error("Error: Movie index out of range for collaborative filtering.")
    else:
        return []

# Function to perform content-based filtering
def content_based_filtering(movie_title):
    try:
        model = load('movie_recommender_model_content-based.joblib')
    except FileNotFoundError:
        st.error("Error: Content-based filtering model not found.")
        return []
    
    movie_index = tmdb_data.index[tmdb_data['title'] == movie_title].tolist()
    if movie_index:
        movie_index = movie_index[0]
        if movie_index < len(model):
            similar_movies = model[movie_index]
            return similar_movies
        else:
            st.error("Error: Movie index out of range for content-based filtering.")
    else:
        return []

# Streamlit app
st.title("Movie Recommender System")

# User input for movie title
movie_title = st.text_input("Enter the title of the movie:")

# User input for filtering method
filtering_method = st.selectbox("Select Filtering Method:", ["Collaborative Filtering", "Content-Based Filtering"])

# "Recommend" button
if st.button("Recommend"):
    # Find closest match to user input
    closest_match = find_closest_match(movie_title)
    
    # Check if a close match is found
    if closest_match:
        st.write("Closest match found:", closest_match)
        
        # Perform recommendation based on selected filtering method
        if filtering_method == "Collaborative Filtering":
            collab_filtering_result = collaborative_filtering(closest_match)
            if collab_filtering_result:
                st.write("\nTop 10 movies similar to", closest_match, "based on Collaborative Filtering:")
                for similarity_score in collab_filtering_result:
                    st.write("- Movie:", tmdb_data.iloc[similarity_score]['title'])
                    st.write("  Similarity Score:", similarity_score)
            else:
                st.error("Error: Unable to perform collaborative filtering.")
        elif filtering_method == "Content-Based Filtering":
            content_based_filtering_result = content_based_filtering(closest_match)
            if content_based_filtering_result:
                st.write("\nTop 10 movies similar to", closest_match, "based on Content-Based Filtering:")
                for similarity_score in content_based_filtering_result:
                    st.write("- Movie:", tmdb_data.iloc[similarity_score]['title'])
                    st.write("  Similarity Score:", similarity_score)
            else:
                st.error("Error: Unable to perform content-based filtering.")
    else:
        st.error("Error: No close match found for:", movie_title)
