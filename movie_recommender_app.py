import streamlit as st
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df

# Load data
tmdb_data = pd.read_csv('tmdb_5000_movies.csv')

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
    model = load('movie_recommender_model_collaborative.joblib')
    movie_index = tmdb_data.index[tmdb_data['title'] == movie_title].tolist()
    if movie_index:
        similar_movies = model[movie_index[0]]
        return similar_movies
    else:
        return []

# Function to perform content-based filtering
def content_based_filtering(movie_title):
    model = load('movie_recommender_model_content-based.joblib')
    movie_index = tmdb_data.index[tmdb_data['title'] == movie_title].tolist()
    if movie_index:
        similar_movies = model[movie_index[0]]
        return similar_movies
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
            st.write("\nTop 10 movies similar to", closest_match, "based on Collaborative Filtering:")
            for movie_id, similarity_score in collab_filtering_result:
                st.write("- Movie:", tmdb_data.iloc[movie_id]['title'])
                st.write("  Similarity Score:", similarity_score)
        elif filtering_method == "Content-Based Filtering":
            content_based_filtering_result = content_based_filtering(closest_match)
            st.write("\nTop 10 movies similar to", closest_match, "based on Content-Based Filtering:")
            for movie_id, similarity_score in content_based_filtering_result:
                st.write("- Movie:", tmdb_data.iloc[movie_id]['title'])
                st.write("  Similarity Score:", similarity_score)
    else:
        st.write("No close match found for:", movie_title)
