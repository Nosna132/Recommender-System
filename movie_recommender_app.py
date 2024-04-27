import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib

# Set page config to wide mode and dark theme
st.set_page_config(layout="wide", page_title="Movie Recommender System", page_icon="🎬")

# Load data
tmdb_data = pd.read_csv('tmdb_5000_movies.csv')

# Function to handle errors and variations in user input
def find_closest_match(user_input):
    # List of movie titles from the dataset
    movie_titles = tmdb_data['title'].tolist()
    
    # Find closest match using difflib's get_close_matches function
    closest_matches = difflib.get_close_matches(user_input, movie_titles, n=1, cutoff=0.6)
    
    if closest_matches:
        # Return the closest match
        return closest_matches[0]
    else:
        # Return None if no close match found
        return None

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

# Main Streamlit app
st.title("Movie Recommender System")

# Sidebar
st.sidebar.title("Filters")
filter_choice = st.sidebar.radio("Select Filter", ("Collaborative Filtering",))

# Input for movie title
movie_title = st.text_input("Enter the title of the movie:", "")

# Button to trigger recommendation
if st.button("Recommend"):
    closest_match = find_closest_match(movie_title)

    if closest_match:
        st.write("Closest match found:", closest_match)
        
        if filter_choice == "Collaborative Filtering":
            collab_filtering_result = collaborative_filtering(closest_match)
            st.subheader("Top 10 movies similar to {} based on Collaborative Filtering:".format(closest_match))
            table_data = []
            for movie in collab_filtering_result:
                table_data.append([tmdb_data.iloc[movie[0]]['title'], movie[1]])
            st.table(pd.DataFrame(table_data, columns=["Movie", "Similarity Score"]))
    else:
        st.write("There's no movie such as", movie_title, "Please enter another title")
