import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df
from joblib import load

# Function to handle errors and variations in user input
def find_closest_match(user_input, movie_titles):
    closest_matches = df.get_close_matches(user_input, movie_titles, n=1, cutoff=0.6)
    
    if closest_matches:
        return closest_matches[0]
    else:
        return None

# Function to perform collaborative filtering
def collaborative_filtering(movie_title, model, tmdb_data):
    movie_index = tmdb_data.index[tmdb_data['title'] == movie_title].tolist()
    if movie_index:
        similar_movies = model[movie_index[0]]
        return similar_movies
    else:
        return []

# Function to perform content-based filtering
def content_based_filtering(movie_title, model, tmdb_data):
    movie_index = tmdb_data.index[tmdb_data['title'] == movie_title].tolist()
    if movie_index:
        similar_movies = model[movie_index[0]]
        return similar_movies
    else:
        return []

# Streamlit app
def main():
    st.title("Movie Recommender System")

    # Load data
    try:
        tmdb_data = pd.read_csv('tmdb_5000_movies.csv')
        movie_titles = tmdb_data['title'].tolist()
    except FileNotFoundError:
        st.error("File 'tmdb_5000_movies.csv' not found.")
        return
    except pd.errors.EmptyDataError:
        st.error("File 'tmdb_5000_movies.csv' is empty.")
        return

    # User input for movie title
    movie_title = st.text_input("Enter the title of the movie:")

    # User input for filtering method
    filtering_method = st.selectbox("Select Filtering Method:", ["Collaborative Filtering", "Content-Based Filtering"])

    # Find closest match to user input
    closest_match = find_closest_match(movie_title, movie_titles)

    # Check if a close match is found
    if closest_match:
        st.write("Closest match found:", closest_match)
        
        # Perform recommendation based on selected filtering method
        if filtering_method == "Collaborative Filtering":
            collab_filtering_result = collaborative_filtering(closest_match)
            if collab_filtering_result:
                st.write("\nTop 10 movies similar to", closest_match, "based on Collaborative Filtering:")
                for movie_id, similarity_score in collab_filtering_result:
                    st.write("- Movie:", tmdb_data.iloc[movie_id]['title'])
                    st.write("  Similarity Score:", similarity_score)
            else:
                st.write("No recommendations found.")
        elif filtering_method == "Content-Based Filtering":
            content_based_filtering_result = content_based_filtering(closest_match)
            if content_based_filtering_result:
                st.write("\nTop 10 movies similar to", closest_match, "based on Content-Based Filtering:")
                for movie_id, similarity_score in content_based_filtering_result:
                    st.write("- Movie:", tmdb_data.iloc[movie_id]['title'])
                    st.write("  Similarity Score:", similarity_score)
            else:
                st.write("No recommendations found.")
    else:
        st.write("No close match found for:", movie_title)

if __name__ == "__main__":
    main()
