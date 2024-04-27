import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import dump, load

# Load the data
@st.cache
def load_data():
    return pd.read_csv("tmdb_5000_movies.csv")

tmdb_data = load_data()

# Load the collaborative filtering model
collaborative_model = load('collaborative_filtering_model.joblib')

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

# Collaborative Filtering
def collaborative_filtering(movie_title):
    # Find index of the input movie
    movie_index = tmdb_data[tmdb_data['title'] == movie_title].index[0]
    
    # Retrieve similar movies from the model
    similar_movies = collaborative_model[movie_index]
    
    return similar_movies

# Content-Based Filtering
def content_based_filtering(movie_title):
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
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    return movie_indices

# Title of the web app
st.title("Movie Recommender System")

# Sidebar for user input
movie_title = st.sidebar.selectbox("Select a Movie:", tmdb_data['title'])

# Button to trigger recommendation
if st.sidebar.button("Recommend"):
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
