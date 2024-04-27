import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df

# Load Movies Metadata
tmdb_data = pd.read_csv('tmdb_5000_movies.csv', low_memory=False)

# Data Preparation
# Calculate mean rating across all movies
C = tmdb_data['vote_average'].mean()

# Calculate the minimum number of votes required to be in the top percentile
m = tmdb_data['vote_count'].quantile(0.90)

# Filter out qualified movies
q_movies = tmdb_data.copy().loc[tmdb_data['vote_count'] >= m]

# Define numeric columns for collaborative filtering
numeric_columns = ['budget', 'popularity', 'vote_average', 'vote_count']
numeric_data = q_movies[numeric_columns].fillna(0)  # Fill missing values with 0

# Collaborative Filtering
def collaborative_filtering(movie_title):
    similarity_matrix = cosine_similarity(numeric_data)
    movie_index = tmdb_data[tmdb_data['title'] == movie_title].index[0]
    similar_movies = list(enumerate(similarity_matrix[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    return sorted_similar_movies[1:11]  # Return top 10 similar movies excluding itself

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

# Main Execution
# Enter Movie Title
movie_title = input("Enter the title of the movie: ")

# Find closest match to user input
closest_match = find_closest_match(movie_title)

# Check if a close match is found
if closest_match:
    print("Closest match found:", closest_match)
    # Collaborative Filtering
    collab_filtering_result = collaborative_filtering(closest_match)
    print("\nTop 10 movies similar to", closest_match, "based on Collaborative Filtering:")
    for movie in collab_filtering_result:
        print("- Movie:", tmdb_data.iloc[movie[0]]['title'])
        print("  Similarity Score:", movie[1])
else:
    print("There's no movie such as", movie_title, "Please enter another title")
