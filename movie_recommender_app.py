from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load

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

# Example
def recommend_movies(movie_title, filter_type):
    closest_match = find_closest_match(movie_title)
    
    if closest_match:
        print("Closest match found:", closest_match)
        if filter_type == "collaborative":
            model = collaborative_filtering(closest_match)
        elif filter_type == "content-based":
            model = content_based_filtering(closest_match)
        
        # Save the model
        dump(model, 'movie_recommender_model.joblib')
        print("Model saved successfully as 'movie_recommender_model.joblib'")
    else:
        print("There's no movie such as", movie_title, "Please enter another title")

# Example usage
movie_title = input("Enter the title of the movie: ")
filter_type = input("Enter the type of filtering (collaborative/content-based): ")

recommend_movies(movie_title, filter_type)
