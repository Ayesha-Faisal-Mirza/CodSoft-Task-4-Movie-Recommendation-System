import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Load the dataset
data = pd.read_csv("tmdb_5000_movies.csv")

# Preprocess the data
data['genres'] = data['genres'].fillna('[]').apply(eval)
data['genres_str'] = data['genres'].apply(lambda x: ' '.join([i['name'] for i in x]))

# Initialize the Flask app
app = Flask(__name__)

# Create a CountVectorizer for genres
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(data['genres_str'])

# Calculate cosine similarity between genres
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Define the recommendation function
def recommend_movies_by_genre(genre):
    idx = data[data['genres_str'].str.contains(genre)].index
    if not idx.empty:
        movie_indices = idx.tolist()
        sim_scores = cosine_sim[movie_indices].mean(axis=0)
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return data['title'].iloc[movie_indices].tolist()
    else:
        return []

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        genre = request.form["genre"]
        recommendations = recommend_movies_by_genre(genre)
        return render_template("index.html", recommendations=recommendations)
    
    return render_template("index.html", recommendations=[])

if __name__ == "__main__":
    app.run(debug=True)
