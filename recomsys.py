import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


movies = {
    'Title': ['Iron Man', 'Mr.Bean', 'Friends', 'Batman:The Dark knight rises'],
    'Genre': ['Action|Adventure', 'Drama|Romance', 'Comedy|Romance', 'Action|Comedy']
}


movies_df = pd.DataFrame(movies)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_df['Genre'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim, movies=movies_df):
    idx = movies[movies['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['Title'].iloc[movie_indices]

movie_title = input("Enter the movie title for recommendation: ")  
recommendations = get_recommendations(movie_title)
print(f"Recommended movies for '{movie_title}':")
print(recommendations)
