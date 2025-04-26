from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Load minimal data
movies = pd.read_csv("data/movies.csv", usecols=["movieId", "title", "genres"], dtype={"movieId": "int32"})
ratings = pd.read_csv("data/ratings.csv", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
mapping = pd.read_csv("data/mapping.csv", dtype={"movieId": "int32", "tmdbId": "int32"})
movies["clean_title"] = movies["title"].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x))
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
tmdb_cache = json.load(open("data/tmdb_cache.json")) if os.path.exists("data/tmdb_cache.json") else {}

if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key":
    raise ValueError("TMDB_API_KEY is not set or invalid. Please set it in .env")

class RecommendationRequest(BaseModel):
    prompt: str | None = None
    watchlist: list[int] | None = None
    watched: list[int] | None = None
    user_id: str | None = None

security = HTTPBearer()

def search(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    return movies.iloc[indices].iloc[::-1]

def find_similar_movies(movie_id):
    # Process ratings in chunks
    chunk_size = 1000000
    similar_users = set()
    for chunk in pd.read_csv("data/ratings.csv", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}, chunksize=chunk_size):
        similar_users.update(chunk[(chunk["movieId"] == movie_id) & (chunk["rating"] > 4)]["userId"].unique())
    similar_users = list(similar_users)
    
    similar_user_recs = pd.Series(dtype="float32")
    for chunk in pd.read_csv("data/ratings.csv", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}, chunksize=chunk_size):
        chunk_recs = chunk[(chunk["userId"].isin(similar_users)) & (chunk["rating"] > 4)]["movieId"]
        similar_user_recs = pd.concat([similar_user_recs, chunk_recs])
    
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    
    all_users = set()
    for chunk in pd.read_csv("data/ratings.csv", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}, chunksize=chunk_size):
        all_users.update(chunk[(chunk["movieId"].isin(similar_user_recs.index)) & (chunk["rating"] > 4)]["userId"].unique())
    
    all_user_recs = pd.Series(dtype="float32")
    for chunk in pd.read_csv("data/ratings.csv", usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}, chunksize=chunk_size):
        chunk_recs = chunk[(chunk["userId"].isin(all_users)) & (chunk["rating"] > 4)]["movieId"]
        all_user_recs = pd.concat([all_user_recs, chunk_recs])
    
    all_user_recs = all_user_recs.value_counts() / len(all_users)
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

def get_tmdb_metadata(tmdb_id):
    if str(tmdb_id) in tmdb_cache:
        return tmdb_cache[str(tmdb_id)]
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        metadata = {
            "id": data["id"],
            "title": data["title"],
            "poster_path": data["poster_path"],
            "vote_average": data["vote_average"],
            "release_date": data["release_date"],
            "genres": [g["name"] for g in data["genres"]]
        }
        tmdb_cache[str(tmdb_id)] = metadata
        json.dump(tmdb_cache, open("data/tmdb_cache.json", "w"))
        return metadata
    except requests.RequestException as e:
        print(f"Error fetching TMDB metadata for ID {tmdb_id}: {e}")
        return None

def recommend_movies(prompt=None, watchlist=None, watched=None, limit=7):
    watchlist = watchlist or []
    watched = watched or []
    watchlist_ml = mapping[mapping["tmdbId"].isin(watchlist)]["movieId"].tolist()
    watched_ml = mapping[mapping["tmdbId"].isin(watched)]["movieId"].tolist()
    recs = pd.DataFrame()
    superhero_keywords = ["superhero", "marvel", "dc", "avengers", "spider-man", "batman", "superman"]
    if prompt:
        search_results = search(prompt)
        if not search_results.empty:
            movie_id = search_results.iloc[0]["movieId"]
            prompt_recs = find_similar_movies(movie_id)
            recs = pd.concat([recs, prompt_recs])
        if "star wars" in prompt.lower():
            star_wars_id = mapping[mapping["tmdbId"] == 11]["movieId"].iloc[0]
            recs = pd.concat([recs, find_similar_movies(star_wars_id)])
    superhero_tmdb_ids = [299536, 912649]
    for movie_id in watchlist_ml + watched_ml:
        movie_recs = find_similar_movies(movie_id)
        recs = pd.concat([recs, movie_recs])
    if not recs.empty:
        recs = recs.groupby("movieId").agg({"score": "mean", "title": "first", "genres": "first"})
        recs = recs.reset_index().sort_values("score", ascending=False)
        exclude_ids = set(watchlist_ml + watched_ml)
        recs = recs[~recs["movieId"].isin(exclude_ids)]
        if prompt and any(kw in prompt.lower() for kw in superhero_keywords) or any(tmdb_id in watchlist + watched for tmdb_id in superhero_tmdb_ids):
            recs.loc[recs["genres"].str.contains("Action|Sci-Fi|Adventure"), "score"] *= 1.5
        recs = recs.head(limit)
    else:
        recs = movies[movies["genres"].str.contains("Action|Sci-Fi")]
        recs = recs.merge(ratings.groupby("movieId")["rating"].mean(), on="movieId")
        recs = recs.sort_values("rating", ascending=False).head(limit)
        recs["score"] = recs["rating"]
    results = []
    for _, row in recs.iterrows():
        movie_id = row["movieId"]
        tmdb_id = mapping[mapping["movieId"] == movie_id]["tmdbId"].iloc[0] if movie_id in mapping["movieId"].values else None
        if tmdb_id:
            metadata = get_tmdb_metadata(tmdb_id)
            if metadata:
                results.append({
                    "id": metadata["id"],
                    "title": metadata["title"],
                    "poster_path": metadata["poster_path"],
                    "vote_average": metadata["vote_average"],
                    "release_date": metadata["release_date"],
                    "genres": metadata["genres"]
                })
        if len(results) < limit and not tmdb_id:
            results.append({
                "id": movie_id,
                "title": row["title"],
                "poster_path": None,
                "vote_average": row.get("rating", 0),
                "release_date": "",
                "genres": row["genres"].split("|")
            })
    return results[:limit]

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/recommend")
async def recommend(request: RecommendationRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        # Add Firebase token verification here if needed
        print(f"Processing request: prompt={request.prompt}, watchlist={request.watchlist}, watched={request.watched}, user_id={request.user_id}")
        results = recommend_movies(request.prompt, request.watchlist, request.watched)
        print(f"Recommendations: {[r['title'] for r in results]}")
        return {"recommendations": results}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))