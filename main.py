from fastapi import FastAPI, HTTPException
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

# Load environment variables
load_dotenv()

app = FastAPI()

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
mapping = pd.read_csv("data/mapping.csv")
movies["clean_title"] = movies["title"].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x))
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
tmdb_cache = json.load(open("data/tmdb_cache.json")) if os.path.exists("data/tmdb_cache.json") else {}

# Validate TMDB API key
if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key":
    raise ValueError("TMDB_API_KEY is not set or invalid. Please set it in .env")

class RecommendationRequest(BaseModel):
    prompt: str | None = None
    watchlist: list[int] | None = None
    watched: list[int] | None = None
    user_id: str | None = None

def search(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    return movies.iloc[indices].iloc[::-1]

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
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

    # Convert TMDB IDs to MovieLens IDs
    watchlist_ml = mapping[mapping["tmdbId"].isin(watchlist)]["movieId"].tolist()
    watched_ml = mapping[mapping["tmdbId"].isin(watched)]["movieId"].tolist()

    # Initialize recommendations
    recs = pd.DataFrame()

    # Handle prompt
    superhero_keywords = ["superhero", "marvel", "dc", "avengers", "spider-man", "batman", "superman"]
    if prompt:
        search_results = search(prompt)
        if not search_results.empty:
            movie_id = search_results.iloc[0]["movieId"]
            prompt_recs = find_similar_movies(movie_id)
            recs = pd.concat([recs, prompt_recs])
        if "star wars" in prompt.lower():
            star_wars_id = mapping[mapping["tmdbId"] == 11]["movieId"].iloc[0]  # Star Wars (1977)
            recs = pd.concat([recs, find_similar_movies(star_wars_id)])

    # Handle watchlist and watched
    superhero_tmdb_ids = [299536, 912649]  # Avengers, Venom
    for movie_id in watchlist_ml + watched_ml:
        movie_recs = find_similar_movies(movie_id)
        recs = pd.concat([recs, movie_recs])

    # Aggregate and filter
    if not recs.empty:
        recs = recs.groupby("movieId").agg({"score": "mean", "title": "first", "genres": "first"})
        recs = recs.reset_index().sort_values("score", ascending=False)
        # Exclude watchlist and watched
        exclude_ids = set(watchlist_ml + watched_ml)
        recs = recs[~recs["movieId"].isin(exclude_ids)]
        # Boost superhero movies
        if prompt and any(kw in prompt.lower() for kw in superhero_keywords) or any(tmdb_id in watchlist + watched for tmdb_id in superhero_tmdb_ids):
            recs.loc[recs["genres"].str.contains("Action|Sci-Fi|Adventure"), "score"] *= 1.5
        recs = recs.head(limit)
    else:
        # Fallback: Top-rated action/sci-fi
        recs = movies[movies["genres"].str.contains("Action|Sci-Fi")]
        recs = recs.merge(ratings.groupby("movieId")["rating"].mean(), on="movieId")
        recs = recs.sort_values("rating", ascending=False).head(limit)
        recs["score"] = recs["rating"]

    # Convert to TMDB format
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

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        print(f"Processing request: prompt={request.prompt}, watchlist={request.watchlist}, watched={request.watched}, user_id={request.user_id}")
        results = recommend_movies(request.prompt, request.watchlist, request.watched)
        print(f"Recommendations: {[r['title'] for r in results]}")
        return {"recommendations": results}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))