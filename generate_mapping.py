import pandas as pd
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key":
    raise ValueError("TMDB_API_KEY is not set or invalid. Please set it in .env")

def get_tmdb_id(title, year):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["results"][0]["id"] if data["results"] else None
    except requests.RequestException as e:
        print(f"Error searching TMDB for {title} ({year}): {e}")
        return None

# Load data
links = pd.read_csv("data/links.csv")
movies = pd.read_csv("data/movies.csv")
mapping = links[["movieId", "tmdbId"]].dropna().astype({"tmdbId": int})

# Find missing tmdbId
missing = movies[~movies["movieId"].isin(mapping["movieId"])][["movieId", "title"]]
for _, row in missing.iterrows():
    title = row["title"]
    movie_id = row["movieId"]
    year_match = re.search(r"\((\d{4})\)", title)
    year = year_match.group(1) if year_match else None
    if year:
        clean_title = re.sub(r"\s*\(\d{4}\)", "", title).strip()
        tmdb_id = get_tmdb_id(clean_title, year)
        if tmdb_id:
            mapping = pd.concat([mapping, pd.DataFrame({"movieId": [movie_id], "tmdbId": [tmdb_id]})], ignore_index=True)
            print(f"Mapped {clean_title} ({year}) to tmdbId {tmdb_id}")

# Save mapping
mapping.to_csv("data/mapping.csv", index=False)
print(f"Saved mapping.csv with {len(mapping)} entries")