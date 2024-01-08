import pandas as pd
import numpy as np

from functions.ETL import load_dfs

games, reviews, items = load_dfs(from_main=True)

# ----------
# QUERY ENDPOINTS for API

def PlayTimeGenre(genre:str):
    """Return year with the highest number of 
    hours played for the provided ``genre``"""

    # Merged dataframe with release years and playtime
    merged = (
        items[['item_id', 'playtime_forever']]
        .merge(games[['item_id', 'app_name', 'genres', 'release_year']], how='left')
    )

    # Mask to filter out genres
    mask = (
        merged['genres']
        .map(lambda genres: genre in genres, na_action='ignore')
        .fillna(False)
    )

    # Filtering using mask, grouping and aggregate sum
    year = (
        merged[mask]
        .groupby(by='release_year') # Group by year
        .sum(numeric_only=True) # Aggregated sum by year
        ['playtime_forever'] #Selecting playtime
        .idxmax() # Getting the index for the maximum 
                  # which is now the year after grouping
    )

    response = {f"Release year with highest playtime for '{genre}' genre": int(year)}
    
    return response

# ----------

def UserForGenre(genre: str):
    """Returns the user with the most hours played
    given the ``genre``."""

    # Merged DataFrame with user id, playtime
    # And genres for each item they've played
    merged = (
        items
        .merge(games[['item_id', 'genres', 'release_year']], how='left')
    )

    # Mask for filtering
    mask = (merged['genres']
        .map(lambda genres: genre in genres, na_action='ignore')
        .fillna(False)
    )

    # Filtering genres out, grouping by user and 
    # aggregating sum for playtime
    user = (
        merged[mask][['user_id', 'playtime_forever']]
        .groupby('user_id')
        .sum()
        # Sorting in descending order
        .sort_values(by='playtime_forever', ascending=False)
    ).iloc[0].name # Name of first record which is the user's id

    # Sum of hours played per year in a list
    years_played = (
        merged[mask][['user_id', 'release_year', 'playtime_forever']]
        .groupby(['user_id', 'release_year'])
        .sum()
        .loc[user]
    ) # This is a list with indexes as years and
      # values as the sum of hours played

    # Creating the response
    response = {
        f"User with most hours played for '{genre}'": user,
        "Playtime_year": [
            f"Year {int(idx)}: {years_played.loc[idx, 'playtime_forever']}" for idx in years_played.index
        ]
    }
    
    return response

# ----------

def UsersRecommend(year: int):
    """Top 3 of most recommended games for the 
    given ``year``."""

    # Merged DataFrame already filtered by 
    # recommendations and positive or neutral
    # reviews
    merged = (
        reviews.loc[(reviews['recommend'] == True) & (reviews['sentiment'] > 0)]
        .merge(games[['item_id', 'app_name', 'release_year']], how='left', on='item_id')
    )

    # Filtering by the year given
    masked = merged[merged['release_year'] == year]

    # Getting titles of games
    titles = (
        masked[['app_name', 'sentiment']] # Selecting 
        .groupby('app_name').sum() # Grouping by app_name
        .sort_values(by='sentiment', ascending=False) # Sorting in descending order
    )[:3].index # Getting the top 3 indexes (now the names)

    response = {
        f"Top {i+1}": titles[i] for i in range(3)
        }
    return response

# ----------

def UsersWorstDeveloper(year: int):
    """Top 3 developers with the least 
    recommended games for the given year.
    
    The criteria for selecting the top 3 
    is simply counting bad reviews for 
    each developer in an already-filtered
    DataFrame containing only negative reviews."""

    # Merged DataFrame already filtered by 
    # recommendations and negative reviews
    merged = (
    reviews.loc[(reviews['recommend'] == False) & (reviews['sentiment'] == 0)]
    .merge(games[['item_id', 'release_year', 'developer']], how='left', on='item_id')
    )

    # Filtering by the year given
    masked = merged[merged['release_year'] == year]

    # Getting titles
    titles = masked['developer'].value_counts()[:3].index

    # Creating json response
    response = {
        f"Top worst dev {i+1}": titles[i] for i in range(3)
        }

    return response

# ---------

def sentiment_analysis(dev: str):
    """Returns a dictionary containing
    the count for each review category.
    
    Negative: sentiment = 0 
    Neutral: sentiment = 1
    Positive: sentiment = 2
    """
    # Merged DataFrame
    merged = (
        reviews
        .merge(games[['item_id', 'release_year', 'developer']], how='left', on='item_id')
    )

    # Filtering developers
    masked = merged[merged['developer'] == dev]

    # Labels to assign
    labels = ['Negative', 'Neutral', 'Positive']
    # Counting and not sorting as indexes need to be ordered
    # to match labels
    revs_count = masked.value_counts('sentiment', sort=False)

    # Building the response json  usind dict comprehension
    response = {
        dev: [f"{label} = {value}" for (label, value) in zip(labels, revs_count)]
        }
    return response

# Preproces data and Fit Computer class
from functions.recomender import CosSimComputer
from functions.preprocessing import preprocess_games

# Applying preprocessing
df = preprocess_games(games)
# Instance computer feeding it with the processed dataset
computer = CosSimComputer(df)

def game_recommend(n_sim:int, to_id:int):
    # Getting the n_sim most similar to to_id
    similars_idx = computer.n_most_similar(n=n_sim, to_=to_id, indexes=True)
    
    # Getting items from the original dataset using indexes
    items = games.loc[similars_idx, ['app_name', 'genres', 'specs', 'release_year', 'price']]
    
    # Creating the json response
    response = items.to_dict(orient='records')

    return response