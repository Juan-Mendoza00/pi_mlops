from typing import Annotated

from fastapi import FastAPI, Query

# Importing queries script where all enpoint functions are stored
import functions.queries as queries


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome. To acces the query engine please ad '/docs' at the end of the url."}

@app.get("/PlayTime")
def PlayTimeGenre(
    genre: Annotated[
        str,
        Query(
            description = "Genre to filter out and query."
        )
    ]
):
    
    """Return year with the highest number 
    of hours played for the provided genre"""
    
    response = queries.PlayTimeGenre(genre)
    return response

@app.get("/Users")
def UserForGenre(
    genre: Annotated[
        str,
        Query(
            description = "Genre to filter out and query."
        )
    ]
):
    """Return the user with the most hours 
    played given the genre."""

    response = queries.UserForGenre(genre)
    return response

@app.get("/UserRec")
def UsersRecommend(
    year: Annotated[
        int,
        Query(
            description = "Year to filter out and query."
        )
    ]
):
    """Return the Top 3 most recommended games 
    during the given year."""

    response = queries.UsersRecommend(year)
    return response

@app.get("/WorstDev")
def UsersWorstDeveloper(
    year: Annotated[
        int,
        Query(
            description = "Year to filter out and query."
        )
    ]
):
    """Top 3 developers with the least recommended 
    games for the given year."""

    response = queries.UsersWorstDeveloper(year)
    return response

@app.get("/Sentiment")
def sentiment_analysis(
    dev: Annotated[
        str,
        Query(
            description = "Developer name"
        )
    ]
):
    """Type the name of some Developer company and 
    it will return the total Positive, Negative and 
    Neutral comments.
    """

    response = queries.sentiment_analysis(dev)
    return response

# Rec Sys
@app.get('/RecSys')
def game_recommend(
    item_id: Annotated[
        int,
        Query(
            description = "Unique id for a game"
        )
    ] = ...,
    n: Annotated[
        int,
        Query(
            description = "The `n` most similar"
        )
    ] = 5
):
    """The n most similar games to the item passed"""

    response = queries.game_recommend(n_sim=n, to_id=item_id)
    return response