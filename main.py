from fastapi import FastAPI
import functions.queries as queries

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/PlayTime/{genre}")
def PlayTimeGenre(genre:str):
    return queries.PlayTimeGenre(genre)

@app.get("/Users")
def UserForGenre(genre:str):
    return queries.UserForGenre(genre)

@app.get("/UserRec")
def UsersRecommend(year:int):
    return queries.UsersRecommend(year)

@app.get("/WorstDev")
def UsersWorstDeveloper(year:int):
    return queries.UsersWorstDeveloper(year)

@app.get("/Sentiment")
def sentiment_analysis(dev:str):
    return queries.sentiment_analysis(dev)

# Rec Sys
