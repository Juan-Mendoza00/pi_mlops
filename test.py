import functions.queries as queries
import functions.ETL

def UserForGenre(genre:str):
    return queries.UserForGenre(genre)

if __name__ == '__main__':
    genre = input('type genre: ')
    print(UserForGenre(genre))