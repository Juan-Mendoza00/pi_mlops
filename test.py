import functions.queries as queries
import functions.ETL

def game_recommend(n, to):
    return queries.game_recommend(n_sim=n, to_id=to)

if __name__ == '__main__':
    n = input('n similar: ')
    to = input('id')
    n = int(n)
    to = int(to)

    print(game_recommend(n=n, to=to))