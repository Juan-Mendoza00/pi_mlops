import numpy as np 
import pandas as pd
import gzip 
import json

def gzip_json_file(
        path:str ='./',
        df=pd.DataFrame,
        subset:list = None
    ):
    '''Saving a dataFrame to a gzipped file in json format.

    The output file is a json-per-line file, compressed 
    with pd.DataFrame.to_json 'gzip' option. 
    
    That is:
        ``orient``: 'records'
        ``lines``: True
        ``compression``: 'gzip'
    
    ## Parametters:
    - path: File location and name.
    - df: pd.Dataframe to save.
    - subset:Optional. Subset of columns to be saved.'''

    if subset is not None:
        df = df[subset]

    # Creating the file
    df.to_json(
        path_or_buf=path,
        orient='records',
        lines=True,
        compression='gzip'    
    )

    print(f'File saved at "{path[2:]}"')

# Loading json.gz files
def load_json_gz(path = str, **kargs):
    """Open and read '.json.gz files', returning
    a list containing every json (it should be a 
    json-per-row) in the file."""

   # Read file, returning a list of strings per row
    with gzip.open(path, **kargs) as file:
        data = file.readlines()
    # Converting each row to a dict
    try:
        data = [eval(line) for line in data]
    except NameError:
        data = [json.loads(line) for line in data]

    # If success, show # of records
    print('Number of records:', len(data))
    print('Item type:', type(data[0]))
    return data


# Function to handle nested json in files:
def json_unpacking(
        df:pd.DataFrame,
        where: str,
        values: list[str],
        old_colums: list[str] | None,
        ) -> list[dict]:
    """Unpacks dictionaries within the given column (`where`
    the json objects are), returning a higher dimensional 
    DataFrame where eache value from json's `values` is a 
    new column in the resulting set.
    
    ## Parametters
        - ``df``: DataFrame 
        - ``where``: Label of the column where the array of 
        dictionaries is.
        - ``values``: Structure of dictionaries **must be known**. Only the
        keys in `values` will be unpacked.
        - ``old_columns``: Optional. Should be labels of original columns
        from the DataFrame; **always a list** -> for a single label:``['label']``
        should be passed."""
    
    # UNPACKING LOOP:
    items = []
    # For each row in data
    for i in df.index:
        row = {}
        if old_colums is not None:
            row = {label: df.loc[i,label] for label in old_colums}
        # for each item inside
        for item in df.loc[i, where]:
            # Keeping only values passed
            for value in values:
                row[value] = item[value]
            items.append(dict(row)) # A copy must be appended.

    # DataFrame again
    items = pd.DataFrame(items)
    # If success, print the shape of the resulting DataFrame.    
    print('Shape of the resulting array:', items.shape)
    return items 

import sys

# Loading data all at once function:
def load_dfs(from_main=False):
    """Read dataset files and return consumible
    DaFrames.
    
    Order: ``games``, ``reviews``, ``items``"""

    sys.path.append('../')

    paths = {
        'games': '../data/games.json.gz',
        'reviews': '../data/reviews.csv.gz',
        'items': '../data/items.csv.gz'
    }
    
    if from_main:
        paths['games'] = paths['games'][3:]
        paths['reviews'] = paths['reviews'][3:]
        paths['items'] = paths['items'][3:]

    games = pd.read_json(
        paths['games'],
        compression='gzip',
        lines=True
    )

    reviews = pd.read_csv(
        paths['reviews'],
        compression='gzip',
    )

    items = pd.read_csv(
        paths['items'],
        compression='gzip',
    )

    # If success
    print('DataFrames succesfully loaded.')
    return games, reviews, items

# Getting the year for games dataset
def get_year(y:str):
    """Funciton to extract only the year from 
    'release_date' column in 'games'dataset.
    
    It should be used as callable argument to 
    pass in ``pd.DataFrame.map() `` or 
    ``pd.DataFrame.apply()``"""
    # There are different formats along the column
    # And also strings from filling Na values before
    # Using exception handling to avoid errors and return years
    try:
        date = pd.to_datetime(y, format='mixed')
        year = date.year
        return year
    except:
        return np.nan