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
    
    ## Parametters:
    - path: File location and name.
    - df: pd.Dataframe to save.
    - subset:Optional. Subset of columns to be saved.'''

    if subset == None:
        columns = list(df.columns)
    else:
        columns = subset
    # Creating the file
    df[columns].to_json(
        path_or_buf=path,
        orient='records',
        lines=True,
        compression='gzip'    
    )

# Loading json.gz files
def load_jsongz(path = str, **kargs):
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