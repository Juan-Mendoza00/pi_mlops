import pandas as pd
import numpy as np 
import re


# Function to return correct values of prices in column
# 'price' for the games dataset
# Converting prices to float
def float_prices(value):
    """## Catching prices inside strings on the 'games' dataset.
     
    This function can be use to extract price values inside string
    labels when matching the pattern: '[...] $(some digits).
    It uses regular expressions.
    
    Ignores null values'"""
    if value is None:
        return np.NaN
    
    # Type converting and extracting block
    try:
        n = round(float(value), 2)
        return n
    # In case of string passed that cannot be
    # converted to float() it will raise a ValueError
    # In the other hand, passing a NoneType will raise a 
    # TypeError, so it must be catched too
    # Catch the exceptions
    except ValueError:
        # Looking for prices inside that string
        # n will be None if it doesn't match.
        n = re.search(r"(\B[$]\d*[.]\d*)", value)

        if n is not None:
            # Returning the value without the first
            # character (must be '$')
            n = round(float(n.group()[1:]), 2)
            return n
        # Returning zero otherwise because probably
        # the string is something like 'Free...' or related
        return 0
    

def genres_unpacking(df:pd.DataFrame, get_unique = False):
    """This function expect the games DataFrame
    to parse the 'genres' column and extract every
    appearance for genres. 
    ### It must be used in the preprocessing pipeline making possible to get dummies for genres
    ### by getiting disctinct values
    
    - ``get_unique`` parametter: If ``True`` returning only unique values
    for genres. Default ``False``"""
    # List of every genre in the dataframe
    genres = []
    # Unpacking loop
    for genres_list in df['genres']:
        if genres_list == 'Empty':
            continue
        for genre in genres_list:
            genres.append(genre)

    # Converting to a pandas Series
    genres = pd.Series(genres)
    unique_genres = genres.unique()

    if get_unique:
        return unique_genres
    return genres


def specs_unpacking(df:pd.DataFrame, get_unique=False):
    """This function expect the games DataFrame
    to parse the 'specs' column and extract every
    appearance for single sepecifications. 

    ### It must be used in the preprocessing pipeline making possible to get dummies for specs
    ### by getiting disctinct values.
    
    - ``get_unique`` parametter: If ``True`` returning only unique values
    for specs. Default ``False``"""
    specs = []
    for l in df['specs']:
        if l == 'Empty':
            continue
        for spec in l:
            specs.append(spec)

    specs = pd.Series(specs)
    unique_specs = specs.unique()
    
    if get_unique:
        return unique_specs
    return specs

# Discretization and Dummies for year segmentation

def _year_binning(y:int):
    if y < 2000:
        return 'Released before 2000'
    elif y >= 2000 and y < 2010:
        return 'Released in 2000-2010'
    else:
        return 'Released after 2010'
    
def year_binning(df:pd.DataFrame, dummies=False, drop_old=False):
    """EXCLUSIVE USAGE FOR games DATASET.
    
    Applies discretization (binning) technique for 
    'release_year' column. It will create a new column
    containing labels instead of years. Resulting groups 
    are 'Year before 2000', 'Year between 2000 and 2010,
    and 'Year after 2010'
    
    ## Parametters
    - ``dumies``: Default ``False``. If ``True``, it will return not only 
    a new labeled column for years but also the corresponding
    dummie variables.
    
    - ``drop_old``: Default ``False``. When set to ``True``, it will drop
    the original column for release years and return only the
    one created after grouping.'"""
    df_copy = df.copy()
    # Creating grouped column
    df_copy['release_period'] = df_copy['release_year'].map(_year_binning, na_action='ignore')

    if drop_old:
        # Dropping years column
        df_copy = df_copy.drop(columns='release_year')

    if dummies:
        # Getting dummies
        df_copy = pd.get_dummies(df_copy, columns=['release_period'], prefix='', dtype=int)
        # return df_copy with dummie columns
        return df_copy
    
    # Rturning df_copy with new binned column
    return df_copy


def _price_binning(p:float):
    if p < 5:
        return 'Very low cost'
    elif p >= 5 and p < 30:
        return 'Cheap'
    elif p >= 30 and p < 60:
        return 'Typical price'
    else:
        return 'Expensive'
    
def price_binning(df:pd.DataFrame, dummies=False, drop_old=False):
    """EXCLUSIVE USAGE FOR games DATASET.
    
    Applies discretization (binning) technique for 
    'price' column. It will create a new column
    containing labels instead of single prices. Resulting groups 
    are 'Very low cost', 'Cheap', 'Typical' and 'Expensive'.
    
    ## Parametters
    - ``dumies``: Default ``False``. If ``True``, it will return not only 
    a new labeled column for prices but also the corresponding
    dummie variables.
    
    - ``drop_old``: Default ``False``. When set to ``True``, it will drop
    the original column for prices and return only the the df with the
    one created after grouping.'"""

    df_copy = df.copy()
    # Creating grouped column
    df_copy['cost'] = df_copy['price'].map(_price_binning, na_action='ignore')

    if drop_old:
        # Dropping years column
        df_copy.drop(columns='price', inplace=True)

    if dummies:
        # Getting dummies
        df_copy = pd.get_dummies(df_copy, columns=['cost'], prefix='', dtype=int)
        # returning with binned column and dummie columns for it
        return df_copy
    
    # Return df_copy only with binned column
    return df_copy


# Dummies for genres
# Function to get the dummie rows for genre
def _genres_dummies(list_, unique):
    # Creating the dummie row
    row = np.zeros(len(unique), dtype=int)
    # No genre returns row full of zeros
    if list_ == 'Empty':
        return row
    # Loop to check every single genre appearance.
    for pos, genre in enumerate(unique):
        if genre in list_:
            # update row position
            row[pos] = 1
    return row

def genres_dummies(df:pd.DataFrame, drop_old=False):

    # Gets unique genres for 
    unique_genres = genres_unpacking(df, get_unique=True)
    df_copy = df.copy()
    # Getting dummies
    dummies = df_copy['genres'].apply(_genres_dummies, unique=unique_genres)

    # Converting the result to a DataFrame
    dummies_df = pd.DataFrame(data=dummies.tolist(), columns=unique_genres)
    # Concatenating dummies to df=df_games
    df_copy = pd.concat([df_copy, dummies_df], axis=1)

    if drop_old:
        df_copy.drop(columns='genres', inplace=True)
    
    return df_copy


# Dummies for specs
# Function to get dummie rows
def _specs_dummies(list_, unique):
    # Creating the dummie row
    row = np.zeros(len(unique), dtype=int)
    # No specs returns a row full of zeros
    if list_ == 'Empty':
        return row
    # Loop to check every single genre appearance.
    for pos, spec in enumerate(unique):
        if spec in list_:
            # update row position
            row[pos] = 1
    return row


def specs_dummies(df:pd.DataFrame, drop_old=False):
    # Getting unique values for specs
    unique_specs = specs_unpacking(df, get_unique=True)

    df_copy = df.copy()

    # Getting dummies
    dummies = df_copy['specs'].apply(_specs_dummies, unique=unique_specs)

    # Converting the result to a DataFrame
    dummies_df = pd.DataFrame(data=dummies.tolist(), columns=unique_specs)

    # Concatenating dataFrame
    df_copy = pd.concat([df_copy, dummies_df], axis=1)

    if drop_old:
        df_copy.drop(columns='specs', inplace=True)
    
    return df_copy