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
    # In case of string passed that cannot be converted to float() it will 
    # raise a ValueError.
    
    except ValueError:
        # Looking for prices inside that string. 
        n = re.search(r"(\B[$]\d*[.]\d*)", value)

        # n will be None if it doesn't match
        if n is not None:
            # Returning the value without the first character (must be '$')
            n = round(float(n.group()[1:]), 2)
            return n
        # Returning zero otherwise because probably the string is 'Free...' or related.
        return 0
    

def genres_unpacking(df:pd.DataFrame, get_unique = False):
    """EXCLUSIVE USAGE FOR games DATASET.
    
    This function expect the games DataFrame
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
    """EXCLUSIVE USAGE FOR games DATASET.

    This function expect the games DataFrame
    to parse the 'specs' column and extract every
    appearance for single sepecifications. 

    ### It must be used in the preprocessing pipeline making possible to get dummies for specs
    ### by getiting disctinct values.
    
    - ``get_unique`` parametter: If ``True`` returning only unique values
    for specs. Default ``False``"""
    
    specs = []

    # Unpacking loop
    for specs_list in df['specs']:
        if specs_list == 'Empty':
            continue
        for spec in specs_list:
            specs.append(spec)

    # Converting to a pandas Series
    specs = pd.Series(specs)
    unique_specs = specs.unique()
    
    if get_unique:
        return unique_specs
    
    return specs

# Discretization and Dummies for year segmentation

def _year_binning(y:int):
    """Private method. It return labels for decades:
    - Before 2000
    - Between 2000 and 2010.
    - After 2010"""

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
    
    # Creating grouped column
    df['release_period'] = df['release_year'].map(_year_binning, na_action='ignore')

    if drop_old:
        # Dropping years column
        df.drop(columns='release_year', inplace=True)

    if dummies:
        # Getting dummies
        df = pd.get_dummies(df, columns=['release_period'], prefix='', dtype=int)
        return df

    return None


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

    # Creating grouped column
    df['cost'] = df['price'].map(_price_binning, na_action='ignore')

    if drop_old:
        # Dropping years column
        df.drop(columns='price', inplace=True)

    if dummies:
        # if True then Getting dummies else don't
        df = pd.get_dummies(df, columns=['cost'], prefix='', dtype=int)
        return df
    
    # Return df only with binned column
    return None


# Function to get the dummie rows for genre
def _genres_dummies(list_, unique):
    """Private method to create dummie rows from a list.
    
    An array of unique values should be passed and it will match
    the position if come item in `unique` is found.
    
    Returns a 1-dimensional array full of ones (for mathed items) and
    zeros (for unmatched items)."""

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
    """ This function es exclusively used for getting dummie genres row
    for the games DataFrame which contains lists stored in tabular data.
    
    Set `drop_old = True` to drop the original genres column."""

    # Gets unique genres for 
    unique_genres = genres_unpacking(df, get_unique=True)

    # Getting dummies
    dummies = df['genres'].apply(_genres_dummies, unique=unique_genres)

    # Converting the result matrix to a DataFrame
    dummies_df = pd.DataFrame(data=dummies.tolist(), columns=unique_genres)

    # Concatenating dummies to df
    df = pd.concat([df, dummies_df], axis=1)

    if drop_old:
        df.drop(columns='genres', inplace=True)

    return df


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
    """ This function es exclusively used for getting dummie specs row
    for the games DataFrame, which contains lists stored in tabular data.
    
    Set `drop_old = True` to drop the original genres column."""

    # Getting unique values for specs
    unique_specs = specs_unpacking(df, get_unique=True)

    # Getting dummies
    dummies = df['specs'].apply(_specs_dummies, unique=unique_specs)

    # Converting the result to a DataFrame
    dummies_df = pd.DataFrame(data=dummies.tolist(), columns=unique_specs)

    # Concatenating dataFrame
    df = pd.concat([df, dummies_df], axis=1)

    if drop_old:
        df.drop(columns='specs', inplace=True)
    
    return df

# Complete pipeline
def preprocess_games(df:pd.DataFrame):
    """Applies all preprocessing functions needed to transform the games
    DataFrame into a higher dimensional sparse matrix that can be used
    to compute similiarities between vectors (which now represent single items)."""
    
    df_copy = df.copy()
    df_copy = df_copy.drop(columns='tags')
    # Working on a copy to avoid modifications on the original

    # Cleaning and filling prices
    df_copy['price'] = df_copy['price'].apply(float_prices)
    df_copy['price'].fillna(df_copy['price'].median(), inplace=True)

    # year binning
    df_copy = year_binning(df_copy, dummies=True, drop_old=True)

    # Price binning
    df_copy = price_binning(df_copy, dummies=True, drop_old=True)

    # Genres dummies
    df_copy = genres_dummies(df_copy, drop_old=True)

    # Specs dummies
    df_copy = specs_dummies(df_copy, drop_old=True)

    print('Games data preprocessed. Ready to store in recomender.')
    return df_copy