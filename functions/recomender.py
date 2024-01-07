import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CosSimComputer:

    def __init__(self, df_train:pd.DataFrame):
        self.df_train = df_train
        self.itemsMatrix = df_train.iloc[:,2:]
        self.items = df_train.loc[:,['item_id', 'app_name']]
        self.basisVector = None

    def set_basisVector(self, id):
        # Creating vector with the corect shape
        vector_idx = self.df_train.loc[self.df_train['item_id'] == id].index

        # Getting values of the resulting Series
        vector = self.itemsMatrix.iloc[vector_idx].values
        
        # Instance basis vector
        self.basisVector = vector.reshape(1,-1)

    def _cos_sim(self, item:pd.Series):
        cos_sim = cosine_similarity(
            self.basisVector, item.values.reshape(1,-1)
            )[0,0]
        return cos_sim
    
    def compute_similarities(self):
        similarities = (
            self.itemsMatrix
            .apply(
                lambda row: self._cos_sim(item=row), 
                axis=1
                )
            )
        return similarities
    
    def n_most_similar(self, n:int, to_:int, indexes = False):

        # Re instancing basis vector for each compute
        self.set_basisVector(to_)

        # Computing similars
        similars = self.compute_similarities()

        # indexes for n largest excluding itself
        n_largest = similars.nlargest(n+1).index[1:]

        # Choosing to return the indexes
        if indexes:
            return n_largest
        
        # Returning items id and names
        items = self.items.iloc[n_largest]
        return items