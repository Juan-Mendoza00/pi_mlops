import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CosSimComputer:

    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.itemsMatrix = df.loc[:,'_Released after 2010':]
        self.items = df.loc[:,['item_id', 'app_name']]
        self.basisVector = None
        self.basisVector_index = None

        print('Cosine Similarity Computer fitted.')

    def set_basisVector(self, id):
        # Creating vector with the corect shape
        vector_idx = self.df.loc[self.df['item_id'] == id].index

        # Getting values of the resulting Series
        vector = self.itemsMatrix.iloc[vector_idx].values
        
        # Instance basis vector and its index
        self.basisVector = vector.reshape(1,-1)
        self.basisVector_index = vector_idx

    def _cos_sim(self, item:pd.Series):
        cos_sim = cosine_similarity(
            self.basisVector, item.values.reshape(1,-1)
            )[0,0]
        return cos_sim
    
    def compute_similarities(self):
        similarities = (
            # Dropping basisVector temporally to avoid computing the
            # similiarity to itself
            self.itemsMatrix.drop(index=self.basisVector_index)
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
        n_largest = similars.nlargest(n).index

        # Choosing to return the indexes
        if indexes:
            return n_largest
        
        # Returning items id and names
        items = self.items.iloc[n_largest]
        return items