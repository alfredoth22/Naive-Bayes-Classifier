'''
Created on Mar 30, 2012

@author: Mujtaba Badat
'''
import numpy as np

class Classifier ( object ):
    def __init__ ( self ):
        self.params = []
    
    def fit ( self, X, y ):
        raise NotImplementedError
    
    def predict ( self , X ):
        raise NotImplementedError
    
    def save_params ( self , fname ):
        params = dict ([( p , getattr ( self, p )) for p in self.params ])
        np.savez( fname, **params )
    
    def load_params ( self, fname ):
        params = np.load( fname )
        for name in self.params:
            setattr ( self, name, params[ name ])   