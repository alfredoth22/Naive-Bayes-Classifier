'''
Helper class to remove rows/columns from sparsed matrix
Created on Apr 2, 2012

@author: Justin Peel
http://stackoverflow.com/questions/2368544/how-can-i-remove-a-column-from-a-sparse-matrix-efficiently
'''

import numpy
from scipy import sparse
from bisect import bisect_left

class lil2(sparse.lil_matrix):
    def removecol(self,j):
        if j < 0:
            j += self.shape[1]

        if j < 0 or j >= self.shape[1]:
            raise IndexError('column index out of bounds')

        rows = self.rows
        data = self.data
        for i in xrange(self.shape[0]):
            pos = bisect_left(rows[i], j)
            if pos == len(rows[i]):
                continue
            elif rows[i][pos] == j:
                rows[i].pop(pos)
                data[i].pop(pos)
                if pos == len(rows[i]):
                    continue
            for pos2 in xrange(pos,len(rows[i])):
                rows[i][pos2] -= 1

        self._shape = (self._shape[0],self._shape[1]-1)
        
    def removerow(self,i):
        if i < 0:
            i += self.shape[0]

        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index out of bounds')

        self.rows = numpy.delete(self.rows,i,0)
        self.data = numpy.delete(self.data,i,0)
        self._shape = (self._shape[0]-1,self.shape[1])