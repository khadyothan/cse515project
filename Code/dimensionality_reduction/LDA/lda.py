from file_paths import *

import numpy as np


def lda(data_matrix, k):
    print(data_matrix.shape)
    
    
    data_matrix_ls = {}
    return data_matrix_ls


def calculateImageIDWeightPairs(feature_descriptor_ls, feature_descriptor): 
    
    return True

if __name__ == "main":
    # data_matrix = np.loadtxt("Code\dimensionality_reduction\data_matrix_cm.csv", delimiter=',')     
    data_matrix = np.loadtxt(os.path.join(data_matrix_root_path, 'data_matrix_cm.csv'), delimiter=',')     
    cm_ls = lda(data_matrix)