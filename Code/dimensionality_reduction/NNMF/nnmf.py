import numpy as np
from sklearn.decomposition import NMF

def nnmf(feature_descriptor):
    data_matrix = np.loadtxt(f"Code\dimensionality_reduction\data_matrix_{feature_descriptor}.csv", delimiter=',')
    k = 100
    U, S, VT = np.linalg.svd(data_matrix, full_matrices=False)
    U[U < 0] = 0
    S[S < 0] = 0
    VT[VT < 0] = 0

    U_reduced = U[:, :k]
    S_reduced = np.diag(S[:k])
    VT_reduced = VT[:k, :]

    data_matrix_lr = np.dot(U_reduced, S_reduced)
    return data_matrix_lr

cm_ls = nnmf("cm")
print(cm_ls.shape)

file_path_cm_ls = "Code\\dimensionality_reduction\\NNMF\\cm_ls.npy"
np.savetxt(file_path_cm_ls, cm_ls, delimiter=",")
