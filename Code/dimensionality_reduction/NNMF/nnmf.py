import numpy as np
from sklearn.decomposition import NMF

def nnmf(data_matrix, k):
    print(data_matrix.shape)
    U, S, VT = np.linalg.svd(data_matrix, full_matrices=False)
    U[U < 0] = 0
    S[S < 0] = 0
    VT[VT < 0] = 0
    U_reduced = U[:, :k]
    S_reduced = np.diag(S[:k])
    VT_reduced = VT[:k, :]
    data_matrix_ls = np.dot(U_reduced, S_reduced)
    print(data_matrix_ls.shape)
    return data_matrix_ls

# cm_ls = nnmf("cm")
# print(cm_ls.shape)
# file_path_cm_ls = "Code\\dimensionality_reduction\\NNMF\\cm_ls.npy"
# np.savetxt(file_path_cm_ls, cm_ls, delimiter=",")
