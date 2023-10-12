import numpy as np
from sklearn.decomposition import NMF


def calculateImageIDWeightPairs(feature_descriptor_ls, feature_descriptor, k, type): 
    image_id_weight_pairs = []
    ls_matrix = feature_descriptor_ls
    for i in range(len(feature_descriptor_ls)):
        image_id = i*2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
        
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True)

    with open(f"C:\\Khadyu\\ASU\\Fall 2023\\Multimedia & Web Databases\\Project\\Phase2\\cse515-project\\Code\\image_id_weight_pairs\\NNMF\\nnmf_{feature_descriptor}_{k}_wp_{type}.txt", "w") as file:
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            file.write(f"Image ID: {image_id}, Weight: {weight}\n")
            
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
