import numpy as np

def svd(feature_descriptor):
    data_matrix = np.loadtxt(f"Code\dimensionality_reduction\data_matrix_{feature_descriptor}.csv", delimiter=',')
    print(data_matrix.shape)
    k=100
    U, S, VT = np.linalg.svd(data_matrix, full_matrices=False)
    U_reduced = U[:, :k]
    S_reduced = np.diag(S[:k])
    VT_reduced = VT[:k, :]
    data_matrix_ls = np.dot(U_reduced, S_reduced)
    return data_matrix_ls
    
def calculateImageIDWeightPairs(feature_descriptor_ls, feature_descriptor): 
    image_id_weight_pairs = []
    ls_matrix = feature_descriptor_ls
    for i in range(4149):
        image_id = i*2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
        
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True)

    with open(f"Code\dimensionality_reduction\SVD\sorted_image_id_weights_{feature_descriptor}.txt", "w") as file:
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            file.write(f"Image ID: {image_id}, Weight: {weight}\n")
            
cm_ls = svd("cm")
hog_ls = svd("hog")
fc_ls = svd("fc")

calculateImageIDWeightPairs(cm_ls, "cm")
calculateImageIDWeightPairs(hog_ls, "hog")
calculateImageIDWeightPairs(fc_ls, "fc")

# file_path_cm_ls = "Code\dimensionality_reduction\SVD\cm_ls.npy"
# file_path_hog_ls = "Code\dimensionality_reduction\SVD\hog_ls.npy"
# file_path_fc_ls = "Code\\dimensionality_reduction\\SVD\\fc_ls.npy"

# np.savetxt(file_path_cm_ls, cm_ls, delimiter=",")
# np.savetxt(file_path_hog_ls, hog_ls, delimiter=",")
# np.savetxt(file_path_fc_ls, fc_ls, delimiter=",")


