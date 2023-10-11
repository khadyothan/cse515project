import numpy as np

def svd(data_matrix, k):
    print(data_matrix.shape)
    cov_matrix = np.dot(data_matrix.T, data_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvectors_k = eigenvectors[:, :k]
    singular_values = np.sqrt(eigenvalues[:k])
    U_reduced = np.dot(data_matrix, eigenvectors_k)
    VT_reduced = eigenvectors_k.T
    S_reduced = np.diag(singular_values)
    data_matrix_ls = np.dot(U_reduced, S_reduced)
    print(data_matrix_ls.shape)
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
            
if __name__ == "__main__":
    data_matrix = np.loadtxt("Code\dimensionality_reduction\data_matrix_cm.csv", delimiter=',')     
    cm_ls = svd(data_matrix)
    data_matrix = np.loadtxt("Code\dimensionality_reduction\data_matrix_hog.csv", delimiter=',')   
    hog_ls = svd(data_matrix)
    data_matrix = np.loadtxt("Code\dimensionality_reduction\data_matrix_fc.csv", delimiter=',')   
    fc_ls = svd(data_matrix)

    calculateImageIDWeightPairs(cm_ls, "cm")
    calculateImageIDWeightPairs(hog_ls, "hog")
    calculateImageIDWeightPairs(fc_ls, "fc")

    file_path_cm_ls = "Code\dimensionality_reduction\SVD\cm_ls.npy"
    file_path_hog_ls = "Code\dimensionality_reduction\SVD\hog_ls.npy"
    file_path_fc_ls = "Code\\dimensionality_reduction\\SVD\\fc_ls.npy"

    np.savetxt(file_path_cm_ls, cm_ls, delimiter=",")
    np.savetxt(file_path_hog_ls, hog_ls, delimiter=",")
    np.savetxt(file_path_fc_ls, fc_ls, delimiter=",")