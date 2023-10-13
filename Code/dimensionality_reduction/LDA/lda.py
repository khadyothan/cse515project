import csv
import json
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

def lda(data_matrix, feature_descriptor, k, type):
    print(data_matrix.shape)
    data_matrix = np.array(data_matrix)
    data_matrix_min_val = data_matrix.min()
    if data_matrix_min_val<0:
        data_matrix = data_matrix - data_matrix_min_val
    fileExist=False
    ls_path = "C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\latent_semantics\LDA"
    try:
        f = open(f"{ls_path}lda_{feature_descriptor}_{k}_ls.json","r")
        fileExist=True
    except:
        fileExist=False
    
    if(not fileExist):
        print("Creating file...")
        lda = LatentDirichletAllocation(n_components=k,verbose=True)
        lda.fit(data_matrix)
        LDA_factor_matrix = lda.components_ /lda.components_.sum(axis=1)[:, np.newaxis]
        f1 = open(f"{ls_path}\lda_{feature_descriptor}_{k}_ls_{type}.json","w")
        json.dump(LDA_factor_matrix.tolist() ,f1)
        f1.close()

    f = open(f"{ls_path}\lda_{feature_descriptor}_{k}_ls.json","r")
    ls = json.loads(f.read())
    ls = np.transpose(ls)
    query_distribution = np.matmul(data_matrix,ls)
    f.close()
    return np.array(query_distribution)


def calculateImageIDWeightPairs(feature_descriptor_ls, feature_descriptor, k, type): 
    image_id_weight_pairs = []
    ls_matrix = feature_descriptor_ls
    for i in range(len(ls_matrix)):
        image_id = i*2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
    
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True) 
    im_id_weight_basepath = "C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\image_id_weight_pairs\LDA"
    with open(f"{im_id_weight_basepath}\lda_{feature_descriptor}_{k}_wp_{type}.json", 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        fields = ["Image ID", "Weight"]
        csvwriter.writerow(fields)
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            csvwriter.writerow([image_id,weight])
    return True

# if __name__ == "__main__":
#     data_matrix = np.loadtxt("C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\dimensionality_reduction\data_matrix_cm.csv", delimiter=',')     
#     cm_ls = lda(data_matrix, "color_moments", 10)