import numpy as np
import json
from sklearn.decomposition import LatentDirichletAllocation
import csv
def lda(data_matrix,feature_descriptor,k):
    data_matrix = np.array(data_matrix)
    data_matrix_min_val = data_matrix.min()
    if data_matrix_min_val<0:
        data_matrix = data_matrix - data_matrix_min_val
    fileExist=False
    ls_path = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/latent_semantics/"
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
        f1 = open(f"{ls_path}lda_{feature_descriptor}_{k}_ls.json","w")
        json.dump(LDA_factor_matrix.tolist() ,f1)
        f1.close()

    f = open(f"{ls_path}lda_{feature_descriptor}_{k}_ls.json","r")
    ls = json.loads(f.read())
    ls = np.transpose(ls)
    #query = np.array(data_matrix).reshape(len(data_matrix),len(data_matrix[0]))
    query_distribution = np.matmul(data_matrix,ls)
    f.close()
    return query_distribution

def calculateImageIDWeightPairs(feature_descriptor_ls,feature_descriptor,k): 
    image_id_weight_pairs = []
    ls_matrix = feature_descriptor_ls
    for i in range(len(ls_matrix)):
        image_id = i*2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
    
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True) 
    im_id_weight_basepath = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/image_id_weight_pairs/lda/"
    with open(f"{im_id_weight_basepath}{feature_descriptor}_{k}_ls.json", 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        fields = ["Image ID", "Weight"]
        csvwriter.writerow(fields)
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            csvwriter.writerow([image_id,weight])

data_matrix_path = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/data_matrix/"
data_matrix = np.loadtxt(f"{data_matrix_path}data_matrix_hog.csv", delimiter=',')  
#print(len(data_matrix))   
cm_ls = lda(data_matrix,"hog_feature_descriptor",10)
print(cm_ls)
#print(len(cm_ls),len(cm_ls[0]),)
calculateImageIDWeightPairs(cm_ls,"hog_feature_descriptor",10)