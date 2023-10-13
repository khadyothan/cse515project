import numpy as np
import json
from sklearn.decomposition import LatentDirichletAllocation
import csv
import pickle
CODE_BASEPATH = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/"

def lda(task,data_matrix,feature_descriptor,k):
    data_matrix = np.array(data_matrix)
    data_matrix_min_val = data_matrix.min()
    if data_matrix_min_val<0:
        data_matrix = data_matrix - data_matrix_min_val
    fileExist=False

    try:
        f = open(f"{CODE_BASEPATH}latent_semantics/{task}_lda_{feature_descriptor}_{k}_ls.json","r")
        f1 = open(f"{CODE_BASEPATH}dimensionality_reduction/LDA/models/{task}_lda_{feature_descriptor}_{k}.pkl","rb")
        fileExist=True
    except:
        fileExist=False
    
    if(not fileExist):
        print("Creating file...")
        lda = LatentDirichletAllocation(n_components=k,verbose=True)
        lda.fit(data_matrix)
        with open(f"{CODE_BASEPATH}dimensionality_reduction/LDA/models/{task}_lda_{feature_descriptor}_{k}.pkl",'wb') as f1:
            pickle.dump(lda,f1)
        LDA_factor_matrix = lda.components_ /lda.components_.sum(axis=1)[:, np.newaxis]
        f2 = open(f"{CODE_BASEPATH}latent_semantics/{task}_lda_{feature_descriptor}_{k}_ls.json","w")
        json.dump(LDA_factor_matrix.tolist() ,f2)
        f1.close()
        f2.close()

    # ls = np.transpose(ls)
    # #query = np.array(data_matrix).reshape(len(data_matrix),len(data_matrix[0]))
    #query_distribution = np.matmul(data_matrix,ls)
    with open(f"{CODE_BASEPATH}dimensionality_reduction/LDA/models/{task}_lda_{feature_descriptor}_{k}.pkl","rb") as f1:
        lda_model = pickle.load(f1)
    query_distribution = lda_model.transform(data_matrix)
    #f.close()
    f1.close()
    return query_distribution

def calculateImageIDWeightPairs(task,feature_descriptor_ls,feature_descriptor,k): 
    image_id_weight_pairs = []
    ls_matrix = feature_descriptor_ls
    for i in range(len(ls_matrix)):
        image_id = i*2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
    
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True) 
    with open(f"{CODE_BASEPATH}image_id_weight_pairs/lda/{task}_{feature_descriptor}_{k}_ls.json", 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        fields = ["Image ID", "Weight"]
        csvwriter.writerow(fields)
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            csvwriter.writerow([image_id,weight])

task = "task3"
data_matrix = np.loadtxt(f"{CODE_BASEPATH}data_matrix/data_matrix_hog.csv", delimiter=',')  
print(data_matrix.shape)
#print(len(data_matrix))   
cm_ls = lda(task,data_matrix,"hog_feature_descriptor",10)
#print(cm_ls)
#print(len(cm_ls),len(cm_ls[0]))
calculateImageIDWeightPairs(task,cm_ls,"hog_feature_descriptor",10)