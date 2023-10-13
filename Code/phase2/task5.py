import sys
import pymongo
import torch
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append('C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code')
import dimensionality_reduction.SVD.svd as svd
import dimensionality_reduction.NNMF.nnmf as nnmf
import dimensionality_reduction.LDA.lda as lda
import dimensionality_reduction.KMEANS.kmeans as kmeans

def createllsm(feature_descriptor, collection):
    
    representative_images = {}
    representative_vectors = [] 
    labels_count = 101
    
    # if len(query_image_vector) > 2:
    #     labels_count += 1
    #     representative_vectors.append(query_image_vector)
        
    for label in range(labels_count):
        label_data = collection.find({"label": label})
        feature_space_vectors = [doc[feature_descriptor] for doc in label_data]
        mean_vector = np.mean(feature_space_vectors, axis=0)
        representative_images[label] = mean_vector
        representative_vectors.append(mean_vector)
    
    num_labels = labels_count
    label_similarity_matrix = np.zeros((num_labels, num_labels))
    
    for i in range(num_labels):
        for j in range(i, num_labels):
            if i == j:
                label_similarity_matrix[i, j] = 1.0
            else:
                feature_i = np.array(representative_vectors[i]).reshape(1, -1)
                feature_j = np.array(representative_vectors[j]).reshape(1, -1)
                similarity_matrix  = cosine_similarity(feature_i, feature_j)
                similarity = similarity_matrix[0, 0]
                label_similarity_matrix[i, j] = similarity
                label_similarity_matrix[j, i] = similarity 
            print(i, j)
    
    print(label_similarity_matrix.shape)
    return label_similarity_matrix            

def task5(query_feature_model, k, dimredtech):
    if dimredtech == 1:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\label_label_sm\label_label_sm_{query_feature_model}.csv", delimiter=',')
        svd_ls = svd.svd(data_matrix, k)
        file_path_ls = f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\latent_semantics\SVD\svd_{query_feature_model}_{k}_ls_llsm.csv"
        np.savetxt(file_path_ls, svd_ls, delimiter=",")
        svd.calculateImageIDWeightPairs(svd_ls, query_feature_model, k, "llsm")
        
    elif dimredtech == 2:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\label_label_sm\label_label_sm_{query_feature_model}.csv", delimiter=',')
        nnmf_ls = nnmf.nnmf(data_matrix, k)
        file_path_ls = f"C:\\Khadyu\\ASU\\Fall 2023\\Multimedia & Web Databases\\Project\\Phase2\\cse515-project\\Code\\latent_semantics\\NNMF\\nnmf_{query_feature_model}_{k}_ls_llsm.csv"
        np.savetxt(file_path_ls, nnmf_ls, delimiter=",")
        nnmf.calculateImageIDWeightPairs(nnmf_ls, query_feature_model, k, "llsm")
        
    elif dimredtech == 3:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\label_label_sm\label_label_sm_{query_feature_model}.csv", delimiter=',')
        lda_ls = lda.lda(data_matrix, query_feature_model, k, "llsm")
        lda.calculateImageIDWeightPairs(lda_ls, query_feature_model, k, "llsm")
    
    elif dimredtech == 4:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\label_label_sm\label_label_sm_{query_feature_model}.csv", delimiter=',')
        kmeans_ls = kmeans.kmeans(data_matrix, k)
        file_path_ls = f"C:\\Khadyu\\ASU\\Fall 2023\\Multimedia & Web Databases\\Project\\Phase2\\cse515-project\\Code\\latent_semantics\\KMEANS\\kmeans_{query_feature_model}_{k}_ls_llsm.csv"
        np.savetxt(file_path_ls, kmeans_ls, delimiter=",")
        kmeans.calculateImageIDWeightPairs(kmeans_ls, query_feature_model, k, "llsm")
    else:
        print("Enter a valid dimensionality reduction technique choice!!")   
        
    return True

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"

    caltech101_directory = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase1/data"
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    print("\nSelect a feature model(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
    query_feature_model = input("Enter input: ")
    query_feature_model = str(query_feature_model)
    
    k = int(input("Enter k value: "))
    
    print("Label-Label similarity matrix is created!!\n")
    
    print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
    print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
    
    dimredtech = int(input("Enter your choice: "))
    
    result = task5(query_feature_model, k, dimredtech)
    
    if result:
        print("Sucessfully completed Task5!!!!")
        
    # label_label_sm_cm = createllsm("color_moments_feature_descriptor", [], collection)
    # file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_cm.csv"
    # np.savetxt(file_path_cm, label_label_sm_cm, delimiter=",")
    
    # label_label_sm_hog = createllsm("hog_feature_descriptor", [], collection)
    # file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_hog.csv"
    # np.savetxt(file_path_cm, label_label_sm_hog, delimiter=",")
    
    # label_label_sm_layer3 = createllsm("resnet50_layer3_feature_descriptor", [], collection)
    # file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_layer3.csv"
    # np.savetxt(file_path_cm, label_label_sm_layer3, delimiter=",")
    
    # label_label_sm_avgpool = createllsm("resnet50_avgpool_feature_descriptor", [], collection)
    # file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_avgpool.csv"
    # np.savetxt(file_path_cm, label_label_sm_avgpool, delimiter=",")
    
    # label_label_sm_fc = createllsm("resnet50_fc_feature_descriptor", [], collection)
    # file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_fc.csv"
    # np.savetxt(file_path_cm, label_label_sm_fc, delimiter=",")
