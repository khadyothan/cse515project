import pymongo
import torchvision.datasets as datasets
import torch
import numpy as np
import sys
sys.path.append('C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code')
import dimensionality_reduction.SVD.svd as svd
import dimensionality_reduction.NNMF.nnmf as nnmf
import dimensionality_reduction.LDA.lda as lda
import dimensionality_reduction.KMEANS.kmeans as kmeans

def task3(query_feature_model, k, dimredtech):
    if dimredtech == 1:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\data_matrices\data_matrix_{query_feature_model}.csv", delimiter=',')
        svd_ls = svd.svd(data_matrix, k)
        file_path_ls = f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\latent_semantics\SVD\svd_{query_feature_model}_{k}_ls.csv"
        np.savetxt(file_path_ls, svd_ls, delimiter=",")
        svd.calculateImageIDWeightPairs(svd_ls, query_feature_model, k, "")
        
    elif dimredtech == 2:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\data_matrices\data_matrix_{query_feature_model}.csv", delimiter=',')
        nnmf_ls = nnmf.nnmf(data_matrix, k)
        file_path_ls = f"C:\\Khadyu\\ASU\\Fall 2023\\Multimedia & Web Databases\\Project\\Phase2\\cse515-project\\Code\\latent_semantics\\NNMF\\nnmf_{query_feature_model}_{k}_ls.csv"
        np.savetxt(file_path_ls, nnmf_ls, delimiter=",")
        nnmf.calculateImageIDWeightPairs(nnmf_ls, query_feature_model, k, "")
        
    elif dimredtech == 3:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\data_matrices\data_matrix_{query_feature_model}.csv", delimiter=',')
        lda_ls = lda.lda(data_matrix, query_feature_model, k, "")
        lda.calculateImageIDWeightPairs(lda_ls, query_feature_model, k, "")
    
    elif dimredtech == 4:
        data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\data_matrices\data_matrix_{query_feature_model}.csv", delimiter=',')
        kmeans_ls = kmeans.kmeans(data_matrix, k)
        file_path_ls = f"C:\\Khadyu\\ASU\\Fall 2023\\Multimedia & Web Databases\\Project\\Phase2\\cse515-project\\Code\\latent_semantics\\KMEANS\\kmeans_{query_feature_model}_{k}_ls.csv"
        np.savetxt(file_path_ls, kmeans_ls, delimiter=",")
        kmeans.calculateImageIDWeightPairs(kmeans_ls, query_feature_model, k, "")
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
    
    print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
    print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
    
    dimredtech = int(input("Enter your choice: "))
    
    result = task3(query_feature_model, k, dimredtech)
    
    if result:
        print("Sucessfully completed Task3!!!!")