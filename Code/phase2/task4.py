import pymongo
import torchvision.datasets as datasets
import torch
import numpy as np
import sys
sys.path.append('C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code')
import dimensionality_reduction.CP.cp as cp

def task4(query_feature_model, k):
    data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\data_matrices\data_matrix_{query_feature_model}.csv", delimiter=',')
    cp_ls = cp.cp(data_matrix, k)
    file_path_ls = f"C:\\Khadyu\\ASU\\Fall 2023\\Multimedia & Web Databases\\Project\\Phase2\\cse515-project\\Code\\latent_semantics\\CP\\cp_{query_feature_model}_{k}_ls.csv"
    np.savetxt(file_path_ls, cp_ls, delimiter=",")
    cp.calculateImageIDWeightPairs(cp_ls, query_feature_model, k)
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
    query_feature_model = input("Enter input ")
    query_feature_model = str(query_feature_model)
    
    k = int(input("Enter k value: "))
    
    result = task4(query_feature_model, k)
    
    if result:
        print("Sucessfully completed Task4!!!!")
    
    
    