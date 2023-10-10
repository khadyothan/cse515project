import pymongo
import torch
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def task6():
    
    image_vectors = [doc["resnet50_fc_feature_descriptor"] for doc in collection.find()]
    num_images = len(image_vectors)
    image_similarity_matrix = np.zeros((num_images, num_images))
    
    for i in range(num_images):
        for j in range(i, num_images):
            if i == j:
                image_similarity_matrix[i, j] = 1.0
            else:
                feature_i = np.array(image_vectors[i]).reshape(1, -1)
                feature_j = np.array(image_vectors[j]).reshape(1, -1)
                similarity_matrix  = cosine_similarity(feature_i, feature_j)
                similarity = similarity_matrix[0, 0]
                image_similarity_matrix[i, j] = similarity
                image_similarity_matrix[j, i] = similarity 
            print(i, j)
            
    file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/image_image_sm_fc.csv"
    np.savetxt(file_path_cm, image_similarity_matrix, delimiter=",")

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"

    caltech101_directory = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase1/data"
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    # task6()