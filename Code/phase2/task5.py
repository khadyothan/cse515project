import pymongo
import torch
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def task5(feature_descriptor, query_image_vector, collection):
    
    representative_images = {}
    representative_vectors = [] 
    labels_count = 101
    
    if len(query_image_vector) > 2:
        labels_count += 1
        representative_vectors.append(query_image_vector)
        
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

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"

    caltech101_directory = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase1/data"
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    label_label_sm_cm = task5("color_moments_feature_descriptor", [], collection)
    file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_cm.csv"
    np.savetxt(file_path_cm, label_label_sm_cm, delimiter=",")
    
    label_label_sm_hog = task5("hog_feature_descriptor", [], collection)
    file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_hog.csv"
    np.savetxt(file_path_cm, label_label_sm_hog, delimiter=",")
    
    label_label_sm_layer3 = task5("resnet50_layer3_feature_descriptor", [], collection)
    file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_layer3.csv"
    np.savetxt(file_path_cm, label_label_sm_layer3, delimiter=",")
    
    label_label_sm_avgpool = task5("resnet50_avgpool_feature_descriptor", [], collection)
    file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_avgpool.csv"
    np.savetxt(file_path_cm, label_label_sm_avgpool, delimiter=",")
    
    label_label_sm_fc = task5("resnet50_fc_feature_descriptor", [], collection)
    file_path_cm = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase2/cse515-project/Code/phase2/label_label_sm/label_label_sm_fc.csv"
    np.savetxt(file_path_cm, label_label_sm_fc, delimiter=",")