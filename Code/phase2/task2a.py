from Code.file_paths import *
import sys
sys.path.append(code_dir)

import pymongo
import torchvision.datasets as datasets
import numpy as np
import torch
import numpy as np
from PIL import Image
from extracting_feature_space import color_moments
import extracting_feature_space.HOG as HOG
import extracting_feature_space.resnet_features as resnet_features

def squared_l2_distance(input_vector, db_vector):
    return np.sqrt(np.sum(np.square(np.array(input_vector) - np.ravel(db_vector))))

def task2a(query_image_id, query_image_file, query_feature_space, k):
    
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
        
        
    representative_images = {}
    
    for label in range(101):
        label_data = collection.find({"label": label})
        feature_space_vectors = [doc[query_feature_space] for doc in label_data]
        mean_vector = np.mean(feature_space_vectors, axis=0)
        representative_images[label] = mean_vector
    
    # query_image_data = collection.find_one({"image_id": query_image_id})
    # query_image_vector = query_image_data[query_feature_space]
    
    if query_feature_space == "color_moments_feature_descriptor":
        query_image_vector = color_moments.color_moments(query_image_data)
    elif query_feature_space == "hog_feature_descriptor":
        query_image_vector = HOG.HOG(query_image_data)
    else :
        query_layer3_vector, query_avgpool_vector, query_fc_vector = resnet_features.resnet_features(query_image_data)
        if query_feature_space == "resnet50_layer3_feature_descriptor":
            query_image_vector = query_layer3_vector
        elif query_feature_space == "resnet50_avgpool_feature_descriptor":
            query_image_vector = query_avgpool_vector
        else:
            query_image_vector = query_fc_vector
    
    similar_images = []
    
    for label, mean_vector in representative_images.items():
        distance = np.linalg.norm(np.ravel(query_image_vector) - np.ravel(mean_vector))
        similarity = 1 / (1 + distance)
        similar_images.append((label, similarity))
    
    similar_images.sort(key=lambda x: x[1], reverse=True)
    
    top_k_similar_images = similar_images[:int(k)]
    for label, similarity in top_k_similar_images:
        print(f"Label: {label}, Similarity: {similarity}")

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"

    caltech101_directory = dataset_path
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    print("Enter '1' if you want to give an image ID as input or enter '2' if you want to give Image File as an Input: ")
    query_type = int(input("Enter query type: "))
    query_image_id = None
    query_image_file = None
    if query_type == 1:
        query_image_id = int(input("Enter query image ID: "))
    elif query_type == 2:
        query_image_file = input("Give the query image file path: ")
    else: 
        print("Enter valid query type!")
        
    print("\nSelect a feature space(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
    query_feature_space = input("Enter input ")
    query_feature_space = str(query_feature_space) + "_feature_descriptor"
    k = input("Enter k: ")
    
    task2a(query_image_id, query_image_file, query_feature_space, k)