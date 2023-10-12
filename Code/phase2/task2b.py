from file_paths import *
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

def task2b(query_image_id, query_image_file, k):
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
        
    

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
        
    k = input("Enter k: ")
    
    task2b(query_image_id, query_image_file, k)