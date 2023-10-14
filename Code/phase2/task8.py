from sklearn.metrics import pairwise_distances_argmin_min
import torch
import torchvision
from torchvision.io import read_image
import torchvision.datasets as datasets
from pymongo import MongoClient
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import json
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()



import dimensionality_reduction.SVD.svd as svd
import dimensionality_reduction.NNMF.nnmf as nnmf
import dimensionality_reduction.CP.cp as cp
import dimensionality_reduction.KMEANS.kmeans as kmeans
import dimensionality_reduction.KMEANS.kmeans2 as kmeans2
import dimensionality_reduction.LDA.lda as lda
import phase2.task5 as task5
import phase1.print_top_k_images as print_top_k_images
import extracting_feature_space.color_moments as color_moments
import extracting_feature_space.HOG as HOG
import extracting_feature_space.resnet_features as resnet_features
# Connecting to mongoDB client and creating database and collections
client = MongoClient("localhost", 27017)
DB = client["caltech101db"]
collection = DB["caltech101withGSimages"]
rep_collection=DB['labelrepresentativeimages']
# dataset_path = "/Users/lalitarvind/Downloads/MWD_Team_project"
dataset_path='../../data'
dataset = torchvision.datasets.Caltech101(root = dataset_path,download = True)
# query_img_id = input("Enter query image id:")
# print("\nSelect a Latent Semantics to be used(Select one among these): ")
# print("1. LS4_lda_resnet50_layer3_5\n2. LS3_nnmf_resnet50_avgpool_5\n\n")
# ls = input("Enter input ")
# k = int(input("Enter k value: "))
# ls_basepath = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/latent_semantics/"
# with open(f"{ls_basepath}{ls}.json","r") as f:
#     query_latent_semantics = json.load(f)

# if ls == "LS4_lda_resnet50_layer3_5":
#     query_latent_features = 
# elif ls == "LS3_nnmf_resnet50_avgpool_5"
#     query_latent_features = 

# def getRequiredFeatureSpace():
#     feature_spaces={"Color Moments":"color_moments_feature_descriptor",
#                 "HOG":"hog_feature_descriptor",
#                 "Resnet-AvgPool":"resnet50_avgpool_feature_descriptor",
#                 "ResNet-Layer3":"resnet50_layer3_feature_descriptor",
#                 "Resnet-FC":"resnet50_fc_feature_descriptor",
#                 "RestNet-Softmax":""}
#     print("Available the feature spaces")
#     for idx,item in enumerate(list(feature_spaces.keys())):
#         print(f"{idx+1}) {item}")
#     feature_spaces_ip=str(input("Enter the name of the feature space"))
#     return feature_spaces[feature_spaces_ip]

# def getRequiredReductionTech():
#     reductionTech={"Color Moments":"color_moments_feature_descriptor",
#                 "HOG":"hog_feature_descriptor",
#                 "Resnet-AvgPool":"resnet50_avgpool_feature_descriptor",
#                 "ResNet-Layer3":"resnet50_layer3_feature_descriptor",
#                 "Resnet-FC":"resnet50_fc_feature_descriptor",
#                 "RestNet-Softmax":""}
#     print("Available the feature spaces")
#     for idx,item in enumerate(list(feature_spaces.keys())):
#         print(f"{idx+1}) {item}")
#     feature_spaces_ip=str(input("Enter the name of the feature space"))
#     return feature_spaces[feature_spaces_ip]

def LS1(query_image_data, query_feature_model, dimredtech, k, K):
    query_image_vector, feature_model_data_file_path = None, None
    rep_keyname = {
        "color_moments_feature_descriptor":["color_moments_rep_image","color_moments_image_id"],
        "hog_feature_descriptor":["hog_rep_image","hog_image_id"],
        "resnet50_layer3_feature_descriptor":["layer3_rep_image","layer3_image_id"],
        "resnet50_avgpool_feature_descriptor":["avgpool_rep_image","avgpool_image_id"],
        "resnet50_fc_feature_descriptor":["fc_rep_image","fc_image_id"],
        "resnet_softmax_feature_descriptor":["resnet_softmax_rep_image","resnet_softmax_image_id"]
    }

    if query_feature_model == "color_moments_feature_descriptor":
        query_image_vector = color_moments.color_moments(query_image_data)
        feature_model_data_file_path = "cm"
    elif query_feature_model == "hog_feature_descriptor":
        query_image_vector = HOG.HOG(query_image_data)
        feature_model_data_file_path = "hog"
    else :
        query_layer3_vector, query_avgpool_vector, query_fc_vector = resnet_features.resnet_features(query_image_data)
        if query_feature_model == "resnet50_layer3_feature_descriptor":
            query_image_vector = query_layer3_vector
            feature_model_data_file_path = "layer3"
        elif query_feature_model == "resnet50_avgpool_feature_descriptor":
            query_image_vector = query_avgpool_vector
            feature_model_data_file_path = "avgpool"
        elif query_feature_model == "resnet50_fc_feature_descriptor":
            query_image_vector = query_fc_vector
            feature_model_data_file_path = "fc"
        else: 
            query_image_vector = resnet_features.resnetSoftmax(query_fc_vector)
            feature_model_data_file_path = "sm"
    query_image_vector = np.ravel(query_image_vector)
    # print(np.array(query_image_vector).shape)
    representatives = list(rep_collection.find({},{rep_keyname[query_feature_model][0]:1,
                                                  rep_keyname[query_feature_model][1]:1,}))
    # print(list(representative[0].keys()))
    # exit(0)
    if dimredtech == "svd":
        feature_model_data_matrix = np.loadtxt(f"D:\Project Multimedia\\new_phase2_repo\cse515project\Code\data_matrix\data_matrix_{feature_model_data_file_path}.csv", delimiter=',')
        combined_matrix = np.vstack((query_image_vector, feature_model_data_matrix))
        datamatrix_ls=svd.svd(combined_matrix,k)
        print(datamatrix_ls)
        query_image_ls = datamatrix_ls[0]
        print(representatives[0][rep_keyname[query_feature_model][1]]/2)
        # representatives_ls = [datamatrix_ls[label_dict[rep_keyname[query_feature_model][1]]/2] for label_dict in representatives]
        # print(representatives_ls)
        exit(0)
    # return query_image_vector
def task8(query_image_id, query_image_file, query_latent_semantics, K):
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
    
    print("\nSelect a feature model for the latent semantics(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n6. resnet_softmax\n\n")
    # query_feature_model = input("Enter input ")
    query_feature_model='color_moments'
    query_feature_model = str(query_feature_model) + "_feature_descriptor" 
    if query_latent_semantics != 2: #LS2 is CP decomposition
        print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
        print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
        # dimredtech = int(input("Enter your choice number: "))
        dimredtech=1
    # k = int(input("Enter k value for dimensionality reduction: "))
    k=10
    dimredtech_dict = {1:"svd",2:"nnmf",3:"lda",4:"k-means"}
    if query_latent_semantics == 1:
        LS1(query_image_data, query_feature_model, dimredtech_dict[dimredtech], k, K)
    elif query_latent_semantics == 2:
        LS2(query_image_data, query_feature_model, k, K)
    elif query_latent_semantics == 3:
        LS3(query_image_data, query_feature_model, dimredtech_dict[dimredtech], k, K)
    elif query_latent_semantics == 4:
        LS4(query_image_data, query_feature_model, dimredtech_dict[dimredtech], k, K)
    else:
        print("Enter a valid latent semantics choice!!")



data_transforms = T.Compose([T.Resize((224,224)), T.ToTensor(),T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x)])
dataset = datasets.Caltech101(dataset_path,download=True)
category_names = dataset.annotation_categories 





# query_type = int(input("Enter query type: "))
# query_image_id = None
# query_image_file = None
# if query_type == 1:
#     query_image_id = int(input("Enter query image ID: "))
# elif query_type == 2:
#     query_image_file = input("Give the query image file path: ")
# else: 
#     print("Enter valid query type!")
    
# print("Enter any of the Latent Semantics: \n1. LS1\n2. LS2\n3. LS3\n4. LS4\n")

# query_latent_semantics = int(input("Enter your choice number: "))

# K = int(input("Enter K value for finding K similar images: "))
if __name__ == "__main__":
    task8(2500, None, 1, 10)
