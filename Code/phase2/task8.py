import torch
import torchvision
from torchvision.io import read_image
import torchvision.datasets
from pymongo import MongoClient
import json
#Connecting to mongoDB client and creating database and collections
client = MongoClient("localhost", 27017)
DB = client.caltech101db
collection = DB.caltech101withGSimages
# dataset_path = "/Users/lalitarvind/Downloads/MWD_Team_project"
# dataset = torchvision.datasets.Caltech101(root = dataset_path,download = False)
query_img_id = input("Enter query image id:")
print("\nSelect a Latent Semantics to be used(Select one among these): ")
print("1. LS4_lda_resnet50_layer3_5\n2. LS3_nnmf_resnet50_avgpool_5\n\n")
ls = input("Enter input ")
k = int(input("Enter k value: "))
ls_basepath = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/latent_semantics/"
with open(f"{ls_basepath}{ls}.json","r") as f:
    query_latent_semantics = json.load(f)

if ls == "LS4_lda_resnet50_layer3_5":
    query_latent_features = 
elif ls == "LS3_nnmf_resnet50_avgpool_5"
    query_latent_features = 
