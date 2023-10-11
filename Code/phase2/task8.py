import torch
import torchvision
from torchvision.io import read_image
import torchvision.datasets
from pymongo import MongoClient
#Connecting to mongoDB client and creating database and collections
client = MongoClient("localhost", 27017)
DB = client.caltech101db
collection = DB.caltech101withGSimages
# dataset_path = "/Users/lalitarvind/Downloads/MWD_Team_project"
# dataset = torchvision.datasets.Caltech101(root = dataset_path,download = False)
query_img_id = input("Enter query image id:")
print("\nSelect a Latent Semantics to be used(Select one among these): ")
print("1. LDA_color_moments\n2. LDA_hog\n3. LDA_resnet50_layer3\n4. LDA_resnet50_avgpool\n5. LDA_resnet50_fc\n6. SVD_color_moments\n7. SVD_hog\n8. SVD_resnet50_layer3\n9. SVD_resnet50_avgpool\n10. SVD_resnet50_fc\n11. NNMF_color_moments\n12. NNMF_hog\n13. NNMF_resnet50_layer3\n14. NNMF_resnet50_avgpool\n15. NNMF_resnet50_fc\n16. kmeans_color_moments\n17. kmeans_hog\n18. kmeans_resnet50_layer3\n19. kmeans_resnet50_avgpool\n20. kmeans_resnet50_fc\n\n")
query_latent_semantics = input("Enter input ")
temp = query_latent_semantics.split('_')[1]
query_feature_descriptor = temp[1]+ "_" + temp[2]
query_latent_semantics = str(query_latent_semantics) + "_feature_descriptor_latentSemantics"

query_features = collection.find_one({"image_id":query_img_id})[query_feature_descriptor]
k = int(input("Enter k value: "))