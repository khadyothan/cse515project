from matplotlib import pyplot as plt
import pymongo 
import numpy as np
import torchvision.datasets as datasets

cl = pymongo.MongoClient("mongodb://localhost:27017")
db = cl["caltech101db"]
collection = db["caltech101withGSimages"]

caltech101_directory = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/"
dataset = datasets.Caltech101(caltech101_directory, download=True)

def createMatrix(feature_descriptor):
    data_vector = collection.find_one({"image_id": 0, })[feature_descriptor]
    data_vector = np.array(data_vector).flatten()
    ncolumns = len(data_vector)
    
    data_matrix = np.empty((0, ncolumns))
    for i in range(8678):
        if i%2 == 0:
            data_vector = collection.find_one({"image_id": i, })[feature_descriptor]
            data_vector = np.array(data_vector).flatten()
            data_matrix = np.vstack((data_matrix, data_vector))
        print(i)
    return data_matrix

data_matrix_cm = createMatrix("color_moments_feature_descriptor")
data_matrix_hog = createMatrix("hog_feature_descriptor")
data_matrix_layer3 = createMatrix("resnet50_layer3_feature_descriptor")
data_matrix_avgpool = createMatrix("resnet50_avgpool_feature_descriptor")
data_matrix_fc = createMatrix("resnet50_fc_feature_descriptor")
basepath = "/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/dimensionality_reduction/"
file_path_cm = basepath + "data_matrix_cm.csv"
np.savetxt(file_path_cm, data_matrix_cm, delimiter=",")
file_path_hog = basepath + "data_matrix_hog.csv"
np.savetxt(file_path_hog, data_matrix_hog, delimiter=",")
file_path_layer3 = basepath + "data_matrix_layer3.csv"
np.savetxt(file_path_layer3, data_matrix_layer3, delimiter=",")
file_path_avgpool = basepath + "data_matrix_avgpool.csv"
np.savetxt(file_path_avgpool, data_matrix_avgpool, delimiter=",")
file_path_fc = basepath + "data_matrix_fc.csv"
np.savetxt(file_path_fc, data_matrix_fc, delimiter=",")
