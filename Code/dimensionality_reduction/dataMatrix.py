from Code.file_paths import *

from matplotlib import pyplot as plt
import pymongo 
import numpy as np
import torchvision.datasets as datasets

cl = pymongo.MongoClient("mongodb://localhost:27017")
db = cl["caltech101db"]
collection = db["phase2trainingdataset"]

caltech101_directory = dataset_path
dataset = datasets.Caltech101(caltech101_directory, download=False)

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

# file_path_cm = "/Users/rohitbathi/Desktop/Masters/CSE_MWD/project/cse515project-master/Code/dimensionality_reduction/data_matrix_cm.csv"
file_path_cm = os.path.join(data_matrix_root_path, 'data_matrix_cm.csv')
np.savetxt(file_path_cm, data_matrix_cm, delimiter=",")
# file_path_hog = "/Users/rohitbathi/Desktop/Masters/CSE_MWD/project/cse515project-master/Code/dimensionality_reduction/data_matrix_hog.csv"
file_path_hog = os.path.join(data_matrix_root_path, 'data_matrix_hog.csv')
np.savetxt(file_path_hog, data_matrix_hog, delimiter=",")
# file_path_layer3 = "/Users/rohitbathi/Desktop/Masters/CSE_MWD/project/cse515project-master/Code/dimensionality_reduction/data_matrix_layer3.csv"
file_path_layer3 = os.path.join(data_matrix_root_path, 'data_matrix_layer3.csv')
np.savetxt(file_path_layer3, data_matrix_layer3, delimiter=",")
# file_path_avgpool = "/Users/rohitbathi/Desktop/Masters/CSE_MWD/project/cse515project-master/Code/dimensionality_reduction/data_matrix_avgpool.csv"
file_path_avgpool = os.path.join(data_matrix_root_path, 'data_matrix_avgpool.csv')
np.savetxt(file_path_avgpool, data_matrix_avgpool, delimiter=",")
# file_path_fc = "/Users/rohitbathi/Desktop/Masters/CSE_MWD/project/cse515project-master/Code/dimensionality_reduction/data_matrix_fc.csv"
file_path_fc = os.path.join(data_matrix_root_path, 'data_matrix_fc.csv')
np.savetxt(file_path_fc, data_matrix_fc, delimiter=",")