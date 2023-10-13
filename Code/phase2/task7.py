import pymongo
import torchvision.datasets as datasets
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code')
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

def LS4(query_image_data, query_feature_model, dimredtech, k, K):

    return True 

def LS3(query_image_data, query_feature_model, dimredtech, k, K):
    query_image_vector, feature_model_data_file_path = None, None
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
        else:
            query_image_vector = query_fc_vector
            feature_model_data_file_path = "fc"
            
    query_image_vector = np.ravel(query_image_vector)
    feature_model_data_matrix = task5.task5(query_feature_model, query_image_vector, collection1) #102x102
    
    if dimredtech == 1: 
        latent_space_matrix = svd.svd(feature_model_data_matrix, k)
    elif dimredtech == 2:
        latent_space_matrix = nnmf.nnmf(feature_model_data_matrix, k) 
    elif dimredtech == 3:
        similar_images = {}  
    elif dimredtech == 4:
        print(dimredtech)
        latent_space_matrix = kmeans.kmeans(feature_model_data_matrix, k) 
    else:
        print("Enter valid dimensionality reduction technique choice!!")
        
    query_image_vector_ls = latent_space_matrix[0]
    database_vectors_ls = latent_space_matrix[1:]
    # distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
    
    similarities = cosine_similarity(query_image_vector_ls.reshape(1, -1), database_vectors_ls)
    similarity_scores = similarities[0]
    print(similarity_scores)
    similar_images = {}
    for i, distance in enumerate(similarity_scores):
        cursor = collection2.find({"label": i})
        for doc in cursor:
            index = doc.get("fc_image_id")
            similar_images[index] = distance   
    similar_images = dict(sorted(similar_images.items(), key=lambda x: x[1]))
    top_k_similar_images = dict(list(similar_images.items())[:K])
    images_to_display = {image_id: {'image': image, 'distance': top_k_similar_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_similar_images}
    print_top_k_images.print_images(images_to_display, "Heading", target_size=(224, 224))
    

def LS2(query_image_data, query_feature_model, k, K):
    query_image_vector, feature_model_data_file_path = None, None
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
        else:
            query_image_vector = query_fc_vector
            feature_model_data_file_path = "fc"
            
    query_image_vector = np.ravel(query_image_vector)
    feature_model_data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\dimensionality_reduction\data_matrix_{feature_model_data_file_path}.csv", delimiter=',')
    print(feature_model_data_matrix.shape)
    combined_matrix = np.vstack((query_image_vector, feature_model_data_matrix))
    latent_space_matrix = cp.cp(combined_matrix, k)
    
    query_image_vector_ls = latent_space_matrix[0]
    database_vectors_ls = latent_space_matrix[1:]
    distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
    similar_images = {}
    for i, distance in enumerate(distances):
        similar_images[i*2] = distance   
    
    similar_images = dict(sorted(similar_images.items(), key=lambda x: x[1]))
    top_k_similar_images = dict(list(similar_images.items())[:K])
    images_to_display = {image_id: {'image': image, 'distance': top_k_similar_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_similar_images}
    print_top_k_images.print_images(images_to_display, "Heading", target_size=(224, 224))
    


def LS1(query_image_data, query_feature_model, dimredtech, k, K):
    
    query_image_vector, feature_model_data_file_path = None, None
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
        else:
            query_image_vector = query_fc_vector
            feature_model_data_file_path = "fc"
            
    query_image_vector = np.ravel(query_image_vector)
    feature_model_data_matrix = np.loadtxt(f"C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\dimensionality_reduction\data_matrix_{feature_model_data_file_path}.csv", delimiter=',')
    print(feature_model_data_matrix.shape)
    combined_matrix = np.vstack((query_image_vector, feature_model_data_matrix))
    
    if dimredtech == 1: 
        latent_space_matrix = svd.svd(combined_matrix, k)
    elif dimredtech == 2:
        latent_space_matrix = nnmf.nnmf(combined_matrix, k) 
    elif dimredtech == 3:
        latent_space_matrix = lda.lda(combined_matrix, "color_moments", k)
    elif dimredtech == 4:
        latent_space_matrix = kmeans2.kmeans2(feature_model_data_matrix, k)
    else:
        print("Enter valid dimensionality reduction technique choice!!")
        
    query_image_vector_ls = latent_space_matrix[0]
    database_vectors_ls = latent_space_matrix[1:]
    distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
    similar_images = {}
    for i, distance in enumerate(distances):
        similar_images[i*2] = distance   
    similar_images = dict(sorted(similar_images.items(), key=lambda x: x[1]))
    top_k_similar_images = dict(list(similar_images.items())[:K])
    images_to_display = {image_id: {'image': image, 'distance': top_k_similar_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_similar_images}
    print_top_k_images.print_images(images_to_display, "Heading", target_size=(224, 224))


def task7(query_image_id, query_image_file, query_latent_semantics, K):
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
    
    print("\nSelect a feature model for the latent semantics(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
    query_feature_model = input("Enter input ")
    query_feature_model = str(query_feature_model) + "_feature_descriptor" 
    if query_latent_semantics != 2: #LS2 is CP decomposition
        print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
        print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
        dimredtech = int(input("Enter your choice number: "))
    k = int(input("Enter k value for dimensionality reduction: "))
    
    if query_latent_semantics == 1:
        LS1(query_image_data, query_feature_model, dimredtech, k, K)
    elif query_latent_semantics == 2:
        LS2(query_image_data, query_feature_model, k, K)
    elif query_latent_semantics == 3:
        LS3(query_image_data, query_feature_model, dimredtech, k, K)
    elif query_latent_semantics == 4:
        LS4(query_image_data, query_feature_model, dimredtech, k, K)
    else:
        print("Enter a valid latent semantics choice!!")
        
        
if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection1 = db["phase2trainingdataset"]
    collection1_name = "phase2trainingdataset"
    collection2 = db["labelrepresentativeimages"]
    collection2_name = "labelrepresentativeimages"

    caltech101_directory = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase1/data"
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
        
    print("Enter any of the Latent Semantics: \n1. LS1\n2. LS2\n3. LS3\n4. LS4\n")
    
    query_latent_semantics = int(input("Enter your choice number: "))
    
    K = int(input("Enter K value for finding K similar images: "))
    
    task7(query_image_id, query_image_file, query_latent_semantics, K)