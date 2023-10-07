import torch
import torchvision.datasets as datasets
import numpy as np
import Code.extracting_feature_space.color_moments as color_moments
import Code.extracting_feature_space.HOG as HOG
import Code.extracting_feature_space.resnet_features as resnet_features

# Function to calculate L2 distance between two vectors
def l2_distance(input_vector, db_vector):
    return np.linalg.norm(np.array(input_vector) - np.array(db_vector))

# Function to calculate squared L2 distance between two vectors
def squared_l2_distance(input_vector, db_vector):
    return np.sqrt(np.sum(np.square(np.array(input_vector) - np.array(db_vector))))

# Function to calculate cosine similarity between two vectors
def cosine_similarity(input_vector, db_vector):
    dot_product = np.dot(input_vector, db_vector)
    norm_input_vector = np.linalg.norm(input_vector)
    norm_db_vector = np.linalg.norm(db_vector)
    return dot_product / (norm_input_vector * norm_db_vector)

# Function to perform Task 3: Finding top K most similar images from the database
def task3(input_image_id, image_data, k, collection, dataset):
    # Initialize dictionaries to store distances for each feature descriptor
    distance_cm_dict, distance_hog_dict, distance_avgpool_dict, distance_layer3_dict, distance_fc_dict = {}, {}, {}, {}, {}
    
    # Retrieve feature vectors for the input image
    input_cm_vector = color_moments.color_moments(input_image_id, image_data).tolist()
    input_hog_vector = HOG.HOG(input_image_id, image_data).tolist()
    input_avgpool_vector, input_layer3_vector, input_fc_vector = resnet_features.resnet_features(input_image_id, image_data)
    input_avgpool_vector =  input_avgpool_vector[0].view(-1, 2).mean(dim=1).tolist()
    input_layer3_vector =  input_layer3_vector[0].view(1024, -1).mean(dim=1).tolist()
    input_fc_vector =  input_fc_vector[0].squeeze().tolist()
    
    # input_cm_vector = collection.find_one({"image_id": input_image_id}, {"color_moments_feature_descriptor": 1})["color_moments_feature_descriptor"]
    # input_hog_vector = collection.find_one({"image_id": input_image_id}, {"hog_feature_descriptor": 1})["hog_feature_descriptor"]
    # input_avgpool_vector = collection.find_one({"image_id": input_image_id}, {"resnet50_avgpool_feature_descriptor": 1})["resnet50_avgpool_feature_descriptor"]
    # input_layer3_vector = collection.find_one({"image_id": input_image_id}, {"resnet50_layer3_feature_descriptor": 1})["resnet50_layer3_feature_descriptor"]
    # input_fc_vector = collection.find_one({"image_id": input_image_id}, {"resnet50_fc_feature_descriptor": 1})["resnet50_fc_feature_descriptor"]

    # Calculate distances for each image in the database
    for doc in collection.find():
        db_image_id = doc["image_id"]
        db_cm_vector = doc["color_moments_feature_descriptor"]
        db_hog_vector = doc["hog_feature_descriptor"]
        db_avgpool_vector = doc["resnet50_avgpool_feature_descriptor"]
        db_layer3_vector = doc["resnet50_layer3_feature_descriptor"]
        db_fc_vector = doc["resnet50_fc_feature_descriptor"]
        
        # Calculate L2 distance for each feature descriptor
        distance_cm_dict[db_image_id] = l2_distance(input_cm_vector, db_cm_vector)
        distance_hog_dict[db_image_id] = squared_l2_distance(input_hog_vector, db_hog_vector)
        distance_avgpool_dict[db_image_id] = squared_l2_distance(input_avgpool_vector, db_avgpool_vector)
        distance_layer3_dict[db_image_id] = squared_l2_distance(input_layer3_vector, db_layer3_vector)
        distance_fc_dict[db_image_id] = squared_l2_distance(input_fc_vector, db_fc_vector)
          
    # Sort the distances and select the top K images for each feature descriptor
    sorted_cm_distances = dict(sorted(distance_cm_dict.items(), key=lambda x: x[1]))
    top_k_cm_images = dict(list(sorted_cm_distances.items())[:k])
    
    sorted_hog_distances = dict(sorted(distance_hog_dict.items(), key=lambda x: x[1]))
    top_k_hog_images = dict(list(sorted_hog_distances.items())[:k])
    
    sorted_avgpool_distances = dict(sorted(distance_avgpool_dict.items(), key=lambda x: x[1]))
    top_k_avgpool_images = dict(list(sorted_avgpool_distances.items())[:k])
    
    sorted_layer3_distances = dict(sorted(distance_layer3_dict.items(), key=lambda x: x[1]))
    top_k_layer3_images = dict(list(sorted_layer3_distances.items())[:k])
    
    sorted_fc_distances = dict(sorted(distance_fc_dict.items(), key=lambda x: x[1]))
    top_k_fc_images = dict(list(sorted_fc_distances.items())[:k])
    
    # Create dictionaries to store top K images with their distances
    images_cm_display = {image_id: {'image': image, 'distance': top_k_cm_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_cm_images}
    images_hog_display = {image_id: {'image': image, 'distance': top_k_hog_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_hog_images}
    images_layer3_display = {image_id: {'image': image, 'distance': top_k_layer3_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_layer3_images}
    images_avgpool_display = {image_id: {'image': image, 'distance': top_k_avgpool_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_avgpool_images}
    images_fc_display = {image_id: {'image': image, 'distance': top_k_fc_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_fc_images}

    # Return the dictionaries with top K images and their distances
    return images_cm_display, images_hog_display, images_layer3_display, images_avgpool_display, images_fc_display
