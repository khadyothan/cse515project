import sys
sys.path.append('C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code')
import extracting_feature_space.color_moments as color_moments
import numpy as np
import extracting_feature_space.HOG as HOG
import extracting_feature_space.resnet_features as resnet_features
from PIL import Image

# Function to extract and store feature descriptors in a MongoDB collection
def feature_descriptors_extraction(collection, dataset):
    # Iterate over each image in the dataset
    for image_id, (image, label) in enumerate(dataset):  
        if image_id % 2 == 0:
        # Check if the image is grayscale and convert it to RGB if necessary
            if(len(np.array(image).shape) != 3):
                converted_img  = np.stack((np.array(image),) * 3, axis=-1)
                image = Image.fromarray(converted_img)
                
            # Extract color moments, HOG, and ResNet features for the image
            color_moments_feature_descriptor = color_moments.color_moments(image_id, image)
            hog_feature_descriptor = HOG.HOG(image_id, image)
            avgpool_feature_descriptor, layer3_feature_descriptor, fc_feature_descriptor = resnet_features.resnet_features(image_id, image)    
            
            # Insert the feature descriptors and related information into the MongoDB collection
            collection.insert_one({
                "image_id": image_id,
                "label": label,
                "color_moments_feature_descriptor": color_moments_feature_descriptor.tolist(), 
                "hog_feature_descriptor": hog_feature_descriptor.tolist(),
                "resnet50_layer3_feature_descriptor": layer3_feature_descriptor[0].view(1024, -1).mean(dim=1).tolist(),
                "resnet50_avgpool_feature_descriptor": avgpool_feature_descriptor[0].view(-1, 2).mean(dim=1).tolist(),
                "resnet50_fc_feature_descriptor": fc_feature_descriptor[0].squeeze().tolist(),
            })
            
            print(image_id)
