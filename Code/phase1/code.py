# Import necessary libraries and modules
import torch
import torchvision.datasets as datasets
import pymongo
import Code.phase1.task3 as task3
import Code.phase1.print_top_k_images as print_top_k_images
import Code.phase1.feature_descriptors_extraction as feature_descriptors_extraction
import Code.phase1.task1_printing as task1_printing
import numpy as np

# Connect to MongoDB
cl = pymongo.MongoClient("mongodb://localhost:27017")
db = cl["caltech101db"]
collection = db["phase2trainingdataset"]

# Set the directory for the Caltech101 dataset
caltech101_directory = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase1/data"

# Create a DataLoader for the Caltech101 dataset
dataset = datasets.Caltech101(caltech101_directory, download=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

# Function to retrieve feature descriptor for a given image using a specific feature model
def task1_program(image_id, feature_model):
    feature_model_descriptor = feature_model + "_feature_descriptor"
    return collection.find_one({"image_id": int(image_id)}).get(feature_model_descriptor)

# Function to perform Task 2: Extract and store feature descriptors for all images in MongoDB
def task2_program():
    if "phase2trainingdataset" not in db.list_collection_names():
        print("\nFeatures are being extracted and stored in the database. Wait patiently.\n")
        feature_descriptors_extraction.feature_descriptors_extraction(collection, dataset)
        print("\nAll the feature descriptors of all the images are stored in MongoDB database in collection called caltech101collection !!\n")
    else:
        print("The database already has all the feature descriptors of all the images stored!!")

# Function to perform Task 3: Finding top K most similar images from the database of an input image given its id and K value
def task3_program(image_id, image_data, k):
    return task3.task3(int(image_id), image_data, int(k), collection, dataset)

# Main program
def main():
    print("Welcome to Phase1 of the Project! Select one among the following by giving a number input only.\n")
    print("1. Task1 - Feature descriptor extraction of an input image\n")
    print("2. Task2 - Extract and store feature descriptors for all the images in MongoDB.\n")
    print("3. Task3 - Finding top K most similar images from the database of an input image given its id and K value.\n\n")

    # Get the user's task choice
    task_input = input("Enter the task number: ")

    if task_input == "1":
        # Task 1: Feature descriptor extraction
        input_image_id = input("\nWelcome to Task1:\nEnter image_id: ")
        print("\nSelect input feature model(Select one among these): ")
        print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
        input_feature_model = input("Enter input ")
        task1_output = task1_program(input_image_id, input_feature_model)
        print(f"\nThe feature model descriptor of {input_feature_model} is: \n\n") 
        task1_printing.readable_output(task1_output, input_feature_model)
    elif task_input == "2":
        # Task 2: Extract and store feature descriptors
        task2_program()
    elif task_input == "3":            
        # Task 3: Finding top K most similar images
        input_image_id = input("\nWelcome to Task3:\nEnter image_id: ")
        input_k = input("\nEnter the value of k: ")
        image_data = None
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(input_image_id):
                image_data = image
                images_to_display = [x for x in task3_program(input_image_id, image_data, input_k)]
                headings = ["Color Moments", "HOG", "Layer 3", "Avg Pool Layer", "FC Layer"]
                for i, heading in zip(images_to_display, headings):
                    print_top_k_images.print_images(i, heading)
                break
    else:
        print("\nEnter valid number.")

# Execute the main program if this script is run
if __name__ == "__main__":
    main()
