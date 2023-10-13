import random
import numpy as np
import pymongo

def initialize_centroids(data, k):
    centroids = random.sample(data.tolist(), k)
    return centroids

def initialize_centroids_kmeans_plus(data, k):
    centroids = [random.choice(data.tolist())]
    for _ in range(k - 1):
        distances = [min([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data]
        next_centroid = random.choices(data.tolist(), weights=distances, k=1)
        centroids.append(next_centroid[0])
    return centroids

def assign_to_clusters(data, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    for point in data:
        nearest_centroid_idx = min(range(len(centroids)), key=lambda i: abs(point - centroids[i]))
        clusters[nearest_centroid_idx].append(point)
    return clusters

def update_centroids(clusters):
    new_centroids = [sum(cluster) / len(cluster) if len(cluster) > 0 else 0 for cluster in clusters.values()]
    return new_centroids

def k_means(data, k, max_iterations, count):
    print(count)
    centroids = initialize_centroids(data, k)
    for i in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break  
        centroids = new_centroids
    return centroids

def kmeans(data_matrix, k):
    latent_semantics = []
    count = 0
    for row in data_matrix:
        cluster_centroids = k_means(row, k, 100, count)
        count+=1
        latent_semantics.append(cluster_centroids)
    data_matrix_ls = np.array(latent_semantics)

    return data_matrix_ls

def calculateImageIDWeightPairs(kmeans_ls, query_feature_model, k):
    
    return True
    
if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"
    
    data_matrix = np.loadtxt("C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\dimensionality_reduction\data_matrix_cm.csv", delimiter=',')     
    cm_ls = kmeans(data_matrix, 10)
    
    file_path_cm_ls = "C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\dimensionality_reduction\KMEANS\cm_ls"
    np.savetxt(file_path_cm_ls, cm_ls, delimiter=",")
    # calculateImageIDWeightPairs(file_path_cm_ls)
    