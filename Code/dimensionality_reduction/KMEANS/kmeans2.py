import numpy as np
import pymongo

import numpy as np
import os
import random

from sklearn.cluster import KMeans

def cluster_calc(x: list, cent_no: int):
    
    ##### using kmeans from sklearn
    # res = KMeans(
    #     n_clusters= cent_no,
    #     random_state= 0,
    #     n_init='auto'
    # ).fit(x)
    
    # return res.cluster_centers_, res
    
    ##### from scratch
    cent_temp = centroids = initialize_kmeans_plusplus(np.array(x), cent_no).tolist()
    # cent_temp = centroids = random.sample(x, cent_no)
    cluster_array = [[] for i in range(cent_no)]
    repeat_flag = True
    
    while repeat_flag:
        cluster_array = [[] for i in range(cent_no)]
        for i in x:
            min_dist, min_j = vector_euclid(i, centroids[0]), 0
            
            for j in range(len(centroids)):
                temp_dist = vector_euclid(i, centroids[j])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_j = j

            cluster_array[min_j].append(i)
            # print(f'DP{i}:\n','centroids:',centroids,'\nclusters array:\n', cluster_array)
            
            
        centroids = [np.mean(i, axis=0).tolist() for i in cluster_array]
        
        if centroid_check(cent_temp, centroids):
            repeat_flag = False

        cent_temp = centroids
        
    return centroids, cluster_array
    

def initialize_kmeans_plusplus(points, K, c_dist=np.linalg.norm):
    centers = []
    centers.append(points[np.random.randint(points.shape[0])])

    for _ in range(1, K):
        dist_sq = np.array([min([c_dist(c - x)**2 for c in centers]) for x in points])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centers.append(points[i])

    return np.array(centers)


def vector_euclid(x, y):
    # dumsum = 0
    # for i,j in zip(x, y):
    #     dumsum += (i - j)**2
    # return np.sqrt(dumsum)
    
    return np.linalg.norm(np.array(x) - np.array(y))

def centroid_check(prev, new):
    # flag = False
    
    # for i,j in zip(prev,new):
    #     if not i==j:
    #         flag= True
    #         break
        
    # return flag
    
    return all(np.allclose(p, n) for p, n in zip(prev, new))

def store_ls_kmeans(x: list) -> None:
    x = np.array(x)
    
    store_path = os.path.join(os.getcwd(),'Outputs/LatentSemantics/latent_semantics_kmeans.csv')
    np.savetxt(store_path, x, delimiter=',')

def kmeans2(data_matrix, k):
    centroids, cluster_array = cluster_calc(data_matrix, k)
    print(f'shape of centroids: {np.array(centroids).shape}')
    data_matrix_ls = []
    for i in data_matrix:
        temp = []
        for j in centroids:
            temp.append(vector_euclid(i, j))
            # print(temp)
        data_matrix_ls.append(temp)
        # print(res)
    # store_ls_kmeans(data_matrix_ls)
    return np.array(data_matrix_ls)

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"
    
    data_matrix = np.loadtxt("C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\Code\dimensionality_reduction\data_matrix_cm.csv", delimiter=',')     
    cm_ls = kmeans2(data_matrix, 10)
    
    file_path_cm_ls = "/Users/rohitbathi/Desktop/Masters/CSE_MWD/project/cse515project-master/Outputs/kmeans/cm_ls.csv"
    np.savetxt(file_path_cm_ls, cm_ls, delimiter=",")
    # calculateImageIDWeightPairs(file_path_cm_ls)