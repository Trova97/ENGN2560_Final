# Alberto Trovamala and Cole Foster
# Brown University ENGN 2560
# Fall Semester 2020
# Professor Benjamin Kimia

# The goal of this code is to reproduce the algorithm detailed in
# "Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors"
# by Baranchuk, Babenko, and Malkov.

# This file includes functions to generate simil_SIFT datasets, perform k-means
# clustering, quantizing from Voronoi regions, saving data with inverted indexing,
# and plotting all the partial results and compare them to the ground truth.

#import libaries
import numpy as np
import time as time
from numpy import random

# import files
from create_datasets import create_centroids,create_dataset
from kmeans import k_means_clusterer
from regionClass import regionClass,addVectorsToRegions
from functions import computeEuclideanDistance


#===================================================
# Train and Search Functions
#===================================================

def train_dataset(learning_vectors,k,centroids_file,labels_file):
    start_time = time.time()

    # applying kmeans
    try:
        
        centroid_vectors = np.loadtxt(centroids_file,delimiter=',')
        if (len(centroid_vectors)!=k):
            print("Error with centroids file")
            return
        data_labels = np.loadtxt(labels_file,delimiter=",",dtype=int)
        print("Loaded centroids and labels")
    except:
        print('Applying K-Means to produce centroids')
        km, data_labels = k_means_clusterer(learning_vectors,k)
        centroid_vectors = km.cluster_centers_
        print(len(data_labels))
        np.savetxt(centroids_file,np.asarray(centroid_vectors),delimiter=',')
        np.savetxt(labels_file,np.asarray(data_labels),delimiter=",",fmt="%i")

    # creating vector of regionClass objects
    regionsVector = []
    for i in range(0,k):
        regionsVector.append(regionClass(k,i,centroid_vectors[i]))

    # adding vectors to respective region
    addVectorsToRegions(unmarked_data,data_labels,centroid_vectors,regionsVector)

    print('Training time: ', time.time() - start_time,'s')
    return regionsVector,centroid_vectors




# search for neighbors of each query
def search_ann(queries,regionsVector,learning_vectors,centroid_vectors,nc,nn,flag_quantization):
    search_time = 0
    num_queries = len(queries)
    query_neighbors = []

    for query in queries:
        start_time = time.time()

        # now, search the centoids for closest one 
        centroid_distances = []
        for i in range(0,k):
            distance_Q2c = computeEuclideanDistance(query,centroid_vectors[i])
            centroid_distances.append([i,distance_Q2c])
        centroid_distances.sort(key=lambda x:x[1])

        centroids_to_search = []
        for i in range(0,nc):
            centroids_to_search.append(centroid_distances[i][0])

        vectors_distances = []
        if flag_quantization:

            for i in centroids_to_search:
                distances = regionsVector[i].computeQuantizedDistancestoQuery(query,centroid_vectors)
                for j in distances:
                    vectors_distances.append(j)

        else:
            # now, create a list of all members within the centroids
            vectors_to_search = []
            for i in centroids_to_search:
                members_temp = regionsVector[i].member_ids
                for j in members_temp:
                    vectors_to_search.append(j)

            # compute distance to all of those vectors
            for i in vectors_to_search:
                distance_Q2v = computeEuclideanDistance(query,unmarked_data[i])
                vectors_distances.append([i,distance_Q2v])

        vectors_distances.sort(key=lambda x:x[1])
        
        # return closest nn vectors
        neighbors = []
        for i in range(0,nn):
            neighbors.append(vectors_distances[i][0])
        query_neighbors.append(neighbors)
        
        search_time += time.time() - start_time

    search_time /= num_queries
    print('Average search time: ', time.time() - start_time,'s')
    return query_neighbors


#===================================================
# Creating the Dataset if not imported
#===================================================
k = 100         # number of centroids to create
n = 128         # number of dimensions (SIFT)
v_min = 0       # min value of element 8-bit
v_max = 256     # max value of element 8-bit
sig = 3         # expansion factor of dataset on centroids

dataset_file = 'data/real_100k.csv'
try:
    print('Load dataset attempt')
    unmarked_data = np.loadtxt(dataset_file,delimiter=',')
    print('Load dataset successful')
except:
    print('Load dataset failed')

#===================================================
# Training and Search
#===================================================

# data paths
centroids_path = 'data/real_centroids_100k.csv'
labels_path = 'data/real_labels_100k.csv'

# Define dimensions:
k = 500         # number of centroids for means

# Search Parameters:
num_queries = 100           # number of queries to test for
nc = 16                     # number of nearest centroids to search through
quantization = True         # search by quantization values or true values

R_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096]  # number of nearest neighbors to return

# Training through k-means
regionsVector,centroid_vectors = train_dataset(unmarked_data,k,centroids_path,labels_path)

# Searching the datset
query_indices = list(random.randint(0,9999,num_queries))
queries = []
for i in query_indices:
    queries.append(unmarked_data[i])

for R in R_list:
    print("Searching for Recall @",R)
    q_neighbors = search_ann(queries,regionsVector,unmarked_data,centroid_vectors,nc,R,quantization)
    # now, we hopefully should have a match
    recalled = 0
    for j in range(0,num_queries):
        if query_indices[j] in q_neighbors[j]:
            recalled += 1
    print('Recall@%i = ' %(R),100*(recalled/num_queries),"%")
    print('')
    