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
from numpy import random
import time as time

# import files
from create_datasets import create_centroids,create_dataset
from kmeans import k_means_clusterer
from regionClass import regionClass,addVectorsToRegions,addSubcentroidsToRegions
from functions import computeEuclideanDistance


#===================================================
# Train and Search Functions
#===================================================

def train_dataset(learning_vectors,k,L,centroids_file,labels_file,alphas_file,subcentroids_members_file):
    #print('Applying K-Means to Produce Centroids')
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
        km, data_labels = k_means_clusterer(learning_vectors,100)
        centroid_vectors = km.cluster_centers_
        np.savetxt(centroids_file,np.asarray(centroid_vectors),delimiter=',',fmt="%f")
        np.savetxt(labels_file,np.asarray(data_labels),delimiter=",",fmt="%i")

    # creating vector of regionClass objects
    regionsVector = []
    for i in range(0,k):
        regionsVector.append(regionClass(k,i,centroid_vectors[i]))

    # adding vectors to respective region
    print("Assigning vectors to appropriate regions")
    addVectorsToRegions(unmarked_data,data_labels,centroid_vectors,regionsVector)

    print("Computing centroid distances and assigning %i subcentroids" %(L))
    addSubcentroidsToRegions(centroid_vectors,k,L,regionsVector)

    # Generating optimized alphas
    try:
        alphas = np.loadtxt(alphas_file,delimiter=',')
        if (len(alphas)!=k):
            print("Error with alphas file")
            return
        for i in range(0,k):
            regionsVector[i].addAlpha(alphas[i])
        print("Loaded optimized alphas file")
    except:
        print("Optimizing alpha for each region")
        alphas = []
        for i in range(0,k):
            alphas.append(regionsVector[i].learnAlpha(unmarked_data,centroid_vectors))
        np.savetxt(alphas_file,np.asarray(alphas),delimiter=',',fmt="%f")

    print("Assigning vectors to subcentroids for each region")
    try:
        subcentroid_members = np.loadtxt(subcentroids_members_file,delimiter=',')
        index_start = 0
        for i in range(0,k):
            index_end = index_start + regionsVector[i].num_members -1
            regionsVector[i].loadGroupingThroughFile(subcentroid_members[index_start:index_end])
            index_start = index_end + 1
        print("Loaded subcentroid members")
    except:
        print("Computing members for subcentroids")
        subcentroid_members = []
        for i in range(0,k):
            sub_mem = regionsVector[i].applyGrouping(unmarked_data,centroid_vectors)
            for j in sub_mem:
                subcentroid_members.append(j)
        np.savetxt(subcentroids_members_file,np.asarray(subcentroid_members),delimiter=',')

    print('Training time: ', time.time() - start_time,'s')
    return regionsVector,centroid_vectors


# search for neighbors of each query
def search_ann(queries,regionsVector,learning_vectors,centroid_vectors,nc,nn,flag_pruning,flag_quantization):
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

        map_centroids_to_subcentroids = {}

        if flag_pruning:
            for i in centroids_to_search:
                map_centroids_to_subcentroids[i] = []

            subcentroid_distances = []
            for i in centroids_to_search:
                distances_temp = regionsVector[i].computeDistancesToSubcentroid(query,centroid_vectors)
                for j in distances_temp:
                    subcentroid_distances.append(j)
            subcentroid_distances.sort(key=lambda x:x[2])
            # tau as 0.5, ratio of subcentroids to prune, L as subcentroids, nc as number of regions to visit
            num_subcentroids_to_keep = round(0.5*L*nc)
            subcentroid_list = subcentroid_distances[0:num_subcentroids_to_keep]

            for i in subcentroid_list:
                map_centroids_to_subcentroids[i[0]].append([i[1],i[2]])

        else:
            for i in centroids_to_search:
                map_centroids_to_subcentroids[i] = []
        
        # compute distances from query to each member of each subcentroid
        vectors_distances = []
        if (flag_quantization):
            if flag_pruning:
                for i in centroids_to_search:
                    distances = regionsVector[i].computeGroupingQuantizedDistancestoQueryPruning(query,centroid_vectors,flag_pruning,map_centroids_to_subcentroids[i])
                    for j in distances:
                        vectors_distances.append(j)
            else:
                for i in centroids_to_search:
                    distances = regionsVector[i].computeGroupingQuantizedDistancestoQuery(query,centroid_vectors,flag_pruning,map_centroids_to_subcentroids[i])
                    for j in distances:
                        vectors_distances.append(j)
        else:

            # getting members of subregions (if pruning, only the specified subregions)
            vectors_to_search = []
            for i in centroids_to_search:
                members_temp = regionsVector[i].returnMembersofSubregions(centroid_vectors,flag_pruning,map_centroids_to_subcentroids[i])
                for j in members_temp:
                    vectors_to_search.append(j)

            # for each subcentroid, compute true distance to each member
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
    #print('Recall for %i neighbors'%(nn))
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
    print('Loading dataset %s'%(dataset_file))
    unmarked_data = np.loadtxt(dataset_file,delimiter=',')
except:
    print('Dataset load failed')

#===================================================
# Training and Search
#===================================================

# files
centroids_path = 'data/real_centroids_100k.csv'
labels_path = 'data/real_labels_100k.csv'
alphas_path = 'data/alpha_values_100k.csv'
subcentroids_members_path = 'data/sub_members_100k.csv'

# if applying pruning
pruning = True
quantization = True

# Define training:
k = 500         # number of centroids for means
L = 16          # number of subcentroids per region for grouping

# Search Parameters:
nc = 16                  # number of nearest centroids to search through
num_queries = 100       

R_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096]  # number of nearest neighbors to return


# Training through k-means
regionsVector,centroid_vectors = train_dataset(unmarked_data,k,L,centroids_path,labels_path,alphas_path,subcentroids_members_path)

# Searching the datset
query_indices = list(random.randint(0,9999,num_queries))
queries = []
for i in query_indices:
    queries.append(unmarked_data[i])

for R in R_list:
    print("Searching for Recall @",R)
    q_neighbors = search_ann(queries,regionsVector,unmarked_data,centroid_vectors,nc,R,pruning,quantization)
    # now, we hopefully should have a match
    recalled = 0
    for j in range(0,num_queries):
        if query_indices[j] in q_neighbors[j]:
            recalled += 1
    print('Recall@%i = ' %(R),100*(recalled/num_queries),"%")
    print('')
    