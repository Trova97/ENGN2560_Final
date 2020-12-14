# Alberto Trovamala and Cole Foster
# Brown University ENGN 2560
# Fall Semester 2020
# Professor Benjamin Kimia

# Libraries
import numpy as np
import random

def create_centroids(n,k,v_min,v_max):
    centroids = np.random.rand(k,n)
    centroids = centroids * (v_max - v_min) + v_min
    return centroids

def create_dataset(centroids,num_centroids,len_point,num_points,sigma,v_min,v_max):
    points_per_cluster = np.random.normal(1.0, 0.5, np.shape(centroids)[0]) 
    points_per_cluster += -min(points_per_cluster)+1
    points_per_cluster *= num_points/sum(points_per_cluster)
    points_per_cluster = [int(round(num)) for num in points_per_cluster]
    data = np.zeros((sum(points_per_cluster),len_point+1),dtype=int)
    current_index = 0
    for centroid_index in range(num_centroids):
        new_data = np.random.normal(centroids[centroid_index,:],sigma,(points_per_cluster[centroid_index],len_point))
        data[current_index:current_index+points_per_cluster[centroid_index],0] = centroid_index
        data[current_index:current_index+points_per_cluster[centroid_index],1:] = new_data
        current_index += points_per_cluster[centroid_index]
    data[np.where(data > v_max)] = v_max
    data[np.where(data < v_min)] = v_min
    data.round
    return data
