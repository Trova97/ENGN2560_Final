# Alberto Trovamala and Cole Foster
# Brown University ENGN 2560
# Fall Semester 2020
# Professor Benjamin Kimia


# Libraries
import numpy as np
from sklearn.cluster import KMeans
import random
from scipy.spatial import Voronoi,cKDTree,voronoi_plot_2d
import matplotlib.pyplot as plt

def k_means_clusterer(data,k):
    km = KMeans(
        n_clusters=k, init='random',
        n_init=10, max_iter=1000, 
        tol=1e-05, random_state=0
    )
    y_km = km.fit_predict(data)
    return km, y_km

# generate the list of vectors in each region
def generate_region_data(unmarked_data,clustered_labels,K):
    num_points = len(clustered_labels)

    learning_vectors = []
    region_vectors = {}
    for i in range(0,K):
        region_vectors[i] = []

    # ad
    for i in range(num_points):
        region_vectors[clustered_labels[i]].append(unmarked_data[i])

    for i in range(0,K):
        for j in range(0,len(region_vectors[i])):
            learning_vectors.append([i,*region_vectors[i][j]])

    return learning_vectors



def region_finder(data,km):
    centers = km.cluster_centers_
    voronoi_kdtree = cKDTree(centers)
    point_dist, point_regions = voronoi_kdtree.query(data, k=1)
    return point_regions

def plot_clustered_data(data,y_km,km,dim1,dim2,k):
    red = np.linspace(0,1,k)
    gre = np.linspace(0,1,k)
    blu = np.linspace(0,1,k)
    random.shuffle(red)
    random.shuffle(gre)
    random.shuffle(blu)
    for dim in range(k):
        plt.scatter(
            data[y_km == dim, dim1], data[y_km == dim, dim2],
            s=50, color=np.array([red[dim],gre[dim],blu[dim]]),
            marker='s', edgecolor='black',
            label= f'cluster {dim+1}'
        )
    plt.scatter(
        km.cluster_centers_[:, dim1], km.cluster_centers_[:, dim2],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

def plot_voronoi_data(data,km):
    centers = km.cluster_centers_[:, 0:2]
    vor = Voronoi(centers)
    fig = voronoi_plot_2d(vor)
    plt.show()