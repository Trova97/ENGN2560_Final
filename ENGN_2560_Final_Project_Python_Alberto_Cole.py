# Alberto Trovamala and Cole Foster
# Brown University ENGN 2560
# Fall Semester 2020
# Professor Benjamin Kimia

# The goal of this code is to reproduce the algorithm detailed in
# "Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors"
# by Baranchuk, Babenko, and Malkov.

# These are the steps followed by the code:
# Preprocessing:
# 1- Access 2100 satellite images (UC_Merced dataset), 100 for each of 21 categories
# 2- Convert images to 256by256by3 matrices, discard the mishaped ones (44 bad ones)
# 3- For each image extract 256 SIFT features and add an image ID to each descriptor
# 4- Run k-means on the data to determine the centroids
# 5- For each descriptor determine the corresponding Voronoi region
# 6- "Flip" the list to get a dictionary that gives the descriptors for each region
# Search:
# 1- For given query (256by256by3 matrix), extract 953 SIFT features
# 2- For each feature perform a search and return a list of image IDs of results
# 3- Return images ranked by how many times their ID appeared


# Imports:
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import random
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from collections import Counter
import time

# Functions used for preprocessing:
# Preprocessing function:
def prepro():
    print("1- Obtaining images")
    array_4D, full_array = create_4D_array()
    print("2- Images arranged into 4D array")
    img_list, array_SIFT = create_2D_SIFT(array_4D)
    print("3- SIFT descriptors arranged")
    SIFT_unlabelled = array_SIFT[:,:128]
    centers = k_means_clusterer(SIFT_unlabelled,160)
    print("4- Centroids obtained")
    regions = region_marker(SIFT_unlabelled,centers)
    print("5- Voronoi regions determined")
    inv_ind = create_inverted_index(centers,array_SIFT,regions)
    print("6- Created inverted indeces dictionary")
    print("Preprocessing concluded!")
    return img_list, centers, inv_ind, full_array

# Preprocessing helper functions:
def create_4D_array():
    database = np.zeros((256,256,3,2100))
    ind = 0
    for filename in os.listdir("C:\\Users\\alber\\OneDrive\\Documenti\\Python Code\\ENGN 2560\\images"):
        image = Image.open(f"C:\\Users\\alber\\OneDrive\\Documenti\\Python Code\\ENGN 2560\\images\\{filename}")
        image.getdata()
        r, g, b = image.split()
        ra = np.array(r)
        ga = np.array(g)
        ba = np.array(b)
        if ra.shape != (256,256) or ga.shape != (256,256) or ba.shape != (256,256):
            continue
        database[:,:,0,ind] = ra
        database[:,:,1,ind] = ga
        database[:,:,2,ind] = ba
        ind += 1
    database = database[:,:,:,:ind]
    return database[:,:,:,0::20], database #TODO: write ind instead of 100

def create_2D_SIFT(data_4D):
    num_imgs = data_4D.shape[3]
    SIFT_2D = np.zeros((256*num_imgs,129))
    searchable_img = [0]*num_imgs
    for ind in range(num_imgs):
        arr1 = data_4D[:,:,:,ind].astype(np.uint8)
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(np.float32(arr1), cv2.COLOR_RGB2GRAY)
        image8bit = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints_sift, descriptors = sift.detectAndCompute(image8bit, None)
        if descriptors is None:
            continue
        if len(descriptors) < 256:
            continue
        searchable_img[ind] = ind
        SIFT_2D[ind*256:(ind+1)*256,:128] = descriptors[:256,:]
        SIFT_2D[ind*256:(ind+1)*256,128] = ind
    SIFT_2D = SIFT_2D[~np.all(SIFT_2D == 0, axis=1)]
    searchable_img = list(dict.fromkeys(searchable_img))
    return searchable_img, SIFT_2D

def create_databases():
    database_1 = create_4D_array()
    img_list, array_2D = create_2D_SIFT(database_1)
    return img_list, database_1, array_2D

def k_means_clusterer(SIFT_data,num_clusters):
    km = KMeans(
        n_clusters=num_clusters, init='k-means++',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    y = km.fit_predict(SIFT_data)
    return km.cluster_centers_

def region_marker(vectors,centers):
    voronoi_kdtree = cKDTree(centers)
    point_dist, point_regions = voronoi_kdtree.query(vectors, k=1)
    return point_regions

def create_inverted_index(centroid_list, vector_list, region_list):
    inverted_index = {}
    for ind in range(len(region_list)):
        key = tuple(centroid_list[region_list[ind],:])
        value = tuple(vector_list[ind,:])
        inverted_index.setdefault(key, [])
        inverted_index[key].append(value)
    return inverted_index

# Functions used for search:
# Search function:
def seacher(image_ID):
    if image_ID in image_list:
        print("The search has begun")
    else:
        print("Choose a correct ID")
        return None
    my_SIFT = get_SIFT(image_ID)
    print("1- Obtained SIFT descriptors")
    cls_list = get_list_closest(my_SIFT, centers)
    print("2- Obtained list of closest images")
    bst = most_frequent(cls_list)
    print("3- Found best match")
    return bst

# Search helper functions:
def get_SIFT(image_ID):
    image = full_data[:,:,:,image_ID].astype(np.uint8)
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY)
    image8bit = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    keypoints_sift, descriptors = sift.detectAndCompute(image8bit, None)
    return descriptors[:256,:]

def get_closest_centroid(vector,centroids):
    distances = np.ones((1,len(centroids)))
    ind = 0
    for centroid in centroids:
        distances[0,ind] = np.linalg.norm(vector-centroid)
        ind += 1
    low_dist = np.argmin(distances)
    return centroids[low_dist]

def get_closest_SIFT(vector,centroid):
    close_list = indexes[tuple(centroid)]
    close_list = [list(ele) for ele in close_list]
    distances = np.ones((1,len(close_list)))
    ind = 0
    for SIFT_vector in close_list:
        distances[0,ind] = np.linalg.norm(vector-SIFT_vector[:128])
        ind += 1
    low_dist = np.argmin(distances)
    return close_list[low_dist][128]


def get_list_closest(vectors,centroids):
    close_images = []
    for vector in vectors:
        centroid = get_closest_centroid(vector,centroids)
        closest_img = get_closest_SIFT(vector,centroid)
        close_images.append(closest_img)
    return close_images

def most_frequent(lst): 
    sorted_list= [item for items, c in Counter(lst).most_common() for item in [items] * c]
    return list(dict.fromkeys(sorted_list))



# Running the code:

# Preprocessing data:
image_list, centers, indexes, full_data = prepro()

# Select ID of image you want to search:
search_ID = 10

# Show the search query:
print("Here is what you searched:")
searched_img = full_data[:,:,:,search_ID]
searched_image = Image.fromarray(np.uint8(searched_img), "RGB")
plt.imshow(searched_image)
plt.show()

# Run the search function:
t_start = time.perf_counter() 
match_list = seacher(search_ID)
t_stop = time.perf_counter()
print("Elapsed time:", t_stop-t_start)  

# Show the best match:
print("Here is the best match:")
match_image = Image.fromarray(np.uint8(full_data[:,:,:,int(match_list[0])]), "RGB")
plt.imshow(match_image)
plt.show()

# Show more matches:
print("Here are other matches:")
for ind in match_list:
    print(f"Image ID: {int(ind)}")
    new_img = Image.fromarray(np.uint8(full_data[:,:,:,int(ind)]), "RGB")
    plt.imshow(new_img)
    plt.show()