# Alberto Trovamala and Cole Foster
# Brown University ENGN 2560
# Fall Semester 2020
# Professor Benjamin Kimia


# Libraries
import numpy as np
import matplotlib.pyplot as plt


def plot_2D_data(data,dim1,dim2):
    plt.scatter(
        data[:, dim1], data[:, dim2],
        c='white', marker='o',
        edgecolor='black', s=50
    )
    plt.show()

def computeEuclideanDistance(v1,v2):
    distance = 0
    for i in range(0,len(v1)):
        distance += (v2[i]-v1[i])**2
    distance2 = np.sqrt(distance)
    return distance2

def computeDistanceNoSqrt(v1,v2):
    distance = 0
    for i in range(0,len(v1)):
        distance += (v2[i]-v1[i])**2
    return distance

# multiply elements of vector v1 by scalar a
def scalarMultiply(v1,a):
    for i in range(len(v1)):
        v1[i] *= a
    return v1

# add two vectors v1+v2
def addVectors(v1,v2):
    resultant = []
    for i in range(len(v1)):
        resultant.append(v1[i] + v2[i])
    return resultant

# subtract two vectors v1-v2
def subtractVectors(v1,v2):
    resultant = []
    for i in range(len(v1)):
        resultant.append(v1[i] - v2[i])
    return resultant

# dot product of two vectors, v1_T * v2
def dotVectors(v1,v2):
    resultant = 0
    for i in range(len(v1)):
        resultant += v1[i]*v2[i]
    return resultant

# take euclidean norm of a vector
def normVector(v1):
    resultant = 0
    for i in range(len(v1)):
        resultant += v1[i]*v1[i]
    return np.sqrt(resultant)

