# Alberto Trovamala and Cole Foster
# Brown University ENGN 2560
# Fall Semester 2020
# Professor Benjamin Kimia

from functions import computeEuclideanDistance as computeDistance
from functions import subtractVectors,normVector,addVectors,scalarMultiply,dotVectors,computeDistanceNoSqrt

class regionClass:
    def __init__(self,num_centroids,centroid_id,centroid_vector):
        # on initialization
        self.num_centroids = num_centroids
        self.centroid_id = centroid_id
        self.centroid_vector = centroid_vector
        # without grouping
        self.num_members = 0
        self.member_ids = []
        self.member_distances = {}
        # grouping
        self.flag_grouping = 0          # 0 for non-grouping, 1 for grouping. decides what member_distances means
        self.num_subcentroids = 0
        self.subcentroid_ids = []
        self.alpha = 0
        self.subcentroid_members = {}

    def add_members(self,member_id,distance_to_centroid):
        self.num_members += 1
        self.member_ids.append(member_id)
        self.member_distances[member_id] = distance_to_centroid

    def add_subcentroids(self,num_subcentroids,neighbor_ids):
        self.flag_grouping = 1
        self.num_subcentroids = num_subcentroids
        self.subcentroid_ids = neighbor_ids
        for i in neighbor_ids:
            self.subcentroid_members[i] = []
    
    def addAlpha(self,alpha):
        self.alpha = alpha

    def addMemberToSubcentroid(self,subcentroid_id,member_id,distance):
        self.num_members += 1
        self.subcentroid_members[subcentroid_id].append(member_id)
        self.member_distances[member_id] = distance

    # learn Alpha on the region
    def learnAlpha(self,unmarked_data,centroids):
        c = centroids[self.centroid_id]

        # for each vector in region, compute:
        # val = norm( x_i - c - [  ((x_i - c)T*(sl_i - c))/||sl_i - c||^2  ]*(sl_i - c) )^2
        # sl_i* = argmin based (val)
        sl_i_star_list = {}

        for i in self.member_ids:
            x_i = unmarked_data[i]

            # initializing values
            sl_val_min = -1
            sl_i_star = []

            # for each other centroid
            for j in self.subcentroid_ids:
                if (j != self.centroid_id):
                    sl_i = centroids[j]

                    term1 = subtractVectors(x_i,c)
                    term2 = dotVectors(subtractVectors(x_i,c),subtractVectors(sl_i,c))/computeDistance(sl_i,c)
                    term3 = subtractVectors(sl_i,c)
                    term4 = subtractVectors(term1,scalarMultiply(term3,term2))
                    sl_val = normVector(term4)**2

                    if (sl_val_min == -1):
                        sl_val_min = sl_val
                        sl_i_star = sl_i
                    else:
                        if (sl_val < sl_val_min):
                            sl_val_min = sl_val
                            sl_i_star = sl_i

            sl_i_star_list[i] = j
        
        # now, compute alpha for the layer
        # alpha = [ sum_overallvectors( (x_i - c)_T * (sli* - c) ) / sum_overallvectors( norm(sli* - c) ) ]

        alpha_numerator = 0
        alpha_denominator = 0

        # sum up num/den for alpha
        for i in self.member_ids:
            x_i = unmarked_data[i]
            sl_i_star = centroids[sl_i_star_list[i]]

            top = dotVectors(subtractVectors(x_i,c),subtractVectors(sl_i_star,c))
            bottom = (computeDistanceNoSqrt(sl_i_star,c))

            alpha_numerator += abs(top)
            alpha_denominator += bottom

        # compute alpha and save to class
        alpha = alpha_numerator/alpha_denominator
        self.alpha = alpha
        print(alpha)
        return alpha

    # apply grouping to the region
    def applyGrouping(self,unmarked_data,centroids):
        c = centroids[self.centroid_id]
        sub_members = []

        for i in self.member_ids:
            x_i = unmarked_data[i]

            # find closest subcentroid
            dist_min = -1
            closest_subcentroid = -1

            for j in self.subcentroid_ids:
                distance_temp = computeDistanceToSubcentroid(x_i,c,centroids[j],self.alpha)

                if (dist_min == -1):
                    dist_min = distance_temp
                    closest_subcentroid = j
                else:
                    if (distance_temp < dist_min):
                        dist_min = distance_temp
                        closest_subcentroid = j
            
            self.addMemberToSubcentroid(closest_subcentroid,i,dist_min)
            sub_members.append([i,closest_subcentroid,dist_min])
        return sub_members

    def loadGroupingThroughFile(self,sub_members):
        for member,sub,dist in sub_members:
            self.addMemberToSubcentroid(int(sub),int(member),dist)

    def returnMembersofSubregions(self,centroids,flag_pruning,list_subcentroids):
        if not flag_pruning:
            list_subcentroids = self.subcentroid_ids
        members_list = []
        for i in list_subcentroids:
            sub_members = self.subcentroid_members[i]
            for j in sub_members:
                members_list.append(j)
        
        return members_list

    def computeGroupingQuantizedDistancestoQuery(self,query_vector,centroids,flag_pruning,list_subcentroids):
        if (self.flag_grouping != 1):
            print('uh oh')
            return

        if not flag_pruning:
            list_subcentroids = self.subcentroid_ids

        distances = []
        for i in list_subcentroids:
            distance_Q2s = computeDistanceToSubcentroid(query_vector,centroids[self.centroid_id],centroids[i],self.alpha)

            for j in self.subcentroid_members[i]:
                distance_sub2member = self.member_distances[j]
                distances.append([j,distance_Q2s+distance_sub2member])

        distances.sort(key=lambda x:x[1]) 
        return distances

    # for grouping and pruning
    def computeGroupingQuantizedDistancestoQueryPruning(self,query_vector,centroids,flag_pruning,list_subcentroids):
        if (self.flag_grouping != 1):
            print('uh oh')
            return

        sub_distances= {}
        for j in list_subcentroids:
            sub_distances[j[0]] = j[1]

        distances = []
        for i in sub_distances:
            distance_Q2s = sub_distances[i]

            for j in self.subcentroid_members[i]:
                distance_sub2member = self.member_distances[j]
                distances.append([j,distance_Q2s+distance_sub2member])

        distances.sort(key=lambda x:x[1]) 
        return distances
    
    # for 
    def computeQuantizedDistancestoQuery(self,query_vector,centroids):
        distances = []
        for i in self.member_ids:
            quant_dist = self.member_distances[i] + computeDistanceNoSqrt(query_vector,centroids[self.centroid_id])
            distances.append([i,quant_dist])
        return distances

    def computeDistancesToSubcentroid(self,query_vector,centroids):
        distances = []
        for i in self.subcentroid_ids:
            distance_Q2s = computeDistanceToSubcentroid(query_vector,centroids[self.centroid_id],centroids[i],self.alpha)
            distances.append([self.centroid_id,i,distance_Q2s])
        distances.sort(key=lambda x:x[2]) 
        return distances
    


            
# assign vertors to their appropriate region
def addVectorsToRegions(unmarked_data,data_labels,centroids,regionsVector):
    for i in range(0,len(data_labels)):
        distance = computeDistance(unmarked_data[i],centroids[data_labels[i]])
        regionsVector[data_labels[i]].add_members(i,distance)

# compute distances between centroids, assign L neighbors to each class
def addSubcentroidsToRegions(centroids,k,L,regionsVector):
    centroid_distances = {}
    for i in range(0,k):
        centroid_distances[i] = []
    for i in range(0,k):
        for j in range(i,k):
            if (i != j):
                distance_temp = computeDistance(centroids[i],centroids[j])
                centroid_distances[i].append([j,distance_temp])
                centroid_distances[j].append([i,distance_temp])
    for i in range(0,k):
        centroid_distances[i].sort(key=lambda x:x[1])
        neighbor_ids = []
        for j in centroid_distances[i]:
            neighbor_ids.append(j[0])
        regionsVector[i].add_subcentroids(L,neighbor_ids)
        
# distance as || x_i - (c + alpha*(sl - c)) ||
def computeDistanceToSubcentroid(exemplar_vector,centroid_vector,neighbor_centroid_vector,alpha):
    term1 = subtractVectors(neighbor_centroid_vector,centroid_vector)
    term2 = scalarMultiply(term1,alpha)
    term3 = addVectors(centroid_vector,term2)
    term4 = subtractVectors(exemplar_vector,term3)
    term5 = normVector(term4)
    return term5


    