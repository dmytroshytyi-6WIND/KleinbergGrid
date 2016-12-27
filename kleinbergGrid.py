import numpy as np
import numpy.matlib as ml
from scipy.spatial import distance as dst
import operator

def euclidianCartesian(fi,distance):
    x=distance*np.cos(fi)
    y=distance*np.sin(fi)
    return x,y

def polarGetOffsets(dst):
    #polar coordinates
    #we have 4*i nodes on destination i with uniform distribution because of the same distance .
    numberOfNodes=dst*4
    #print ('nodes: ', numberOfNodes)
    i=360/numberOfNodes
    #print ('fi',fi)
    offsets=np.arange(0,360,fi)
    #print ('offsets', offsets)
              
return offsets


#normalize probs
def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]

def getDistance(r,n):
    #assign probabilities(probs) to distances
    probs=np.fromfunction(lambda distance: 1/(distance**r), (n,))
    #delete prob for element with disatnce 0
    probs=np.delete(probs, 0, 0)
    #normalize probs(after calculation of sum of probabilities for all distances should be 1)
    probs=normalize(probs)
    #print (probs)
    #choose distance based on probabilities according to the (1/distance^r)
    distance=np.random.choice(np.arange(1,n),p=probs)    
    return distance


#get all possible shortcuts for distance "dst".
#dst=distance
def getAllOffsets(dst):
    offsets = list();
    #we have 4*i nodes on destination i with uniform distribution because of the same distance .
    numberOfNodes=dst*4
    x=np.arange(-dst-1,dst+1)
    y=np.arange(-dst-1,dst+1)
    for i in x:
        for j in y:
            if (abs(i)+abs(j))==dst:
                offsets.append((i,j))
    return offsets


#pefrorm check and filter only the sum of (shortcut and offset) coordinates fits the original matrix.  
def rejection(allOffsets,k,nodePosition,matrixSize):
    import random
    offsets = list();
    for coords in allOffsets:
        #sum initial node and offstet 
        offsetNodeCoords=tuple(map(operator.add, coords, nodePosition))
        #check if summ of (initial node and shortcut offset) fits in the initial matrix
        if 0 <= offsetNodeCoords[0] <= matrixSize[0] and  0 <= offsetNodeCoords[1] <= matrixSize[1]:
            #if summ fits in matrix, add the shourtcut offset to the list 
            #offsets.append(coords)
            offsets.append(offsetNodeCoords)
        #exit the cycle because of shortcut number limit in main function.
        #if (len(offsets) >= k):
        #    break
    offsets=random.sample(offsets, k)
    return offsets    
    
def dist(u,v):
    #print (u,v)
    #print (u[0],v[0],u[1],v[1])
    return (abs(u[0]-v[0])+abs(u[1]-v[1]))

#k = number of offsets
#r = magic value :D
#n = matrix size
def make_shortcut_offset(k,r,n,nodePos):
    #fit the sircle such that there always could be a solution when trying to find an offset.
    
    #radius=dist((n-nodePos[0],n-nodePos[1]) ,nodePos)
    radius=dist((n,n) ,nodePos)
    #print (nodePos)
    #radius=2*(n-1)
    #choose distance from [0,radius] according to the probability (1/distance^r)
    distance=getDistance(r,radius)   
    #print('dst:',distance)
    #allOffsets=set of "fi" variables= degree from 0-360 in eucalidian space
    allOffsets=getAllOffsets(distance)
    #print('allOffsets',allOffsets)
    offsets=rejection(allOffsets,k,nodePos,(n,n))
    #print('resultingOffsets: ', offsets)
    return offsets


def shortestDistance(u,v,shortcut):
    #tuple(map(tuple, nodePos))
    x=0
    y=1
    #print('u:', u)
    uR=[u[x]+1,u[y]]
    uL=[u[x]-1,u[y]]
    uU=[u[x],u[y]+1]
    uD=[u[x],u[y]-1]
    
    
    neigh=[uR,uL,uU,uD,shortcut]
    direct=[dist(uR,v),dist(uL,v),dist(uU,v),dist(uD,v),dist(shortcut,v)]
    total=zip(neigh,direct)
    
    #print('neigh: ',neigh)
    minimum=min(direct)
    for i,j in total:
        if j == minimum:
            #print ('minimum: ',i)
            return i
            
def kleinberg_distance(runs,n,r):
    randNodeU=(np.random.choice(n, 2))
    randNodeV=(np.random.choice(n, 2))
    #print(randNodeU)
    #print(randNodeV)
    #randNodeU=[4,4]
    #randNodeV=[55,59]
    #number of shortcuts==1
    k=1
    closestNode=randNodeU
    steps=0
    #per each step of this algorithm we should be closer and closer to node randNodeV(finish node).
    while set(randNodeV) != set(closestNode):
        #print ('closestNode: ',closestNode)
        #calculate shortcut(
        #shortcut=make_shortcut_offset(k,r,n, nodePos=tuple(randNodeU))
        shortcut=make_shortcut_offset(k,r,n, nodePos=tuple(closestNode))
        #print ('shortcutRAW',shortcut)
        shortcut=list(chain(*shortcut))
        #print ('shortcut',shortcut)
        closestNode=shortestDistance(closestNode,randNodeV,shortcut)
        #print(closestNode)
        steps+=1
    #if dist(randNodeU,randNodeV) != cnt:
        #print ("shortcut choosed")
    return steps

