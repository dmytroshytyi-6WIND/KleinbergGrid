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
    print ('nodes: ', numberOfNodes)
    i=360/numberOfNodes
    print ('fi',fi)
    offsets=np.arange(0,360,fi)
    print ('offsets', offsets)
              
return offsets

