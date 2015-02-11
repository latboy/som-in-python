"""
Kohonen SOM in python
"""
import numpy as np
import matplotlib.pyplot as plt


"""
Global parameters
"""
N = 20 # linear size of 2D map
n_teacher = 10000 # # of teacher signal
np.random.seed(100)# test seed for random number

def main():
    # initialize node vectors
    nodes = np.random.rand(N,N,3)# node array. each node has 3-dim weight vector
    #initial out put
    #TODO; make out put function to simplify here 
    plt.imshow(nodes, interpolation='none')
    plt.savefig("init.png")
    """"""
    """ Learning """
    """"""
    # teacher signal
    teachers = np.random.rand(n_teacher,3)
    for i in range(n_teacher):
        train(nodes, teachers, i)
        # intermediate out put
        if i%1000 ==0 or i< 100: #out put for i<100 or each 1000 iteration
            plt.imshow(nodes, interpolation='none')
            plt.savefig(str(i)+".png")
    #output
    plt.imshow(nodes, interpolation='none')
    plt.savefig("final.png")
    
def train(nodes, teachers, i):
    bmu = best_matching_unit(nodes, teachers[i])
    #print bmu
    for x in range(N):
        for y in range(N):
            c = np.array([x,y])# coordinate of unit
            d = np.linalg.norm(c-bmu)
            L = learning_ratio(i)
            S = learning_radius(i,d)
            for z in range(3): #TODO clear up using numpy function
                nodes[x,y,z] += L*S*(teachers[i,z] - nodes[x,y,z])



def best_matching_unit(nodes, teacher):
    #compute all norms (square)
    #TODO simplify using numpy function
    norms = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(3):
                norms[i,j] += (nodes[i,j,k] - teacher[k])**2
    #then, choose the minimum one
    bmu = np.argmin(norms) #argment with minimum element 
    # argmin returns just flatten, serial index, 
    # so convert it using unravel_index
    return np.unravel_index(bmu,(N,N))

def neighbourhood(t):#neighbourhood radious
    halflife = float(n_teacher/4) #for testing
    initial  = float(N/2)
    return initial*np.exp(-t/halflife)

def learning_ratio(t):
    halflife = float(n_teacher/4) #for testing
    initial  = 0.1
    return initial*np.exp(-t/halflife)

def learning_radius(t, d):
    # d is distance from BMU
    s = neighbourhood(t)
    return np.exp(-d**2/(2*s**2))

main()




