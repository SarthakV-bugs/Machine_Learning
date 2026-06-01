#Kernel Trick implementation
import numpy as np


#define the data
def transform(x,y):
    x1,x2,x3 = x
    return(np.array[x1*x1,
           x1*x2,
           x1*x3,
           x2*x1,
           x2*x2,
           x2*x3,
           x3*x1,
           x3*x2,
           x3*x3
        ])

def kernel_trick(x,y):
    return ((np.dot(x,y))**2

def main():
    x = np.array([1,2,3])
    y =











x = [1,2,3]
y = [4,5,6]

