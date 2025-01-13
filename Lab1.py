from random import randint

import numpy as np
import matplotlib.pyplot as plt
from math import pi

from mpl_toolkits.axes_grid1 import host_subplot


#Exercise 1
def Ex1():
    A = np.array([[1,2,3],[4,5,6]])
    A_trans  = A.transpose()
    print(A_trans)
    print(A.dot(A_trans))

def Ex2():
    x = np.linspace(-100,100,100)
    print(x)
    y = 2*x + 3
    print(y)
    plt.figure(figsize=(8,6))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Y = 2x + 3")
    plt.plot(x,y,label="y = 2*x + 3", color="b")
    plt.legend()
    plt.show()

def Ex3():

    x = np.linspace(-10,10,100)
    y = 2*x**2 + 3*x + 4
    print(y)
    plt.figure(figsize=(8, 6))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot of y = 2x^2 + 3*x + 4")
    plt.plot(x, y, label="y = 2*x**2 + 3*x + 4", color="b")
    plt.legend()
    plt.show()

def Ex4():
    mu = 0
    sigma = 15
    x = np.linspace(-100,100,100)
    y = (np.exp(-0.5 * ((x - mu) / sigma)**2))/(sigma * (np.sqrt(2 * pi)) )
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Gaussian PDF", color="r")
    plt.legend()
    plt.show()

def Ex5():
    x = np.linspace(-100,100,100)
    y = x**2
    dydx = np.gradient(y,x)
    print(dydx)
    plt.plot(x,y,label="y=x^2",color="r")
    plt.plot(dydx,x,label="f(x)=2x", color="b")
    plt.legend()
    plt.show()

def Ex6():
        # x0 = 0
        def htax(d):

            theta = []
            x_d = []
            for _ in range(d):
                theta_in = int(input("Enter the theta value: "))
                theta.append(theta_in)
            for _ in range(d):
                x_d_in = int(input("Enter the x value: "))
                x_d.append(x_d_in)

            H_x = [(x*y) for x,y in zip(theta,x_d)]
            h_x_sum = sum(H_x)
            return h_x_sum

        d = int(input("Enter the number of features: "))
        n = int(input("Enter the number of samples: "))
        hes = []
        for _ in range(n):
            y = int(input("Enter the value of y: "))
            h = htax(d)
            hes.append((h-y)**2)
        E = 0.5*sum(hes)
        print(E)


def main():
    # Ex1()
    # Ex2()
    # Ex3()
    # Ex4()
    # Ex5()
    Ex6()
if __name__ == '__main__':
    main()