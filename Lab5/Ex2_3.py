import matplotlib.pyplot as plt
import numpy as np

#Ex2
#sigmoid function
def sigmoid_f(z):
    return 1 / (1 + np.exp(-z))

#Ex3
#derivative function
def derivative(g):
    return g * (1-g)

def sig_plot():

    z = np.linspace(-10, 10, 100)
    g = sigmoid_f(z)
    g_derivative = derivative(g)


    plt.figure(figsize=(8, 6))
    plt.plot(z, g, label="Sigmoid Function")
    plt.plot(z,g_derivative,label="Derivative of sigmoid function")
    plt.title("Sigmoid Function and it's derivative")
    plt.xlabel("z")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
  sig_plot()
