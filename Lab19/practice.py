#Exercise1
import numpy as np


# Let x1 = [3, 6], x2 = [10, 10].
# Use the above “Transform” function to transform these vectors to a higher dimension and
# compute the dot product in a higher dimension. Print the value.
def transform(x1, x2):
    return np.array([x1**2, x2**2, np.sqrt(2)*(x1 * x2)])


def main():
    x1 = [3, 6] #sample1
    x2 = [10, 10] #sample2

    phi_x1 = transform(x1[0],x1[1])
    phi_x2 =  transform(x2[0],x2[1])

    print(f"phi_x1:{phi_x1}")
    print(f"phi_x2:{phi_x2}")

    #compute the dot product
    print(f"Dot product in higher dimension:" ,np.dot(phi_x1,phi_x2))

if __name__ == '__main__':
    main()