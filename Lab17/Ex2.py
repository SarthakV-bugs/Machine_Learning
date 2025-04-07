# 2. Implement a polynomial kernel K(a,b) =  a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2
# Apply this kernel function and evaluate the output for the same x1 and x2 values.
# Notice that the result is the same in both scenarios demonstrating the power of kernel trick
import numpy as np


def poly_transform(a,b):
    return a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2

def main():
    x1 = [3, 6]  # sample1
    x2 = [10, 10]  # sample2


    print(f"poly_transform(x1,x2): {poly_transform(x1,x2)}")
    #result is the same in both scenarios i.e. custom transformation function and
    # kernel method employed here, demonstrating the power of kernel trick.

    # kernel function works as a dot product in higher-dimensional space.



if __name__ == '__main__':
    main()




