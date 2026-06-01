import numpy as np
from mpmath import inf

import numpy as np

A = []
for theta_0 in range(-500000, 6000000):
    for theta_1 in range(-500000, 6000000):
        for theta_2 in range(-5, 6):
            A.append([theta_0, theta_1, theta_2])

A = np.array(A)
print(np.min(A))
