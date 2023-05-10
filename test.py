import numpy as np
arr = np.array([0.00, 3.50, 3.50])

print(np.modf(arr)[0] != 0)
