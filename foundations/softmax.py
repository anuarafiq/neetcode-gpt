import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        softmax = np.empty(0)
        z -= max(z)
        for num in z:
            softmax = np.append(softmax, np.exp(num))
        
        softmax /= sum(softmax)
        return np.round(softmax, decimals=4)