import numpy as np

# This class has the following methods:
# gradient_descendent: calculates the gradient of a function f(x) = 2x
# gradient_descendent_ssr: calculates the gradient of the sum of squared residuals
# sgd: calculates the stochastic gradient descent

class gradient_ml:
    # This class has the following attributes:
    # start: initial value of the variable
    # lr: learning rate
    # ni: number of iterations
    # tol: tolerance
    def __init__(self, start: list[float] | float, learn: int | float, n_iter: int | float, tolerance: int | float) -> float:
        self.start = start if isinstance(start, list) else float(start)
        self.lr = float(learn)
        self.ni = int(n_iter)
        self.tol = float(tolerance)
    
# This method calculates the gradient of the function f(x) = 2x
    def gradient_descendent(self):
        cache = {}
        def gradient(x:int | float) -> float:
            if x in cache:
                return cache[x]
            # This function calculates the gradient of the function f(x) = 2x
            return 2*x
        vector = self.start
        for _ in range(self.ni):
             # Compute the gradient
            grad_value = gradient(vector)
            cache[vector] = grad_value
            # Compute the step
            diff = -self.lr * grad_value
            # Verify if the absolute difference is small enough
            if np.abs(diff) <= self.tol:
                break
            # Update the value of the variable
            vector += diff 
        return vector

# This method calculates the gradient of the sum of squared residuals
    def ssr_grad(self, x: float, y: float, b: float) -> float:
        res = b[0] + b[1] * x - y
        return np.array([res.mean(), (res * x).mean()])  # .mean() is a method of np.ndarray

# This method calculates the gradient of the sum of squared residuals using the method ssr_grad
    def gradient_descendent_ssr(self, x:float, y:float) -> float:
        vector = np.array(self.start)
        for _ in range(self.ni):
            diff = - self.lr * np.array(self.ssr_grad(x, y, vector))
            if np.all(np.abs(diff) <= self.tol):
                break
            vector += diff
        return vector

# This method calculates the stochastic gradient descent
    def sgd(self, x:float, y:float, batch_size:float, dtype="float64", random_state=None) -> float:
        # Converting x and y to NumPy arrays
        x, y = np.array(x, dtype=dtype), np.array(y, dtype=dtype)
        n_obs = x.shape[0]
        if n_obs != y.shape[0]:
            raise ValueError("'x' and 'y' lengths do not match")
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
        
        # Initializing the random number generator
        seed = None if random_state is None else int(random_state)
        rng = np.random.default_rng(seed=seed)
        # Initializing the vector
        vector = np.array(self.start, dtype=dtype)
        
        # Performing the gradient descent loop
        for _ in range(self.ni):
            # Shuffle x and y
            rng.shuffle(xy)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            # Recalculating the difference
            grad = np.array(self.ssr_grad(x_batch, y_batch, vector), dtype)
            diff = -self.lr * grad

            # Checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= self.tol):
                break

            # Updating the values of the variables
            vector += diff

        return vector if vector.shape else vector.item()
    
# Main function to test the class
if __name__ == "__main__":
    var = 1
    if var == 1:
        grad = gradient_ml(10.0, 0.2, 50, 1e-06)
        result1 = grad.gradient_descendent()
        print("This is the gradient descent:")
        print(result1)
    elif var == 2:
        grad = gradient_ml([10.0, 10.0], 0.0008, 100_000, 1e-06)
        x = np.array([5, 15, 25, 35, 45, 55])
        y = np.array([5, 20, 14, 32, 22, 38])
        result2 = grad.gradient_descendent_ssr(x,y)
        print("This is the gradient descent of the sum of squared residuals:")
        print(result2)
    elif var == 3:
        x = np.array([5, 25, 55, 75, 95, 115])
        y = np.array([5, 30, 60, 70, 80, 100])
        grad = gradient_ml([0.5, 0.5], 0.0008, 100_000, 1e-06)
        result3 = grad.sgd(x, y, batch_size=3, random_state=0)
        print("This is the stochastic gradient descent:")
        print(result3)