import copy
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

class MultipleLinearRegression:

    def __init__(self):
        """Initializing a Multiple Linear Regression Class"""

    def predict_single_loop(self, x, w, b) -> float:
        """
        single predict using linear regression

        Args:
        x (ndarray): Shape (n,) example with multiple features
        w (ndarray): Shape (n,) model parameters
        b (scalar):  model parameter

        Returns:
        p (scalar):  prediction
        """
        n = x.shape[0]
        p = 0
        for i in range(n):
            p_i = x[i] * w[i]
            p = p + p_i
        p = p + b
        return p

    def predict(self, x, w, b) -> float:
        """
        single predict using linear regression
        Args:
        x (ndarray): Shape (n,) example with multiple features
        w (ndarray): Shape (n,) model parameters
        b (scalar):             model parameter

        Returns:
        p (scalar):  prediction
        """
        p = np.dot(x, w) + b
        return p

    def compute_cost(self, X, y, w, b) -> float:
        """
        compute cost
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter

        Returns:
        cost (scalar): cost
        """
        m = X.shape[0]
        cost = 0.0
        for i in range(m):
            f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
            cost = cost + (f_wb_i - y[i]) ** 2  # scalar
        cost = cost / (2 * m)  # scalar
        return cost

    def compute_gradient(self, X, y, w, b) -> (float, float):
        """
        Computes the gradient for linear regression
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter

        Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
        dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
        """
        m, n = X.shape  # (number of examples, number of features)
        dj_dw = np.zeros((n,))
        dj_db = 0.

        for i in range(m):
            err = (np.dot(X[i], w) + b) - y[i]
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err * X[i, j]
            dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_db, dj_dw

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters) -> (float, float, list):
        """
        Performs batch gradient descent to learn w and b. Updates w and b by taking
        num_iters gradient steps with learning rate alpha

        Args:
        X (ndarray (m,n))   : Data, m examples with n features
        y (ndarray (m,))    : target values
        w_in (ndarray (n,)) : initial model parameters
        b_in (scalar)       : initial model parameter
        cost_function       : function to compute cost
        gradient_function   : function to compute the gradient
        alpha (float)       : Learning rate
        num_iters (int)     : number of iterations to run gradient descent

        Returns:
        w (ndarray (n,)) : Updated values of parameters
        b (scalar)       : Updated value of parameter
        """

        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w = copy.deepcopy(w_in)  # avoid modifying global w within function
        b = b_in

        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_db, dj_dw = self.compute_gradient(X, y, w, b)  ##None

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw  ##None
            b = b - alpha * dj_db  ##None

            # Save cost J at each iteration
            if i < 100000:  # prevent resource exhaustion
                J_history.append(self.compute_cost(X, y, w, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

        return w, b, J_history  # return final w,b and J history for graphing


# Load CSV into a NumPy array (excluding header)
beds,baths,den,parking,D_mkt,building_age,maint,price,new_size,new_exposure,new_ward = (
    np.loadtxt("load_data1.csv", delimiter=",", skiprows=1, unpack=True))

# Stack the features (all columns except price) into a single array
X = np.column_stack((beds, baths, den, maint, new_size, new_ward))
X1, mu, sig = zscore_normalize_features(X)

# The target variable (price)
y = price

# Split the data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X1, y, test_size=0.2, random_state=42)

l_regression = MultipleLinearRegression()
w_in = np.ones(X.shape[1])
w_array, b_array, j_hist = l_regression.gradient_descent(X_train, y_train, w_in, 1, 0.001, 10000)

# Initialize an empty list to store predictions
predictions = []

# Loop through each example in X_val and predict
for x in X_val:
    prediction = l_regression.predict_single_loop(x, w_array, b_array)
    predictions.append(prediction)

# Convert the list of predictions to a numpy array
predictions = np.array(predictions)

# Now compare predictions to y_val
mse = (np.mean((predictions - y_val) ** 2))
acc = np.mean((np.abs(predictions - y_val)/y_val)) * 100
print("Mean Squared Error on the validation set: ",mse)
print("Accuracy is: ", acc)

plt.plot(j_hist)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History during Gradient Descent')
plt.show()