import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# Load CSV into a NumPy array (excluding header)
beds, baths, den, parking, D_mkt, building_age, maint, price, new_size, new_exposure, new_ward = np.loadtxt("load_data1.csv", delimiter=",", skiprows=1, unpack=True)

# Stack the features (all columns except price) into a single array
X = np.column_stack((beds, baths, den, parking, D_mkt, building_age, maint, new_size, new_exposure, new_ward))

# The target variable (price)
y = price

# Split the data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LinearRegression model from sklearn
lin_reg = LinearRegression()

# Train the model
lin_reg.fit(X_train, y_train)

# Make predictions on the validation set
predictions = lin_reg.predict(X_val)
acc = np.mean(np.abs((predictions - y_val)/y_val) * 100)
# Compute the Mean Squared Error (MSE) for the validation set
mse = np.mean((predictions - y_val) ** 2)
print(f"Mean Squared Error on the validation set: {mse}")
print("Accuracy is: ", acc)


# Plotting the cost history is not applicable here since sklearn handles the optimization internally.
# But we can visualize the predictions vs. actual values on the validation set.
plt.plot()
plt.scatter(y_val, predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Price')
plt.show()
