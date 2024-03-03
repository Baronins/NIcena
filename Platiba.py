

# %%



# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization

# Step 1: Load the dataset
data = pd.read_excel("TG_XLSX_20234_caka_train.xlsx")

# Step 2: Selecting features and target
X = data[['Telpu skaits telpu grupā']]  # Feature: Telpu skaits telpu grupā (number of rooms)
y = data['Darījuma summa, EUR']  # Target: Darījuma summa, EUR (price)

# Applying logarithmic transformation to the feature
X_log = np.log(X)

# Step 3: Splitting the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.2, random_state=42)

# Step 4: Adding bias term to features
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Adding bias term to training features

# Step 5: Initialize parameters
theta = np.random.randn(2, 1)  # Random initialization of parameters (including bias term)
learning_rate = 0.0005  # Learning rate
n_iterations = 3000  # Number of iterations

# Step 6: Gradient Descent
cost_history = []  # List to store the cost at each iteration
for iteration in range(n_iterations):
    gradients = 2/X_train_bias.shape[0] * X_train_bias.T.dot(X_train_bias.dot(theta) - y_train.values.reshape(-1, 1))
    theta -= learning_rate * gradients
    # Calculate and store the cost
    cost = np.mean((X_train_bias.dot(theta) - y_train.values.reshape(-1, 1)) ** 2)
    cost_history.append(cost)

# Step 7: Make predictions on the testing set
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # Adding bias term to testing features
y_pred = X_test_bias.dot(theta)

# Step 8: Visualize the Results
plt.scatter(X_test, y_test,  color='black', label='Actual data')  # Plotting actual data points
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted line')  # Plotting the predicted line

plt.xlabel('Log(Telpu skaits telpu grupā)')  # Label for x-axis
plt.ylabel('Price (EUR)')  # Label for y-axis

plt.legend()  # Displaying legend
plt.grid(True)  # Adding grid

plt.show()  # Displaying the plot

# Step 9: Plotting the Cost Function
plt.plot(range(1, n_iterations + 1), cost_history, color='red')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.grid(True)
plt.show()


# %%
