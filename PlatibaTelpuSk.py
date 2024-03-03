# %%


# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization

# Step 1: Load the dataset
data = pd.read_excel("TG_XLSX_20234_caka_train.xlsx")

# Step 2: Selecting features and target
X1 = data[['Telpu skaits telpu grupā']]  # Feature 1: Telpu skaits telpu grupā (number of rooms)
X2 = data[['Telpu grupas platība, m2']]  # Feature 2: Telpu grupas platība, m2 (size of the apartment)
y = data['Darījuma summa, EUR']  # Target: Darījuma summa, EUR (price)

# Applying logarithmic transformation to the features
X1_log = np.log(X1)
X2_log = np.log(X2)

# Step 3: Splitting the data into training and testing sets (80/20)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1_log, X2_log, y, test_size=0.2, random_state=42)

# Step 4: Adding bias term to features
X_train_bias = np.c_[np.ones((X1_train.shape[0], 1)), X1_train, X2_train]  # Adding bias term and both features to training features

# Step 5: Initialize parameters
theta = np.random.randn(3, 1)  # Random initialization of parameters (including bias term)
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
X1_test_bias = np.c_[np.ones((X1_test.shape[0], 1)), X1_test, X2_test]  # Adding bias term and both features to testing features
y_pred = X1_test_bias.dot(theta)

# Step 8: Visualize the Results
plt.scatter(y_test, y_pred, color='blue')  # Plotting actual vs predicted prices

plt.xlabel('Actual Price (EUR)')  # Label for x-axis
plt.ylabel('Predicted Price (EUR)')  # Label for y-axis

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


# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from sklearn.preprocessing import PolynomialFeatures  # For creating polynomial features
from sklearn.linear_model import LinearRegression  # For building the linear regression model

# Step 1: Load the dataset
data = pd.read_excel("TG_XLSX_20234_caka_train.xlsx")

# Step 2: Selecting features and target
X1 = data[['Telpu skaits telpu grupā']]  # Feature 1: Telpu skaits telpu grupā (number of rooms)
X2 = data[['Telpu grupas platība, m2']]  # Feature 2: Telpu grupas platība, m2 (size of the apartment)
y = data['Darījuma summa, EUR']  # Target: Darījuma summa, EUR (price)

# Applying square root transformation to the features
X1_sqrt = np.sqrt(X1)
X2_sqrt = np.sqrt(X2)

# Step 3: Splitting the data into training and testing sets (80/20)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1_sqrt, X2_sqrt, y, test_size=0.2, random_state=42)

# Step 4: Creating polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(np.c_[X1_train, X2_train])

# Step 5: Initialize linear regression model
model = LinearRegression()

# Step 6: Train the model
model.fit(X_poly_train, y_train)

# Step 7: Make predictions on the testing set
X_poly_test = poly.transform(np.c_[X1_test, X2_test])
y_pred = model.predict(X_poly_test)

# Step 8: Visualize the Results
plt.scatter(y_test, y_pred, color='blue')  # Plotting actual vs predicted prices

plt.xlabel('Actual Price (EUR)')  # Label for x-axis
plt.ylabel('Predicted Price (EUR)')  # Label for y-axis

plt.grid(True)  # Adding grid

plt.show()  # Displaying the plot

# %%


# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from sklearn.linear_model import PoissonRegressor  # For building the Poisson regression model

# Step 1: Load the dataset
data = pd.read_excel("TG_XLSX_20234_caka_train.xlsx")

# Step 2: Selecting features and target
X1 = data[['Telpu skaits telpu grupā']]  # Feature 1: Telpu skaits telpu grupā (number of rooms)
X2 = data[['Telpu grupas platība, m2']]  # Feature 2: Telpu grupas platība, m2 (size of the apartment)
y = data['Darījuma summa, EUR']  # Target: Darījuma summa, EUR (price)

# Applying square root transformation to the features
X1_sqrt = np.sqrt(X1)
X2_sqrt = np.sqrt(X2)

# Step 3: Splitting the data into training and testing sets (80/20)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1_sqrt, X2_sqrt, y, test_size=0.2, random_state=42)

# Step 4: Initialize Poisson regression model
model = PoissonRegressor()

# Step 5: Train the model
model.fit(np.c_[X1_train, X2_train], y_train)

# Step 6: Make predictions on the testing set
y_pred = model.predict(np.c_[X1_test, X2_test])

# Step 7: Visualize the Results
plt.scatter(y_test, y_pred, color='blue')  # Plotting actual vs predicted prices

plt.xlabel('Actual Price (EUR)')  # Label for x-axis
plt.ylabel('Predicted Price (EUR)')  # Label for y-axis

plt.grid(True)  # Adding grid

plt.show()  # Displaying the plot

# %%


# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from sklearn.linear_model import PoissonRegressor  # For building the Poisson regression model

# Step 1: Load the dataset
data = pd.read_excel("TG_XLSX_20234_caka_train.xlsx")

# Step 2: Selecting features and target
X1 = data[['Telpu skaits telpu grupā']]  # Feature 1: Telpu skaits telpu grupā (number of rooms)
X2 = data[['Telpu grupas platība, m2']]  # Feature 2: Telpu grupas platība, m2 (size of the apartment)
y = data['Darījuma summa, EUR']  # Target: Darījuma summa, EUR (price)

# Applying square root transformation to the features
X1_sqrt = np.sqrt(X1)
X2_sqrt = np.sqrt(X2)

# Step 3: Splitting the data into training and testing sets (80/20)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1_sqrt, X2_sqrt, y, test_size=0.2, random_state=42)

# Step 4: Initialize Poisson regression model
model = PoissonRegressor()

# Step 5: Train the model
model.fit(np.c_[X1_train, X2_train], y_train)

# Step 6: Make predictions on the testing set
y_pred = model.predict(np.c_[X1_test, X2_test])

# Step 7: Visualize the Results
plt.scatter(y_test, y_pred, color='blue')  # Plotting actual vs predicted prices

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Plotting the line of equality

plt.xlabel('Actual Price (EUR)')  # Label for x-axis
plt.ylabel('Predicted Price (EUR)')  # Label for y-axis

plt.grid(True)  # Adding grid

plt.show()  # Displaying the plot

# %%


