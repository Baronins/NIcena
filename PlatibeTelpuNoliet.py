# %%

# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from sklearn.preprocessing import PolynomialFeatures  # For creating polynomial features
from sklearn.linear_model import PoissonRegressor  # For building the Poisson regression model

# Step 1: Load the dataset
data = pd.read_excel("TG_XLSX_20234_caka_train.xlsx")

# Step 2: Selecting features and target
X1 = data[['Telpu skaits telpu grupā']]  # Feature 1: Telpu skaits telpu grupā (number of rooms)
X2 = data[['Telpu grupas platība, m2']]  # Feature 2: Telpu grupas platība, m2 (size of the apartment)
X3 = data[['Būves fiziskais nolietojums, %']]  # Feature 3: Būves fiziskais nolietojums, % (physical wear and tear)
y = data['Darījuma summa, EUR']  # Target: Darījuma summa, EUR (price)

# Step 3: Splitting the data into training and testing sets (80/20)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(X1, X2, X3, y, test_size=0.2, random_state=42)

# Step 4: Creating polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(np.c_[X1_train, X2_train, X3_train])

# Step 5: Initialize Poisson regression model
model = PoissonRegressor()

# Step 6: Train the model
model.fit(X_poly_train, y_train)

# Step 7: Make predictions on the testing set
X_poly_test = poly.transform(np.c_[X1_test, X2_test, X3_test])
y_pred = model.predict(X_poly_test)

# Step 8: Visualize the Results
plt.scatter(y_test, y_pred, color='blue')  # Plotting actual vs predicted prices

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Plotting the line of equality

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
from sklearn.preprocessing import PolynomialFeatures  # For creating polynomial features
from sklearn.linear_model import PoissonRegressor  # For building the Poisson regression model

# Step 1: Load the dataset
data = pd.read_excel("TG_XLSX_20234_caka_train.xlsx")

# Step 2: Selecting features and target
X1 = data[['Telpu skaits telpu grupā']]  # Feature 1: Telpu skaits telpu grupā (number of rooms)
X2 = data[['Telpu grupas platība, m2']]  # Feature 2: Telpu grupas platība, m2 (size of the apartment)
X3 = data[['Būves fiziskais nolietojums, %']]  # Feature 3: Būves fiziskais nolietojums, % (physical wear and tear)
y = data['Darījuma summa, EUR']  # Target: Darījuma summa, EUR (price)

# Step 3: Splitting the data into training and testing sets (80/20)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(X1, X2, X3, y, test_size=0.2, random_state=42)

# Step 4: Creating polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(np.c_[X1_train, X2_train, X3_train])

# Step 5: Initialize Poisson regression model with increased max_iter
model = PoissonRegressor(max_iter=1000)

# Step 6: Train the model
model.fit(X_poly_train, y_train)

# Step 7: Make predictions on the testing set
X_poly_test = poly.transform(np.c_[X1_test, X2_test, X3_test])
y_pred = model.predict(X_poly_test)

# Step 8: Compute and store the cost at each iteration
cost_history = []
for iteration in range(1000):  # Change the number of iterations as needed
    model.fit(X_poly_train, y_train)  # Fit the model
    y_pred_train = model.predict(X_poly_train)  # Predictions on training set
    cost = np.mean((y_pred_train - y_train)**2)  # Calculate the cost
    cost_history.append(cost)

# Step 9: Visualize the Cost Function
plt.plot(range(1, len(cost_history) + 1), cost_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.grid(True)
plt.show()


# %%
