# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_excel('TG_XLSX_20234_caka_train.xlsx')

# Select relevant columns for the features (independent variables) and the target (dependent variable)
X = df[["Būves fiziskais nolietojums, %", "Telpu grupas platība, m2", "Telpu skaits telpu grupā"]]
y = df["Darījuma summa, EUR"]

# Feature scaling (optional, but can help gradient descent converge faster)
X = (X - X.mean()) / X.std()

# Add a column of ones for the bias term
X['bias'] = 1

# Convert dataframe to numpy arrays
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)  # Reshape y to a column vector

# Split the dataset into training and testing sets (80/20 split)
m = len(df)
m_train = int(0.8 * m)
X_train, X_test = X[:m_train], X[m_train:]
y_train, y_test = y[:m_train], y[m_train:]

# Initialize parameters
theta = np.zeros((X.shape[1], 1))

# Define the number of iterations
iterations = 30000

# Define the learning rate
alpha = 0.0002

# Define a list to store the cost at each iteration
cost_history = []

# Perform gradient descent
for _ in range(iterations):
    # Compute predictions
    predictions = np.dot(X_train, theta)
    
    # Compute error
    error = predictions - y_train
    
    # Compute gradient
    gradient = np.dot(X_train.T, error)
    
    # Update parameters
    theta -= (alpha / m_train) * gradient
    
    # Compute cost
    cost = np.sum(np.square(error)) / (2 * m_train)
    cost_history.append(cost)

# Print final parameters
print("Final parameters (theta):", theta)

# Plot cost history
plt.plot(range(1, iterations + 1), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

# Predict on the testing set
y_pred = np.dot(X_test, theta)

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (EUR)")
plt.ylabel("Predicted Price (EUR)")
plt.title("Actual vs Predicted Prices")
plt.show()



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_excel('TG_XLSX_20234_caka_train.xlsx')

# Apply square root transformation to "Būves fiziskais nolietojums, %" feature
df["Būves fiziskais nolietojums, %"] = np.sqrt(df["Būves fiziskais nolietojums, %"])

# Select relevant columns for the features (independent variables) and the target (dependent variable)
X = df[["Būves fiziskais nolietojums, %", "Telpu grupas platība, m2", "Telpu skaits telpu grupā"]]
y = df["Darījuma summa, EUR"]

# Feature scaling (optional, but can help gradient descent converge faster)
X = (X - X.mean()) / X.std()

# Add a column of ones for the bias term
X['bias'] = 1

# Convert dataframe to numpy arrays
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)  # Reshape y to a column vector

# Split the dataset into training and testing sets (80/20 split)
m = len(df)
m_train = int(0.8 * m)
X_train, X_test = X[:m_train], X[m_train:]
y_train, y_test = y[:m_train], y[m_train:]

# Initialize parameters
theta = np.zeros((X.shape[1], 1))

# Define the number of iterations
iterations = 30000

# Define the learning rate
alpha = 0.0002

# Define a list to store the cost at each iteration
cost_history = []

# Perform gradient descent
for _ in range(iterations):
    # Compute predictions
    predictions = np.dot(X_train, theta)
    
    # Compute error
    error = predictions - y_train
    
    # Compute gradient
    gradient = np.dot(X_train.T, error)
    
    # Update parameters
    theta -= (alpha / m_train) * gradient
    
    # Compute cost
    cost = np.sum(np.square(error)) / (2 * m_train)
    cost_history.append(cost)

# Print final parameters
print("Final parameters (theta):", theta)

# Plot cost history
plt.plot(range(1, iterations + 1), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

# Predict on the testing set
y_pred = np.dot(X_test, theta)

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (EUR)")
plt.ylabel("Predicted Price (EUR)")
plt.title("Actual vs Predicted Prices")
plt.show()



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_excel('TG_XLSX_20234_caka_train.xlsx')

# Apply square root transformation to "Būves fiziskais nolietojums, %" feature
df["Būves fiziskais nolietojums, %"] = np.sqrt(df["Būves fiziskais nolietojums, %"])

# Select relevant columns for the features (independent variables) and the target (dependent variable)
X = df[["Būves fiziskais nolietojums, %", "Telpu grupas platība, m2", "Telpu skaits telpu grupā"]]
y = df["Darījuma summa, EUR"]

# Feature scaling (optional, but can help gradient descent converge faster)
X = (X - X.mean()) / X.std()

# Add a column of ones for the bias term
X['bias'] = 1

# Convert dataframe to numpy arrays
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)  # Reshape y to a column vector

# Split the dataset into training and testing sets (80/20 split)
m = len(df)
m_train = int(0.8 * m)
X_train, X_test = X[:m_train], X[m_train:]
y_train, y_test = y[:m_train], y[m_train:]

# Initialize parameters
theta = np.zeros((X.shape[1], 1))

# Define the number of iterations
iterations = 100000

# Define the learning rate
alpha = 0.0002

# Define a list to store the cost at each iteration
cost_history = []

# Perform gradient descent
for _ in range(iterations):
    # Compute predictions
    predictions = np.dot(X_train, theta)
    
    # Compute error
    error = predictions - y_train
    
    # Compute gradient
    gradient = np.dot(X_train.T, error)
    
    # Update parameters
    theta -= (alpha / m_train) * gradient
    
    # Compute cost
    cost = np.sum(np.square(error)) / (2 * m_train)
    cost_history.append(cost)

# Predict on the testing set
y_pred = np.dot(X_test, theta)

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel("Actual Price (EUR)")
plt.ylabel("Predicted Price (EUR)")
plt.title("Actual vs Predicted Prices")
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_excel('TG_XLSX_20234_caka_train.xlsx')

# Select relevant columns for the features (independent variables) and the target (dependent variable)
X = df[["Būves fiziskais nolietojums, %", "Telpu grupas platība, m2", "Telpu skaits telpu grupā"]]
y = df["Darījuma summa, EUR"]

# Feature scaling (optional, but can help gradient descent converge faster)
X = (X - X.mean()) / X.std()

# Add a column of ones for the bias term
X['bias'] = 1

# Convert dataframe to numpy arrays
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)  # Reshape y to a column vector

# Split the dataset into training and testing sets (80/20 split)
m = len(df)
m_train = int(0.8 * m)
X_train, X_test = X[:m_train], X[m_train:]
y_train, y_test = y[:m_train], y[m_train:]

# Initialize parameters
theta = np.zeros((X.shape[1], 1))

# Define the number of iterations
iterations = 30000

# Define the learning rate
alpha = 0.0002

# Define a list to store the cost at each iteration
cost_history = []

# Perform gradient descent
for _ in range(iterations):
    # Compute predictions
    predictions = np.dot(X_train, theta)
    
    # Compute error
    error = predictions - y_train
    
    # Compute gradient
    gradient = np.dot(X_train.T, error)
    
    # Update parameters
    theta -= (alpha / m_train) * gradient
    
    # Compute cost
    cost = np.sum(np.square(error)) / (2 * m_train)
    cost_history.append(cost)

# Print final parameters
print("Final parameters (theta):", theta)

# Plot cost history
plt.plot(range(1, iterations + 1), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

# Predict on the testing set
y_pred = np.dot(X_test, theta)

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel("Actual Price (EUR)")
plt.ylabel("Predicted Price (EUR)")
plt.title("Actual vs Predicted Prices")
plt.show()

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    # Read the dataset
    df = pd.read_excel('TG_XLSX_20234_caka_train.xlsx')

    # Select relevant columns for the features (independent variables) and the target (dependent variable)
    X = df[["Būves fiziskais nolietojums, %", "Telpu grupas platība, m2", "Telpu skaits telpu grupā"]]
    y = df["Darījuma summa, EUR"]

    # Feature scaling (optional, but can help gradient descent converge faster)
    X = (X - X.mean()) / X.std()

    # Add a column of ones for the bias term
    X['bias'] = 1

    # Convert dataframe to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy().reshape(-1, 1)  # Reshape y to a column vector

    # Split the dataset into training and testing sets (80/20 split)
    m = len(df)
    m_train = int(0.8 * m)
    X_train, X_test = X[:m_train], X[m_train:]
    y_train, y_test = y[:m_train], y[m_train:]

    # Initialize parameters
    theta = np.zeros((X.shape[1], 1))

    # Define the number of iterations
    iterations = 30000

    # Define the learning rate
    alpha = 0.0002

    # Define a list to store the cost at each iteration
    cost_history = []

    # Perform gradient descent
    for _ in range(iterations):
        # Compute predictions
        predictions = np.dot(X_train, theta)
        
        # Compute error
        error = predictions - y_train
        
        # Compute gradient
        gradient = np.dot(X_train.T, error)
        
        # Update parameters
        theta -= (alpha / m_train) * gradient
        
        # Compute cost
        cost = np.sum(np.square(error)) / (2 * m_train)
        cost_history.append(cost)

    # Print final parameters
    print("Final parameters (theta):", theta)

    # Plot cost history
    plt.plot(range(1, iterations + 1), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()

    # Predict on the testing set
    y_pred = np.dot(X_test, theta)

    # Plot actual vs predicted prices
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
    plt.xlabel("Actual Price (EUR)")
    plt.ylabel("Predicted Price (EUR)")
    plt.title("Actual vs Predicted Prices")
    plt.show()

except Exception as e:
    print("An error occurred:", e)



# %%
