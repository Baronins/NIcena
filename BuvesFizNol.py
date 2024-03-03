# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel("TG_XLSX_20234_train.xlsx")

# Select the column "Būves fiziskais nolietojums, %"
physical_depreciation = df["Būves fiziskais nolietojums, %"].values

# Log transformation
log_depreciation = np.log(physical_depreciation + 1)  # Adding 1 to handle zero values

# Square root transformation
sqrt_depreciation = np.sqrt(physical_depreciation)

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.hist(physical_depreciation, bins=20, color='blue', alpha=0.7)
plt.title('Original Data')

plt.subplot(1, 3, 2)
plt.hist(log_depreciation, bins=20, color='green', alpha=0.7)
plt.title('Log Transformation')

plt.subplot(1, 3, 3)
plt.hist(sqrt_depreciation, bins=20, color='red', alpha=0.7)
plt.title('Square Root Transformation')

plt.show()

# %%
