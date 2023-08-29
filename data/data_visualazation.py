import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read the CSV files using Pandas
training = pd.read_csv("data/labelled_training_data.csv", header=0)
testing = pd.read_csv("data/labelled_testing_data.csv", header=0)
validation = pd.read_csv("data/labelled_validation_data.csv", header=0)

# Calculate the correlation matrix for training data
corr = training.corr()

# Display the correlation matrix
plt.figure(figsize=(10, 8))
plt.title("Correlation Matrix")
heatmap = plt.imshow(corr, cmap='coolwarm', interpolation='none', vmin=-1, vmax=1)

# Add numerical labels to each box
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.3f}", ha='center', va='center', color='black')

plt.colorbar(heatmap)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

# Save the plot to a file
plt.savefig("correlation_matrix.png")
plt.show()

# Similarly, you can repeat the above code for testing and validation data
