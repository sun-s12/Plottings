import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import os

# 🔹 Step 1: Import Data
file_path = 'C:/Users/Careline M01/Downloads/PCA/Vs Code Project/CRM/PCA_All_before selection-SSA updated.xlsx'
data = pd.read_excel(file_path)

# Remove the 'ID' column (assuming the first column is named 'ID')
features = data.drop(columns=['ID'])

# 🔹 Step 2: Perform Shapiro-Wilk normality test for each feature and choose correlation method
def calculate_correlation_matrix(features):
    num_features = features.shape[1]
    correlation_matrix = pd.DataFrame(np.zeros((num_features, num_features)), 
                                      columns=features.columns, index=features.columns)
    for i in range(num_features):
        for j in range(i, num_features):
            feature_i = features.iloc[:, i]
            feature_j = features.iloc[:, j]
            p_value_i = shapiro(feature_i)[1]
            p_value_j = shapiro(feature_j)[1]
            method = 'pearson' if p_value_i > 0.05 and p_value_j > 0.05 else 'spearman'
            corr = features.iloc[:, [i, j]].corr(method=method).iloc[0, 1]
            correlation_matrix.iloc[i, j] = corr
            correlation_matrix.iloc[j, i] = corr
    return correlation_matrix

# Calculate the correlation matrix
correlation_matrix = calculate_correlation_matrix(features)

# 🔹 Step 3: Save Correlation Matrix to Excel
output_file = 'C:/Users/Careline M01/Downloads/PCA/Vs Code Project/CRM/Correlation_Matrix.xlsx'
correlation_matrix.to_excel(output_file, index=True)
print(f"\n✅ Correlation matrix saved successfully to: {output_file}")

# 🔹 Step 4: Plot the correlation matrix using circles
fig, ax = plt.subplots(figsize=(12, 10))

# Create a scatter plot to visualize correlations
scatter = []
for i in range(correlation_matrix.shape[0]):
    for j in range(i):  # Only plot the lower triangle
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.3:  # Show only correlations above a threshold
            color = plt.cm.coolwarm((corr_value + 1) / 2)  # Normalize correlation values to [0, 1]
            size = abs(corr_value) * 200  # Adjust size for better visibility
            scatter.append(ax.scatter(j + 0.5, i + 0.5, s=size, c=[corr_value], cmap='coolwarm', 
                                      edgecolor='black', alpha=0.8, vmin=-1, vmax=1))

# 🔹 Step 5: Add a color bar linked to scatter plot
if scatter:
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
    cbar.set_label('Correlation Coefficient', fontsize=12)

# 🔹 Step 6: Customize axis labels and ticks
plt.xticks(np.arange(len(correlation_matrix.columns)) + 0.5, correlation_matrix.columns, rotation=90, fontsize=10)
plt.yticks(np.arange(len(correlation_matrix.columns)) + 0.5, correlation_matrix.columns, fontsize=10)

# Add gridlines
plt.grid(visible=False)
plt.gca().invert_yaxis()  # Match the inverted y-axis style
plt.title("Correlation Matrix with Circle Visualization", fontsize=16)
plt.tight_layout()

# Show the plot
plt.show()
