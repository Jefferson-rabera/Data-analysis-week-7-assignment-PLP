# Task 1: Loading and Exploring the Dataset
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading the dataset
iris = fetch_ucirepo(id=53)
df_features = iris.data.features
df_target = iris.data.targets

# Combining features and target into one DataFrame for easier analysis
df = pd.concat([df_features, df_target], axis=1)

# Displaying the first few rows
print(df.head(10))

# Exploring the structure
print("\n2. Dataset structure:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n3. Data types:")
print(df.dtypes)

print("\n4. Missing values:")
print(df.isnull().sum())

# Cleaning the dataset (though Iris has no missing values)
print("\n5. Data cleaning:")
if df.isnull().sum().sum() > 0:
    # Fill numerical columns with mean
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    # Filling categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    print("Missing values have been filled")
else:
    print("No missing values found")

# Task 2: Basic Data Analysis
print("\n\n TASK 2: BASIC DATA ANALYSIS")

# Basic statistics of numerical columns
print("\n1. Basic statistics of numerical columns:")
print(df.describe())

# Grouping by categorical column and computing mean
print("\n2. Group analysis - Mean of numerical columns by species:")
grouped_stats = df.groupby('class').mean()
print(grouped_stats)

# Additional insights
print("\n3. Interesting findings:")
print(f"- Dataset contains {len(df)} samples across {len(df['class'].unique())} species")
print(f"- Sepal length ranges from {df['sepal length'].min():.1f}cm to {df['sepal length'].max():.1f}cm")
print(f"- Setosa species has the smallest petals (avg width: {grouped_stats.loc['Iris-setosa', 'petal width']:.2f}cm)")
print(f"- Virginica species has the largest petals (avg width: {grouped_stats.loc['Iris-virginica', 'petal width']:.2f}cm)")

# Task 3: Data Visualization
print("\n\n=== TASK 3: DATA VISUALIZATION ===")

# Creating subplots for better organization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Visualizations', fontsize=16, fontweight='bold')

# 1. Line chart showing trends (using index as pseudo-time)
print("\n1. Creating line chart...")
axes[0, 0].plot(df.index, df['sepal length'], label='Sepal Length', color='blue', alpha=0.7)
axes[0, 0].plot(df.index, df['petal length'], label='Petal Length', color='red', alpha=0.7)
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].set_title('Line Chart: Sepal vs Petal Length Trends')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart comparing numerical values across categories
print("2. Creating bar chart...")
species_means = df.groupby('class')['sepal length'].mean()
colors = ['lightblue', 'lightcoral', 'lightgreen']
axes[0, 1].bar(species_means.index, species_means.values, color=colors, alpha=0.7)
axes[0, 1].set_xlabel('Iris Species')
axes[0, 1].set_ylabel('Average Sepal Length (cm)')
axes[0, 1].set_title('Bar Chart: Average Sepal Length by Species')
axes[0, 1].tick_params(axis='x', rotation=45)

# Adding value labels on bars
for i, v in enumerate(species_means.values):
    axes[0, 1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')

# 3. Histogram of a numerical column
print("3. Creating histogram...")
axes[1, 0].hist(df['petal width'], bins=15, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Petal Width (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram: Distribution of Petal Width')
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot of two numerical columns
print("4. Creating scatter plot...")
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
for species in df['class'].unique():
    subset = df[df['class'] == species]
    axes[1, 1].scatter(subset['sepal length'], subset['petal length'], 
                      label=species, alpha=0.7, color=colors[species])
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].set_title('Scatter Plot: Sepal Length vs Petal Length')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Adjusting layout and display
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Summary of my analysis
print("• Successfully loaded and explored the Iris dataset")
print("• Performed statistical analysis and grouping operations")
print("• Created 4 different types of visualizations:")
print("  1. Line chart showing measurement trends")
print("  2. Bar chart comparing species averages")
print("  3. Histogram showing distribution")
print("  4. Scatter plot showing relationship between variables")

# Saving the cleaned dataset
df.to_csv('cleaned_iris_dataset.csv', index=False)
print(f"\nCleaned dataset saved as 'cleaned_iris_dataset.csv'")