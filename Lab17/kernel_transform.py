import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data for transformation
data = {
    'x1': [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16],
    'x2': [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3],
    'Label': ['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Red', 'Red', 'Red', 'Red', 'Red',
              'Red', 'Red', 'Red']
}

df = pd.DataFrame(data)

# Define transform function
def transform(x1, x2):
    return np.array([x1**2, x2**2, np.sqrt(2)*(x1 * x2)])

# Apply transform to each row
x_transformed = []
Labels = []
for x1, x2, label in zip(df['x1'], df['x2'], df['Label']):
    x_new = transform(x1, x2)
    x_transformed.append(x_new)
    Labels.append(label)

x_transformed = np.array(x_transformed)

# Build transformed DataFrame
df_transformed = pd.DataFrame(x_transformed, columns=['x1_transformed', 'x2_transformed', 'x3_transformed'])
df_transformed['Label'] = Labels
print(df_transformed)

# Plot original 2D data
sns.scatterplot(x='x1', y='x2', hue='Label', style='Label', size='x1', data=df)
plt.title("Original 2D Data")
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Color map for clarity
colors = {'Blue': 'blue', 'Red': 'red'}

for label in df_transformed['Label'].unique():
    subset = df_transformed[df_transformed['Label'] == label]
    ax.scatter(subset['x1_transformed'], subset['x2_transformed'], subset['x3_transformed'],
               label=label, color=colors[label], s=50)

ax.set_xlabel('x1^2')
ax.set_ylabel('x2^2')
ax.set_zlabel('sqrt(x1*x2)')
ax.set_title('3D Transformed Data')
ax.view_init(elev=25, azim=120)
# elev=25: Elevation (up-down angle).
#
# azim=120: Azimuth (rotation around the plot)

ax.legend()
plt.tight_layout()
plt.show()




