import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to generate randomly distributed clusters
def generate_cluster(center, spread, num_points):
    x = center[0] + spread * (np.random.rand(num_points) - 0.5)
    y = center[1] + spread * (np.random.rand(num_points) - 0.5)
    z = center[2] + spread * (np.random.rand(num_points) - 0.5)
    return x, y, z

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define centers and spread for the clusters
centers = [(50, 50, 50), (150, 150, 150), (100, 200, 100)]
spread = 30
num_points = 50

# Generate and plot randomly distributed clusters
colors = ['brown', 'black', 'purple']
for center, color in zip(centers, colors):
    x, y, z = generate_cluster(center, spread, num_points)
    ax.scatter(x, y, z, c=color)

# Data for a new dot
dot_x = [100]
dot_y = [100]
dot_z = [100]
dot_color = "green"

# Plot the single dot with customized color
ax.scatter(dot_x, dot_y, dot_z, c=dot_color, s=100, alpha = 0.5, label='New and Unseen Dot')

# Set axis labels
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

# Set axis limits
ax.set_xlim(0, 260)
ax.set_ylim(0, 260)
ax.set_zlim(0, 260)

# Add a legend
ax.legend()

# Show plot
plt.show()
