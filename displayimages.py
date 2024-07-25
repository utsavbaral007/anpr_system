# To display images in a grid
import os
import random
import matplotlib.pyplot as plt

# Set the directory where the images are located
image_dir = 'anpr_nepal/train/images'

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Select 12 random images from the list
selected_images = random.sample(image_files, 12)

# Create a figure with 3 rows and 4 subplots
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

# Iterate over the selected images and display them in the subplots
for i, image_file in enumerate(selected_images):
    image_path = os.path.join(image_dir, image_file)
    image = plt.imread(image_path)
    row = i // 4
    col = i % 4
    axes[row, col].imshow(image)
    axes[row, col].axis('off')

# Remove any empty subplots
for i in range(12, 12):
    row = i // 4
    col = i % 4
    fig.delaxes(axes[row, col])

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Display the figure
plt.show()
