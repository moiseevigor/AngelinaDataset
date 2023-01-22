import json, os
import numpy as np
from PIL import Image

# Use the mean and std to normalize your dataset
r_means, g_means, b_means = [], [], []
r_stds, g_stds, b_stds = [], [], []
for filepath in [
    '/app/books/train.txt',
    # '/app/not_braille/train.txt',
    '/app/handwritten/train.txt',
    '/app/uploaded/test2.txt',
    '/app/books/val.txt',
    '/app/handwritten/val.txt',
]:
    with open(filepath, 'r') as file:
        annotation_files = ([os.path.join(os.path.dirname(filepath), line.strip().replace('.jpg', '.json')) for line in file.readlines()])
        for file in annotation_files:
            with open(file, 'r') as f:
                annotation = json.load(f)
                imagePath = os.path.join(os.path.dirname(file), annotation["imagePath"])

                # Load the images
                img = np.array(Image.open(imagePath))
                img = img / 255.0

                # Compute the mean and standard deviation of the pixel values for each channel
                r_means.append(np.mean(img[:,:,0]))
                g_means.append(np.mean(img[:,:,1]))
                b_means.append(np.mean(img[:,:,2]))
                r_stds.append(np.std(img[:,:,0]))
                g_stds.append(np.std(img[:,:,1]))
                b_stds.append(np.std(img[:,:,2]))

# Compute the overall mean and standard deviation of the means and standard deviations for each channel
r_mean, g_mean, b_mean = np.mean(r_means), np.mean(g_means), np.mean(b_means)
r_std, g_std, b_std = np.mean(r_stds), np.mean(g_stds), np.mean(b_stds)

print('means', [r_mean, g_mean, b_mean])
print('stds', [r_std, g_std, b_std])