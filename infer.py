import torch
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision import transforms
from PIL import Image
from matplotlib.patches import Rectangle


import matplotlib.pyplot as plt
import pprint

import requests
image = Image.open("/app/books/chudo_derevo_redmi/IMG_20190715_112912.labeled.jpg")

label_to_int = {'б': 0, 'с': 1, 's': 2, '+': 3, '?': 4, 'ю': 5, '=': 6, '0': 7, 'д': 8, '.': 9, 'ь': 10, '~46': 11, '()': 12, ':': 13, 'k': 14, '4': 15, '~4': 16, 'м': 17, 't': 18, 'm': 19, 'ы': 20, 'ф': 21, '1': 22, '>>': 23, 'и': 24, 'е': 25, 'г': 26, '{': 27, 'l': 28, '6': 29, 'у': 30, 'я': 31, 'о': 32, 'o': 33, 'ё': 34, 'т': 35, '/4': 36, ')': 37, '-': 38, 'e': 39, 'z': 40, '!': 41, '§': 42, '3': 43, 'а': 44, '»': 45, 'a': 46, 'n': 47, '«': 48, 'ъ': 49, 'к': 50, 'XX': 51, 'щ': 52, '7': 53, 'q': 54, 'л': 55, 'в': 56, '8': 57, 'п': 58, ';': 59, 'р': 60, '5': 61, 'э': 62, '9': 63, '(': 64, '2': 65, 'й': 66, 'ж': 67, ',': 68, 'ч': 69, '|': 70, 'w': 71, 'ц': 72, 'х': 73, 'н': 74, 'ш': 75, '##': 76, '~3': 77, 'd': 78, '}': 79, '/1': 80, 'з': 81, 'CC': 82}
int_to_label = {0: 'б', 1: 'с', 2: 's', 3: '+', 4: '?', 5: 'ю', 6: '=', 7: '0', 8: 'д', 9: '.', 10: 'ь', 11: '~46', 12: '()', 13: ':', 14: 'k', 15: '4', 16: '~4', 17: 'м', 18: 't', 19: 'm', 20: 'ы', 21: 'ф', 22: '1', 23: '>>', 24: 'и', 25: 'е', 26: 'г', 27: '{', 28: 'l', 29: '6', 30: 'у', 31: 'я', 32: 'о', 33: 'o', 34: 'ё', 35: 'т', 36: '/4', 37: ')', 38: '-', 39: 'e', 40: 'z', 41: '!', 42: '§', 43: '3', 44: 'а', 45: '»', 46: 'a', 47: 'n', 48: '«', 49: 'ъ', 50: 'к', 51: 'XX', 52: 'щ', 53: '7', 54: 'q', 55: 'л', 56: 'в', 57: '8', 58: 'п', 59: ';', 60: 'р', 61: '5', 62: 'э', 63: '9', 64: '(', 65: '2', 66: 'й', 67: 'ж', 68: ',', 69: 'ч', 70: '|', 71: 'w', 72: 'ц', 73: 'х', 74: 'н', 75: 'ш', 76: '##', 77: '~3', 78: 'd', 79: '}', 80: '/1', 81: 'з', 82: 'CC'}

model = retinanet_resnet50_fpn_v2(num_classes=83, score_thresh=0.55)
model.load_state_dict(torch.load('model.pth'))

# Put the model in inference mode
model.eval()
# Get the transforms for the model's weights
# preprocess = weights.transforms()

# Define multiple transforms
transform = transforms.Compose([transforms.ToTensor()])
# Apply the transforms to the image
img = transform(image)

batch = [img]
prediction = model(batch)[0]

labels = [int_to_label[i] for i in prediction["labels"].tolist()]

img = img.type(torch.uint8)

# Get the boxes, labels, and scores from the prediction results
boxes = prediction["boxes"]
labels = prediction["labels"]
scores = prediction["scores"]

# Create a figure and axes
fig, ax = plt.subplots(1)
ax.imshow(image)

# Iterate over the boxes, labels, and scores
for box, label, score in zip(boxes, labels, scores):
    # Get the coordinates of the bounding box
    xmin, ymin, xmax, ymax = box.detach().numpy()

    # Create a rectangle patch
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the axes
    ax.add_patch(rect)

    # Add the label and score as text
    plt.text(xmin, ymin-10, f"{label}: {score:0.2f}", fontsize=10, color='r')

plt.show()
