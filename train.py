import os, json, pprint
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.optim import Adam
from torch.nn import BCELoss
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class LabelboxDataset(Dataset):
#     def __init__(self, json_files):
#         self.annotations = []
#         self.image_paths = []
#         self.shapes = []
#         self.image_heights = []
#         self.image_widths = []

#         for json_file in json_files:
#             with open(json_file, 'r') as f:
#                 annotation = json.load(f)
#                 self.annotations.append(annotation)
#                 self.image_paths.append(os.path.join(os.path.dirname(json_file), annotation['imagePath']))
#                 self.shapes.extend(annotation['shapes'])
#                 self.image_heights.append(annotation['imageHeight'])
#                 self.image_widths.append(annotation['imageWidth'])

#     def __len__(self):
#         return len(self.shapes)

#     def __getitem__(self, idx):
#         shape = self.shapes[idx]
#         label = shape['label']
#         points = shape['points']
#         image_path = self.image_paths[idx // len(self.shapes)]
#         image_height = self.image_heights[idx // len(self.shapes)]
#         image_width = self.image_widths[idx // len(self.shapes)]

#         sample = {
#             'label': label, 
#             'points': points, 
#             'image_path': image_path, 
#             'image_height': image_height, 
#             'image_width': image_width,
#         }
#         return image_path, label


class RetinaNetDataset(Dataset):
    def __init__(self, annotation_files, transform=None):
        self.transform = transform
        self.annotation_files = annotation_files
        self.annotations = []
        self.labels = []

        for json_file in self.annotation_files:
            with open(json_file, 'r') as f:
                annotation = json.load(f)
                annotation['imagePath'] = os.path.join(os.path.dirname(json_file), annotation["imagePath"])
                self.annotations.append(annotation)

                # Extract label
                for shape in annotation["shapes"]:
                    self.labels.append(shape["label"])

        self.label_to_int = {label: i for i, label in enumerate(set(self.labels))}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        

    def __len__(self):
        return len(self.annotation_files)

    def read_and_process_annotation(self, annotation):
        # Read image
        image = Image.open(annotation["imagePath"])

        labels = []
        boxes = []
        # Iterate through each shape in the annotation
        for shape in annotation["shapes"]:
            # Extract label
            labels.append(self.label_to_int[shape["label"]])

            # Extract x, y coordinates of the bounding box
            x1, y1 = shape["points"][0]
            x2, y2 = shape["points"][1]
            # Append a bounding box coordinates to the boxes list
            boxes.append([x1, y1, x2, y2])

        return image, {
            'boxes': torch.tensor(boxes),
            'labels': torch.tensor(labels)
        }

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        # Code to read and process the annotation file
        image, targets = self.read_and_process_annotation(annotation)

        if self.transform:
            imageTensor = self.transform(image)

        if imageTensor.size() != torch.Size([3, 1376, 1024]):
            # TODO check what is cropped
            image = image.crop([0, 0, 1024, 1376])
            imageTensor = self.transform(image)

        return imageTensor, targets

# Define the collate_fn function
def labelbox_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# dataset = RetinaNetDataset(annotation_files, transform=transforms.ToTensor())
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# usage
train_txt = '/app/books/train.txt'
with open(train_txt, 'r') as file:
    annotation_files = ['/app/books/' + line.strip().replace('.jpg', '.json') for line in file.readlines()]

# dataset = LabelboxDataset(annotation_files)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

transform = transforms.Compose([transforms.ToTensor()])
dataset = RetinaNetDataset(annotation_files, transform=transform)
print(dataset.label_to_int)
print(dataset.int_to_label)


# define the dataloader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, collate_fn=labelbox_collate_fn)

# for i, data in enumerate(dataloader):
#     print(data)
#     import pdb; pdb.set_trace()


# def plot_bboxes(ax, shapes):

#     # Iterate through the shapes and draw rectangles
#     for shape in shapes:
#         x1 = shape[1]
#         y1 = shape[2]
#         w = shape[3]
#         h = shape[4]
#         rect = Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#         plt.text(x=x1, y=y1, s=shape[0], fontsize=20, color='red')


# # example usage
# for i in range(10): #len(dataset)):
#     sample = dataset[i]
#     image = sample[0]
#     shapes = sample[1]

#     # Create figure and axes
#     fig, ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(image)
#     plot_bboxes(ax, shapes)
#     # Show the plot
#     plt.show()
# exit()

# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = BCELoss(reduction='none')(inputs, targets)
#         else:
#             BCE_loss = BCELoss(reduction='none')(torch.sigmoid(inputs), targets)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

# define model, loss function and optimizer
num_classes = len(set(dataset.labels))
model = models.detection.retinanet_resnet50_fpn_v2(
    weights=None,
    weights_backbone=models.ResNet50_Weights,
    num_classes=num_classes
)
model = model.to(device)

import pdb; pdb.set_trace()

# loss_fn = FocalLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# define the number of training steps
num_steps = 10

model.train()
for step in range(num_steps):
    # get a batch of data
    for iter, data in enumerate(dataloader):
        images, targets = data

        for i, im in enumerate(images):
            images[i] = images[i].to(device)
            targets[i]['boxes'] = targets[i]['boxes'].to(device)
            targets[i]['labels'] = targets[i]['labels'].to(device)

        # forward pass
        logits = model(images, targets)
        print(iter, logits)
        # loss = loss_fn(logits, targets)
        losses = sum(loss for loss in logits.values())

        # backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # print loss every X steps
    if step % 1 == 0:
        print('Step {}, Loss: {}'.format(step, losses))

# save the model
torch.save(model.state_dict(), 'model.pth')

