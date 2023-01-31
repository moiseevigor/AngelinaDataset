import os, json, pprint, copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torchvision.ops as ops
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.optim import Adam
from torch.nn import BCELoss
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
from codes import get_code

# Define the argument parser
parser = argparse.ArgumentParser(description='Load model weights')
parser.add_argument('--weights', required=False, help='Path to the model weights file')
args = parser.parse_args()

num_experiment = 21
# Create a SummaryWriter object
writer = SummaryWriter(f'/app/experiments/retinanet/adamw/exp-{num_experiment}-resnet50-onecycle-max_lr-1e-4')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RetinaNetDataset(Dataset):
    def __init__(self, train_annotation_files, transform=None, labels=None):
        self.transform = transform
        self.train_annotation_files = train_annotation_files
        self.annotations = []
        self.set_labels = labels is not None
        self.labels = []
        if self.set_labels:
            self.labels = copy.deepcopy(labels)
        
        for json_file in self.train_annotation_files:
            with open(json_file, 'r') as f:
                annotation = json.load(f)
                annotation['imagePath'] = os.path.join(os.path.dirname(json_file), annotation["imagePath"])
                self.annotations.append(annotation)

                # Extract label
                if not self.set_labels:
                    for shape in annotation["shapes"]:
                        self.labels.append(get_code(shape["label"]))
        
        self.label_to_int = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}

    def __len__(self):
        return len(self.train_annotation_files)

    def read_and_process_annotation(self, annotation):
        # Read image
        image = Image.open(annotation["imagePath"])

        labels = []
        boxes = []
        # Iterate through each shape in the annotation
        for shape in annotation["shapes"]:
            # Extract label
            labels.append(self.label_to_int[get_code(shape["label"])])

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

        return imageTensor, targets

# Define the collate_fn function
def labelbox_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

annotation_files = []
for filepath in [
    '/app/books/train.txt',
    '/app/not_braille/train.txt',
    '/app/handwritten/train.txt',
    '/app/uploaded/test2.txt',
    '/app/books/val.txt',
    '/app/handwritten/val.txt',
]:
    with open(filepath, 'r') as file:
        annotation_files.extend([os.path.join(os.path.dirname(filepath), line.strip().replace('.jpg', '.json')) for line in file.readlines()])

dataset = RetinaNetDataset(annotation_files)
print('labels:', sorted(set(dataset.labels)), '\n')
print('label_to_int:', dataset.label_to_int, '\n')
print('int_to_label:', dataset.int_to_label, '\n')

train_annotation_files = []
for filepath in [
    '/app/books/train.txt',
    # '/app/not_braille/train.txt',
    '/app/handwritten/train.txt',
    '/app/uploaded/test2.txt',
]:
    with open(filepath, 'r') as file:
        train_annotation_files.extend([os.path.join(os.path.dirname(filepath), line.strip().replace('.jpg', '.json')) for line in file.readlines()])

# # Define the transformation
# rotation_transform = transforms.RandomRotation(degrees=45)
# # Define the bounding box transformation
# def bbox_transform(bboxes, rotation_transform):
#     bboxes = np.array([[xmin, ymin, xmax - xmin, ymax - ymin] for xmin, ymin, xmax, ymax in bboxes])
#     rotated_bboxes = rotation_transform.get_params(rotation_transform.degrees)(bboxes)
#     rotated_bboxes = np.array([[x, y, x + w, y + h] for x, y, w, h in rotated_bboxes])
#     return rotated_bboxes
# transform = transforms.Compose([
#     rotation_transform,
#     transforms.Lambda(lambda img, bboxes: (img, bbox_transform(bboxes, rotation_transform))),
# ])

# define the train_dataloader
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=1, p=0.1),
    transforms.RandomGrayscale(p=0.1),
    # transforms.RandomResizedCrop((1376, 1024), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5749533646009656, 0.5758692075743113, 0.5564374772810018], 
        std=[0.12675546510063618, 0.13864833881922706, 0.14966126335877825]
    )
])
train_dataset = RetinaNetDataset(train_annotation_files, transform=transform, labels=dataset.labels)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=labelbox_collate_fn)

# define the train_dataloader
val_annotation_files = []
for filepath in [
    '/app/books/val.txt',
    # '/app/not_braille/train.txt',
    '/app/handwritten/val.txt',
    # '/app/uploaded/test2.txt',
]:
    with open(filepath, 'r') as file:
        val_annotation_files.extend([os.path.join(os.path.dirname(filepath), line.strip().replace('.jpg', '.json')) for line in file.readlines()])

val_dataset = RetinaNetDataset(val_annotation_files, transform=transform, labels=dataset.labels)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=labelbox_collate_fn)

# for i, data in enumerate(train_dataloader):
#     print(data)

# def plot_bboxes(ax, shapes):

#     # Iterate through the shapes and draw rectangles
#     for shape in shapes['boxes']:

#         x1 = shape[0]
#         y1 = shape[1]
#         x2 = shape[2]
#         y2 = shape[3]

#         rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#         plt.text(x=x1, y=y1, s='', fontsize=20, color='red')

# from torchvision.transforms.functional import to_pil_image

# # example usage
# for i in range(100,200): #len(train_dataset)):
#     sample = train_dataset[i]

#     mean=[0.5749533646009656, 0.5758692075743113, 0.5564374772810018]
#     std=[0.12675546510063618, 0.13864833881922706, 0.14966126335877825]
#     image = sample[0]
#     mean = torch.tensor(mean).view(-1, 1, 1)
#     std = torch.tensor(std).view(-1, 1, 1)
#     image = image * std + mean

#     shapes = sample[1]

#     # Create figure and axes
#     fig, ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(to_pil_image(image))
#     plot_bboxes(ax, shapes)
#     # Show the plot
#     plt.show()
# exit()

def compute_acc_iou(model, dataloader, threshold=0.45):
    """
    Computes accuracy and average IoU
        True Positive (TP): model correctly detects and classifies a braille char as a char.
        False Positive (FP): model detects an object as a braille char, even though it is not a braille char.
        False Negative (FN): model fails to detect a braille char in an image, even though a braille char is present.
        True Negative (TN): model correctly detects that there is no braille char in an image, and classifies it as negative.
    """
    total = 0
    iou_sum = 0
    errors_sum = 0

    # correctly detected
    true_postives_sum = 0
    # misses
    false_negative_sum = 0
    # wrongly classified
    false_positives_sum = 0

    with torch.no_grad():
        for iter, data in enumerate(dataloader):
            images, targets = data
            # filename = targets[0]['filename']

            for i, im in enumerate(images):
                images[i] = images[i].to(device)
                targets[i]['boxes'] = targets[i]['boxes'].to(device)
                targets[i]['labels'] = targets[i]['labels'].to(device)

            outputs = model(images)

            predicted_bboxes = outputs[0]['boxes']
            predicted_scores = outputs[0]['scores']
            predicted_labels = outputs[0]['labels']
            
            total += len(targets[0]['boxes'])

            iou = ops.boxes.box_iou(targets[0]['boxes'], predicted_bboxes)
            iou_val, best_candidate_index = iou.max(1)
            iou_sum += iou_val.sum()
            
            true_postives = ((iou_val>threshold)*(targets[0]['labels']==predicted_labels[best_candidate_index])*(predicted_scores[best_candidate_index]>threshold))
            false_positives = ((iou_val>0)*(targets[0]['labels']!=predicted_labels[best_candidate_index]))
            false_negative = (iou_val==0)
            errors = false_positives+false_negative

            true_postives_sum += true_postives.sum()
            false_positives_sum += false_positives.sum()
            false_negative_sum += false_negative.sum()
            errors_sum += errors.sum()

    acc = true_postives_sum / total
    iou = iou_sum / total
    precision = true_postives_sum/(true_postives_sum+false_positives_sum)
    recall = true_postives_sum/(true_postives_sum+false_negative_sum)
    f1 = true_postives_sum/(true_postives_sum+1/2*(false_positives_sum+false_negative_sum))

    return {
        "acc": acc.item(), 
        "iou": iou.item(), 
        "tp": true_postives_sum.item(), 
        "fp": false_positives_sum.item(), 
        "fn": false_negative_sum.item(), 
        "precision": precision.item(), 
        "recall": recall.item(), 
        "f1": f1.item()
    }

# define model, loss function and optimizer
num_classes = len(set(train_dataset.labels))
model = models.detection.retinanet_resnet50_fpn_v2(
    weights=None,
    weights_backbone=models.ResNet50_Weights,
    num_classes=num_classes,
    detections_per_img=840
)

if args.weights is not None:
    model.load_state_dict(torch.load(args.weights))

model = model.to(device)

max_lr=1e-4
# optimizer = Adam(model.parameters(), lr=max_lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.5, weight_decay=1e-4)

# define the number of training steps
num_epochs = 10

# define validation accuracy
val_accuracy = 0 

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    # total_steps=batch_size*num_epochs,
    epochs=num_epochs,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.4,
    anneal_strategy='linear',
    # cycle_momentum=True,
    # base_momentum=0.85,
    # max_momentum=0.95,
    div_factor=1,
    final_div_factor=5.0,
    # three_phase=True
)

for epoch in range(num_epochs):
    # Initialize a progress bar
    progress_bar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs}')

    model.train()
    for iter, data in enumerate(train_dataloader):
        images, targets = data

        for i, im in enumerate(images):
            images[i] = images[i].to(device)

            # background
            # if len(targets[i]['boxes']) == 0:
            #     targets[i]['boxes'] = torch.tensor([[0,0,0.1,0.1]], dtype=torch.float32)

            # if len(targets[i]['labels']) == 0:
            #     targets[i]['labels'] = torch.tensor([[train_dataset.label_to_int['background']]], dtype=torch.long)

            targets[i]['boxes'] = targets[i]['boxes'].to(device)
            targets[i]['labels'] = targets[i]['labels'].to(device)

        # forward pass
        logits = model(images, targets)
        losses = sum(loss for loss in logits.values())

        # backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Step the learning rate scheduler
        scheduler.step()
        lr = scheduler.get_last_lr()

        # Update the progress bar
        progress_bar.set_postfix(train_loss=losses.item(), lr=lr[0])
        progress_bar.update()

    # Validation
    model.eval()
    with torch.no_grad():
        train_metrics = compute_acc_iou(model, train_dataloader)
        val_metrics = compute_acc_iou(model, val_dataloader)

    # Update the progress bar
    progress_bar.set_postfix(
        train_loss=losses.item(), 
        lr=lr[0],
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )

    # Update tensorboard summary
    writer.add_scalar('Loss/train', losses.item(), epoch)
    writer.add_scalar('Accuracy/train', train_metrics["acc"], epoch)
    writer.add_scalar('Accuracy/val', val_metrics["acc"], epoch)
    writer.add_scalar('IoU/train', train_metrics["iou"], epoch)
    writer.add_scalar('IoU/val', val_metrics["iou"], epoch)

    writer.add_scalar('Metrics/train_tp', train_metrics["tp"], epoch)
    writer.add_scalar('Metrics/train_fp', train_metrics["fp"], epoch)
    writer.add_scalar('Metrics/train_fn', train_metrics["fn"], epoch)
    writer.add_scalar('Metrics/train_precision', train_metrics["precision"], epoch)
    writer.add_scalar('Metrics/train_recall', train_metrics["recall"], epoch)
    writer.add_scalar('Metrics/train_f1', train_metrics["f1"], epoch)

    writer.add_scalar('Metrics/val_tp', val_metrics["tp"], epoch)
    writer.add_scalar('Metrics/val_fp', val_metrics["fp"], epoch)
    writer.add_scalar('Metrics/val_fn', val_metrics["fn"], epoch)
    writer.add_scalar('Metrics/val_precision', val_metrics["precision"], epoch)
    writer.add_scalar('Metrics/val_recall', val_metrics["recall"], epoch)
    writer.add_scalar('Metrics/val_f1', val_metrics["f1"], epoch)

    # save best checkpoint the model
    if val_accuracy < val_metrics["acc"]:
        torch.save(model.state_dict(), f'weights/model-{num_experiment}-{val_metrics["acc"]:.3f}.pth')
        if os.path.exists(f'weights/model-{num_experiment}-{val_accuracy:.3f}.pth'):
            os.remove(f'weights/model-{num_experiment}-{val_accuracy:.3f}.pth')     
        val_accuracy = val_metrics["acc"]

        print(
           "train_loss:", losses.item(), 
           "train_metrics:", train_metrics,
           "val_metrics:", val_metrics,
        )
