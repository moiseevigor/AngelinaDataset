import os, json, pprint, copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torchvision.ops as ops

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.optim import Adam
from torch.nn import BCELoss
from torchvision import transforms
from tqdm import tqdm

# model-3, 116
# label_to_int = {'!': 0, '##': 1, '(': 2, '()': 3, ')': 4, '*': 5, '+': 6, ',': 7, '-': 8, '.': 9, '..': 10, '/': 11, '/0': 12, '/1': 13, '/2': 14, '/3': 15, '/4': 16, '/5': 17, '/6': 18, '/7': 19, '/8': 20, '/9': 21, '0': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, ':': 32, '::': 33, ';': 34, '=': 35, '>>': 36, '?': 37, 'CC': 38, 'XX': 39, 'a': 40, 'b': 41, 'c': 42, 'd': 43, 'e': 44, 'en': 45, 'f': 46, 'i': 47, 'k': 48, 'l': 49, 'm': 50, 'n': 51, 'o': 52, 'q': 53, 'r': 54, 's': 55, 't': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61, '{': 62, '|': 63, '}': 64, '~1236': 65, '~12456': 66, '~13456': 67, '~1456': 68, '~3': 69, '~34': 70, '~346': 71, '~4': 72, '~46': 73, '~5': 74, '~56': 75, '~6': 76, '§': 77, '«': 78, '»': 79, 'Н': 80, 'СС': 81, 'ХХ': 82, 'а': 83, 'б': 84, 'в': 85, 'г': 86, 'д': 87, 'е': 88, 'ж': 89, 'з': 90, 'и': 91, 'й': 92, 'к': 93, 'л': 94, 'м': 95, 'н': 96, 'о': 97, 'п': 98, 'р': 99, 'с': 100, 'т': 101, 'у': 102, 'ф': 103, 'х': 104, 'ц': 105, 'ч': 106, 'ш': 107, 'щ': 108, 'ъ': 109, 'ы': 110, 'ь': 111, 'э': 112, 'ю': 113, 'я': 114, 'ё': 115, 'p': 116, 'В': 117, 'Л': 118}
# int_to_label = {0: '!', 1: '##', 2: '(', 3: '()', 4: ')', 5: '*', 6: '+', 7: ',', 8: '-', 9: '.', 10: '..', 11: '/', 12: '/0', 13: '/1', 14: '/2', 15: '/3', 16: '/4', 17: '/5', 18: '/6', 19: '/7', 20: '/8', 21: '/9', 22: '0', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: ':', 33: '::', 34: ';', 35: '=', 36: '>>', 37: '?', 38: 'CC', 39: 'XX', 40: 'a', 41: 'b', 42: 'c', 43: 'd', 44: 'e', 45: 'en', 46: 'f', 47: 'i', 48: 'k', 49: 'l', 50: 'm', 51: 'n', 52: 'o', 53: 'q', 54: 'r', 55: 's', 56: 't', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z', 62: '{', 63: '|', 64: '}', 65: '~1236', 66: '~12456', 67: '~13456', 68: '~1456', 69: '~3', 70: '~34', 71: '~346', 72: '~4', 73: '~46', 74: '~5', 75: '~56', 76: '~6', 77: '§', 78: '«', 79: '»', 80: 'Н', 81: 'СС', 82: 'ХХ', 83: 'а', 84: 'б', 85: 'в', 86: 'г', 87: 'д', 88: 'е', 89: 'ж', 90: 'з', 91: 'и', 92: 'й', 93: 'к', 94: 'л', 95: 'м', 96: 'н', 97: 'о', 98: 'п', 99: 'р', 100: 'с', 101: 'т', 102: 'у', 103: 'ф', 104: 'х', 105: 'ц', 106: 'ч', 107: 'ш', 108: 'щ', 109: 'ъ', 110: 'ы', 111: 'ь', 112: 'э', 113: 'ю', 114: 'я', 115: 'ё', 116: 'p', 117: 'В', 118: 'Л'}

# model-6, 126
# label_to_int = {'!': 0, '##': 1, '(': 2, '()': 3, ')': 4, '*': 5, '+': 6, ',': 7, '-': 8, '.': 9, '..': 10, '/': 11, '/0': 12, '/1': 13, '/2': 14, '/3': 15, '/4': 16, '/5': 17, '/6': 18, '/7': 19, '/8': 20, '/9': 21, '0': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, ':': 32, '::': 33, ';': 34, '=': 35, '>>': 36, '?': 37, 'CC': 38, 'XX': 39, 'a': 40, 'b': 41, 'c': 42, 'd': 43, 'e': 44, 'en': 45, 'f': 46, 'i': 47, 'k': 48, 'l': 49, 'm': 50, 'n': 51, 'o': 52, 'p': 53, 'q': 54, 'r': 55, 's': 56, 't': 57, 'v': 58, 'w': 59, 'x': 60, 'y': 61, 'z': 62, '{': 63, '|': 64, '}': 65, '~1236': 66, '~12456': 67, '~13456': 68, '~1456': 69, '~3': 70, '~34': 71, '~346': 72, '~4': 73, '~46': 74, '~5': 75, '~56': 76, '~6': 77, '§': 78, '«': 79, '»': 80, 'В': 81, 'И': 82, 'Л': 83, 'М': 84, 'Н': 85, 'О': 86, 'П': 87, 'С': 88, 'СС': 89, 'Т': 90, 'Ф': 91, 'ХХ': 92, 'а': 93, 'б': 94, 'в': 95, 'г': 96, 'д': 97, 'е': 98, 'ж': 99, 'з': 100, 'и': 101, 'й': 102, 'к': 103, 'л': 104, 'м': 105, 'н': 106, 'о': 107, 'п': 108, 'р': 109, 'с': 110, 'т': 111, 'у': 112, 'ф': 113, 'х': 114, 'ц': 115, 'ч': 116, 'ш': 117, 'щ': 118, 'ъ': 119, 'ы': 120, 'ь': 121, 'э': 122, 'ю': 123, 'я': 124, 'ё': 125}
# int_to_label = {0: '!', 1: '##', 2: '(', 3: '()', 4: ')', 5: '*', 6: '+', 7: ',', 8: '-', 9: '.', 10: '..', 11: '/', 12: '/0', 13: '/1', 14: '/2', 15: '/3', 16: '/4', 17: '/5', 18: '/6', 19: '/7', 20: '/8', 21: '/9', 22: '0', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: ':', 33: '::', 34: ';', 35: '=', 36: '>>', 37: '?', 38: 'CC', 39: 'XX', 40: 'a', 41: 'b', 42: 'c', 43: 'd', 44: 'e', 45: 'en', 46: 'f', 47: 'i', 48: 'k', 49: 'l', 50: 'm', 51: 'n', 52: 'o', 53: 'p', 54: 'q', 55: 'r', 56: 's', 57: 't', 58: 'v', 59: 'w', 60: 'x', 61: 'y', 62: 'z', 63: '{', 64: '|', 65: '}', 66: '~1236', 67: '~12456', 68: '~13456', 69: '~1456', 70: '~3', 71: '~34', 72: '~346', 73: '~4', 74: '~46', 75: '~5', 76: '~56', 77: '~6', 78: '§', 79: '«', 80: '»', 81: 'В', 82: 'И', 83: 'Л', 84: 'М', 85: 'Н', 86: 'О', 87: 'П', 88: 'С', 89: 'СС', 90: 'Т', 91: 'Ф', 92: 'ХХ', 93: 'а', 94: 'б', 95: 'в', 96: 'г', 97: 'д', 98: 'е', 99: 'ж', 100: 'з', 101: 'и', 102: 'й', 103: 'к', 104: 'л', 105: 'м', 106: 'н', 107: 'о', 108: 'п', 109: 'р', 110: 'с', 111: 'т', 112: 'у', 113: 'ф', 114: 'х', 115: 'ц', 116: 'ч', 117: 'ш', 118: 'щ', 119: 'ъ', 120: 'ы', 121: 'ь', 122: 'э', 123: 'ю', 124: 'я', 125: 'ё'}

# model-7, 127
label_to_int = {'!': 0, '##': 1, '(': 2, '()': 3, ')': 4, '*': 5, '+': 6, ',': 7, '-': 8, '.': 9, '..': 10, '/': 11, '/0': 12, '/1': 13, '/2': 14, '/3': 15, '/4': 16, '/5': 17, '/6': 18, '/7': 19, '/8': 20, '/9': 21, '0': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, ':': 32, '::': 33, ';': 34, '=': 35, '>>': 36, '?': 37, 'CC': 38, 'XX': 39, 'a': 40, 'b': 41, 'background': 42, 'c': 43, 'd': 44, 'e': 45, 'en': 46, 'f': 47, 'i': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'v': 59, 'w': 60, 'x': 61, 'y': 62, 'z': 63, '{': 64, '|': 65, '}': 66, '~1236': 67, '~12456': 68, '~13456': 69, '~1456': 70, '~3': 71, '~34': 72, '~346': 73, '~4': 74, '~46': 75, '~5': 76, '~56': 77, '~6': 78, '§': 79, '«': 80, '»': 81, 'В': 82, 'И': 83, 'Л': 84, 'М': 85, 'Н': 86, 'О': 87, 'П': 88, 'С': 89, 'СС': 90, 'Т': 91, 'Ф': 92, 'ХХ': 93, 'а': 94, 'б': 95, 'в': 96, 'г': 97, 'д': 98, 'е': 99, 'ж': 100, 'з': 101, 'и': 102, 'й': 103, 'к': 104, 'л': 105, 'м': 106, 'н': 107, 'о': 108, 'п': 109, 'р': 110, 'с': 111, 'т': 112, 'у': 113, 'ф': 114, 'х': 115, 'ц': 116, 'ч': 117, 'ш': 118, 'щ': 119, 'ъ': 120, 'ы': 121, 'ь': 122, 'э': 123, 'ю': 124, 'я': 125, 'ё': 126}
int_to_label = {0: '!', 1: '##', 2: '(', 3: '()', 4: ')', 5: '*', 6: '+', 7: ',', 8: '-', 9: '.', 10: '..', 11: '/', 12: '/0', 13: '/1', 14: '/2', 15: '/3', 16: '/4', 17: '/5', 18: '/6', 19: '/7', 20: '/8', 21: '/9', 22: '0', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: ':', 33: '::', 34: ';', 35: '=', 36: '>>', 37: '?', 38: 'CC', 39: 'XX', 40: 'a', 41: 'b', 42: 'background', 43: 'c', 44: 'd', 45: 'e', 46: 'en', 47: 'f', 48: 'i', 49: 'k', 50: 'l', 51: 'm', 52: 'n', 53: 'o', 54: 'p', 55: 'q', 56: 'r', 57: 's', 58: 't', 59: 'v', 60: 'w', 61: 'x', 62: 'y', 63: 'z', 64: '{', 65: '|', 66: '}', 67: '~1236', 68: '~12456', 69: '~13456', 70: '~1456', 71: '~3', 72: '~34', 73: '~346', 74: '~4', 75: '~46', 76: '~5', 77: '~56', 78: '~6', 79: '§', 80: '«', 81: '»', 82: 'В', 83: 'И', 84: 'Л', 85: 'М', 86: 'Н', 87: 'О', 88: 'П', 89: 'С', 90: 'СС', 91: 'Т', 92: 'Ф', 93: 'ХХ', 94: 'а', 95: 'б', 96: 'в', 97: 'г', 98: 'д', 99: 'е', 100: 'ж', 101: 'з', 102: 'и', 103: 'й', 104: 'к', 105: 'л', 106: 'м', 107: 'н', 108: 'о', 109: 'п', 110: 'р', 111: 'с', 112: 'т', 113: 'у', 114: 'ф', 115: 'х', 116: 'ц', 117: 'ч', 118: 'ш', 119: 'щ', 120: 'ъ', 121: 'ы', 122: 'ь', 123: 'э', 124: 'ю', 125: 'я', 126: 'ё'}

labels = sorted(set(int_to_label.values()))
print('labels', labels)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

mean=[0.5749533646009656, 0.5758692075743113, 0.5564374772810018]
std=[0.12675546510063618, 0.13864833881922706, 0.14966126335877825]
mean = torch.tensor(mean).view(-1, 1, 1)
std = torch.tensor(std).view(-1, 1, 1)

class RetinaNetDataset(Dataset):
    def __init__(self, train_annotation_files, transform=None, labels=None):
        self.transform = transform
        self.train_annotation_files = train_annotation_files
        self.annotations = []
        self.set_labels = labels is not None
        self.labels = ['background']
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
                        self.labels.append(shape["label"])

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
            labels.append(self.label_to_int[shape["label"]])

            # Extract x, y coordinates of the bounding box
            x1, y1 = shape["points"][0]
            x2, y2 = shape["points"][1]
            # Append a bounding box coordinates to the boxes list
            boxes.append([x1, y1, x2, y2])

        return image, {
            'boxes': torch.tensor(boxes),
            'labels': torch.tensor(labels),
            'filename': annotation["imagePath"]
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

def plot_image(ax, plt, prediction, is_gt=True):
    # Get the boxes, labels, and scores from the prediction results
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"] if "scores" in prediction else torch.ones(labels.size())

    edge_color = 'lightgreen' if is_gt else 'r'

    # Iterate over the boxes, labels, and scores
    for box, label, score in zip(boxes, labels, scores):
        # Get the coordinates of the bounding box
        xmin, ymin, xmax, ymax = box.detach().cpu().numpy()

        # Create a rectangle patch
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor=edge_color, facecolor='none')

        # Add the patch to the axes
        ax.add_patch(rect)

        label_text = int_to_label[int(label.detach().cpu().numpy())]

        # Add the label and score as text
        if is_gt:
            plt.text(xmin, ymin-5, f"{label_text}", fontsize=12, color=edge_color)
        else:
            plt.text(xmin, ymax+10, f"{label_text}: {score:0.2f}", fontsize=12, color=edge_color)

    # plt.show()


annotation_files = []
for filepath in [
    # '/app/books/train.txt',
    # '/app/not_braille/train.txt',
    # '/app/handwritten/train.txt',
    # '/app/uploaded/test2.txt',
    '/app/books/val.txt',
    # '/app/handwritten/val.txt',
]:
    with open(filepath, 'r') as file:
        annotation_files.extend([os.path.join(os.path.dirname(filepath), line.strip().replace('.jpg', '.json')) for line in file.readlines()])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean, 
        std=std
    )
])
dataset = RetinaNetDataset(annotation_files, transform=transform, labels=labels)
print('label_to_int:', dataset.label_to_int)
print('int_to_label:', dataset.int_to_label)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=labelbox_collate_fn)

# def bbox_iou(predicted_bboxes, target_bboxes):
#     """
#     Computes IoU between two bounding boxes
#     """
#     pred_x1, pred_y1, pred_x2, pred_y2 = torch.unbind(predicted_bboxes, dim=1)
#     target_x1, target_y1, target_x2, target_y2 = torch.unbind(target_bboxes, dim=0)

#     # width and height of the intersection box
#     w_intsec = torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1)
#     h_intsec = torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1)
#     # Clamp negative values to zero
#     w_intsec = torch.clamp(w_intsec, min=0)
#     h_intsec = torch.clamp(h_intsec, min=0)

#     # area of intersection box
#     area_int = w_intsec * h_intsec
#     # area of predicted and target boxes
#     area_pred = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
#     area_target = (target_x2 - target_x1) * (target_y2 - target_y1)

#     # Compute IoU
#     return area_int / (area_pred + area_target - area_int)

def compute_acc_iou(model, dataloader, threshold=0.5):
    """
    Computes accuracy and average IoU
    """
    global mean, std

    model.eval()
    correct = 0
    total = 0
    iou_sum = 0
    progress_bar = tqdm(total=len(dataloader), desc=f'Iter')

    with torch.no_grad():
        for iter, data in enumerate(dataloader):
            images, targets = data
            filename = targets[0]['filename']

            for i, im in enumerate(images):
                images[i] = images[i].to(device)
                targets[i]['boxes'] = targets[i]['boxes'].to(device)
                targets[i]['labels'] = targets[i]['labels'].to(device)

            outputs = model(images)

            predicted_bboxes = outputs[0]['boxes']
            predicted_scores = outputs[0]['scores']
            predicted_labels = outputs[0]['labels']
            
            total += len(targets[0]['boxes'])
            errors = {
                'boxes': [],
                'labels': [],
                'scores': []
            }
            for i in range(len(targets[0]['boxes'])):
                # find the best candidate for targets['boxes'] among predicted_bboxes
                # iou = bbox_iou(predicted_bboxes, targets[0]['boxes'][i])
                iou = ops.boxes.box_iou(predicted_bboxes, targets[0]['boxes'][i].view(1, -1))
                if len(iou) <= 0: continue

                best_candidate_index = iou.argmax()
                best_candidate_label = predicted_labels[best_candidate_index]
                best_candidate_score = predicted_scores[best_candidate_index]
                iou_sum += iou.max()

                if iou.max() > threshold and best_candidate_label == targets[0]['labels'][i]: # and best_candidate_score > threshold:
                    # overlap is over threshold and label is correct
                    correct += 1

                    # if targets[0]['labels'][i].item() == 102:
                    #     # и
                    #     print('bingo')

                # elif iou.max() > threshold and best_candidate_score>0.4:
                #   print(best_candidate_label.item(), targets[0]['labels'][i].item(), dataset.int_to_label[targets[0]['labels'][i].item()], best_candidate_score.item())

                elif best_candidate_label != targets[0]['labels'][i] and best_candidate_score > threshold:
                    print(predicted_labels, predicted_labels[best_candidate_index], iou)
                    errors['boxes'].append(predicted_bboxes[best_candidate_index])
                    errors['labels'].append(predicted_labels[best_candidate_index])
                    errors['scores'].append(predicted_scores[best_candidate_index])

                #     print(
                #         best_candidate_label.item(), 
                #         targets[0]['labels'][i].item(), 
                #         dataset.int_to_label[targets[0]['labels'][i].item()], 
                #         best_candidate_score.item(),
                #         iou.max(),
                #     )

                    # plot_image(images[0].cpu(), {
                    #     'boxes': [predicted_bboxes[best_candidate_index], targets[0]['boxes'][i]],
                    #     'labels': [predicted_labels[best_candidate_index], targets[0]['labels'][i]],
                    #     'scores': [predicted_scores[best_candidate_index], 1]
                    # })
            
            if len(errors['boxes'])>0:
                # Create a figure and axes
                # Reshaping the mean and std
                image = images[0].cpu()
                image = image * std + mean

                fig, ax = plt.subplots(1)
                ax.imshow(to_pil_image(image))
            
                print("filename", filename)

                plot_image(ax, plt, targets[0], is_gt=True)
                plot_image(ax, plt, errors, is_gt=False)
            
                plt.show()

            progress_bar.set_postfix(
                iter=iter,
                total=total,
                correct=correct,
                acc=correct/total,
                iou=(iou_sum/total).item()
            )
            progress_bar.update()


    acc = correct / total
    iou = iou_sum / total

    return acc, iou

# define model, loss function and optimizer
num_classes = len(set(dataset.labels))
model = models.detection.retinanet_resnet50_fpn_v2(
    weights=None,
    weights_backbone=models.ResNet50_Weights,
    num_classes=num_classes,
    # fg_iou_thresh=0.2,
    # bg_iou_thresh=0.1
)

# model_name='model-3.pth'
# model_name='model-6-0.823.pth'
# model_name='model-7-0.778.pth'
# model_name='model-8-0.847.pth'
# model_name='model-9-0.862.pth'
model_name='model-12-0.865.pth'

model.load_state_dict(torch.load('weights/' + model_name))
model = model.to(device)

# Validation
model.eval()
with torch.no_grad():
    val_acc, val_iou = compute_acc_iou(model, dataloader)

print(
    model_name,
    'val_acc', val_acc, 
    'val_iou', val_iou.item()
)
