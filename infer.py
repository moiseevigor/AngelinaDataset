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

# image = Image.open("/app/books/chudo_derevo_redmi/IMG_20190715_112912.labeled.jpg")
# image = Image.open("/app/books/chudo_derevo_redmi/IMG_20190715_113048.labeled.jpg")
# image = Image.open("/app/test/foto-braille-batch-1/5F804152-AF64-48EA-9E0D-15476EC82A56.JPG")
# image = Image.open("/app/test/foto-braille-batch-2/8A2551D7-745F-4517-B401-F5C43D3308D4.JPG")
# image = Image.open("/app/test/foto-braille-batch-1/18B9ABA9-9D54-462B-8D85-88646D0498F2.JPG")
# image = Image.open("/app/test/2.png")
# image = Image.open("/app/DSBI/data/Fundamentals of Massage/FM+1.jpg")
# image = Image.open("/app/DSBI/data/Ordinary Printed Document/OPD+1.jpg")
# image = Image.open("/app/DSBI/data/Shaver Yang Fengting/SYF+3.jpg")
# image = Image.open("/app/DSBI/data/The Second Volume of Ninth Grade Chinese Book 1/SVNGCB1+1.jpg")
# image = Image.open("/app/DSBI/data/The Second Volume of Ninth Grade Chinese Book 1/SVNGCB1+2.jpg")
# image = Image.open("/app/test/IMG_1969.jpg")
image = Image.open("/app/test/pagina_braille_fotografata_lowres.jpg")
# image = Image.open("/app/test/pagina_braille_scannerizzata_1.jpg")
# image = Image.open("/app/test/pagina_braille_scannerizzata_2.jpeg")
# image = Image.open("/app/test/braille.png")
# image = Image.open("/app/test/chemistry.webp")
# image = Image.open("/app/test/text-1.jpeg")
# image = Image.open("/app/test/braille-outdoor.jpg")


# model 1, 83
# label_to_int = {'б': 0, 'с': 1, 's': 2, '+': 3, '?': 4, 'ю': 5, '=': 6, '0': 7, 'д': 8, '.': 9, 'ь': 10, '~46': 11, '()': 12, ':': 13, 'k': 14, '4': 15, '~4': 16, 'м': 17, 't': 18, 'm': 19, 'ы': 20, 'ф': 21, '1': 22, '>>': 23, 'и': 24, 'е': 25, 'г': 26, '{': 27, 'l': 28, '6': 29, 'у': 30, 'я': 31, 'о': 32, 'o': 33, 'ё': 34, 'т': 35, '/4': 36, ')': 37, '-': 38, 'e': 39, 'z': 40, '!': 41, '§': 42, '3': 43, 'а': 44, '»': 45, 'a': 46, 'n': 47, '«': 48, 'ъ': 49, 'к': 50, 'XX': 51, 'щ': 52, '7': 53, 'q': 54, 'л': 55, 'в': 56, '8': 57, 'п': 58, ';': 59, 'р': 60, '5': 61, 'э': 62, '9': 63, '(': 64, '2': 65, 'й': 66, 'ж': 67, ',': 68, 'ч': 69, '|': 70, 'w': 71, 'ц': 72, 'х': 73, 'н': 74, 'ш': 75, '##': 76, '~3': 77, 'd': 78, '}': 79, '/1': 80, 'з': 81, 'CC': 82}
# int_to_label = {0: 'б', 1: 'с', 2: 's', 3: '+', 4: '?', 5: 'ю', 6: '=', 7: '0', 8: 'д', 9: '.', 10: 'ь', 11: '~46', 12: '()', 13: ':', 14: 'k', 15: '4', 16: '~4', 17: 'м', 18: 't', 19: 'm', 20: 'ы', 21: 'ф', 22: '1', 23: '>>', 24: 'и', 25: 'е', 26: 'г', 27: '{', 28: 'l', 29: '6', 30: 'у', 31: 'я', 32: 'о', 33: 'o', 34: 'ё', 35: 'т', 36: '/4', 37: ')', 38: '-', 39: 'e', 40: 'z', 41: '!', 42: '§', 43: '3', 44: 'а', 45: '»', 46: 'a', 47: 'n', 48: '«', 49: 'ъ', 50: 'к', 51: 'XX', 52: 'щ', 53: '7', 54: 'q', 55: 'л', 56: 'в', 57: '8', 58: 'п', 59: ';', 60: 'р', 61: '5', 62: 'э', 63: '9', 64: '(', 65: '2', 66: 'й', 67: 'ж', 68: ',', 69: 'ч', 70: '|', 71: 'w', 72: 'ц', 73: 'х', 74: 'н', 75: 'ш', 76: '##', 77: '~3', 78: 'd', 79: '}', 80: '/1', 81: 'з', 82: 'CC'}

# model 2, 116
# label_to_int = {'ф': 0, 'Н': 1, '9': 2, '}': 3, 'СС': 4, '##': 5, 'x': 6, '/4': 7, '0': 8, 'х': 9, '~12456': 10, 'f': 11, 'ж': 12, '/6': 13, '>>': 14, '/5': 15, 'ю': 16, '?': 17, 'ХХ': 18, 'i': 19, '~56': 20, 'д': 21, ':': 22, '/7': 23, '~34': 24, 'п': 25, 'k': 26, '5': 27, 'р': 28, '/8': 29, '~4': 30, 'ц': 31, 'й': 32, '~1456': 33, '~6': 34, '~346': 35, '{': 36, 'т': 37, '/2': 38, 'з': 39, '8': 40, 'a': 41, '~46': 42, '.': 43, 'm': 44, '!': 45, 'e': 46, 'о': 47, 'CC': 48, 'а': 49, '/3': 50, 'л': 51, 'и': 52, 't': 53, '|': 54, 'XX': 55, 'c': 56, 'е': 57, '()': 58, 'l': 59, '«': 60, 'ъ': 61, 'w': 62, 'ш': 63, '4': 64, '/9': 65, 'n': 66, 'щ': 67, 'ы': 68, 'м': 69, 'я': 70, 'у': 71, 'н': 72, '1': 73, 'o': 74, '~5': 75, '/0': 76, '§': 77, 'y': 78, '(': 79, 'б': 80, 'к': 81, 'ч': 82, 'в': 83, 'r': 84, 'э': 85, '7': 86, 's': 87, '*': 88, '2': 89, 'en': 90, '/': 91, 'b': 92, ')': 93, 'г': 94, '..': 95, 'с': 96, '/1': 97, '~1236': 98, '=': 99, '~3': 100, '3': 101, 'q': 102, 'd': 103, '::': 104, ',': 105, '~13456': 106, '6': 107, ';': 108, '+': 109, '»': 110, 'v': 111, 'z': 112, 'ё': 113, 'ь': 114, '-': 115}
# int_to_label = {0: 'ф', 1: 'Н', 2: '9', 3: '}', 4: 'СС', 5: '##', 6: 'x', 7: '/4', 8: '0', 9: 'х', 10: '~12456', 11: 'f', 12: 'ж', 13: '/6', 14: '>>', 15: '/5', 16: 'ю', 17: '?', 18: 'ХХ', 19: 'i', 20: '~56', 21: 'д', 22: ':', 23: '/7', 24: '~34', 25: 'п', 26: 'k', 27: '5', 28: 'р', 29: '/8', 30: '~4', 31: 'ц', 32: 'й', 33: '~1456', 34: '~6', 35: '~346', 36: '{', 37: 'т', 38: '/2', 39: 'з', 40: '8', 41: 'a', 42: '~46', 43: '.', 44: 'm', 45: '!', 46: 'e', 47: 'о', 48: 'CC', 49: 'а', 50: '/3', 51: 'л', 52: 'и', 53: 't', 54: '|', 55: 'XX', 56: 'c', 57: 'е', 58: '()', 59: 'l', 60: '«', 61: 'ъ', 62: 'w', 63: 'ш', 64: '4', 65: '/9', 66: 'n', 67: 'щ', 68: 'ы', 69: 'м', 70: 'я', 71: 'у', 72: 'н', 73: '1', 74: 'o', 75: '~5', 76: '/0', 77: '§', 78: 'y', 79: '(', 80: 'б', 81: 'к', 82: 'ч', 83: 'в', 84: 'r', 85: 'э', 86: '7', 87: 's', 88: '*', 89: '2', 90: 'en', 91: '/', 92: 'b', 93: ')', 94: 'г', 95: '..', 96: 'с', 97: '/1', 98: '~1236', 99: '=', 100: '~3', 101: '3', 102: 'q', 103: 'd', 104: '::', 105: ',', 106: '~13456', 107: '6', 108: ';', 109: '+', 110: '»', 111: 'v', 112: 'z', 113: 'ё', 114: 'ь', 115: '-'}

# model 3, 116
# label_to_int = {'!': 0, '##': 1, '(': 2, '()': 3, ')': 4, '*': 5, '+': 6, ',': 7, '-': 8, '.': 9, '..': 10, '/': 11, '/0': 12, '/1': 13, '/2': 14, '/3': 15, '/4': 16, '/5': 17, '/6': 18, '/7': 19, '/8': 20, '/9': 21, '0': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, ':': 32, '::': 33, ';': 34, '=': 35, '>>': 36, '?': 37, 'CC': 38, 'XX': 39, 'a': 40, 'b': 41, 'c': 42, 'd': 43, 'e': 44, 'en': 45, 'f': 46, 'i': 47, 'k': 48, 'l': 49, 'm': 50, 'n': 51, 'o': 52, 'q': 53, 'r': 54, 's': 55, 't': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61, '{': 62, '|': 63, '}': 64, '~1236': 65, '~12456': 66, '~13456': 67, '~1456': 68, '~3': 69, '~34': 70, '~346': 71, '~4': 72, '~46': 73, '~5': 74, '~56': 75, '~6': 76, '§': 77, '«': 78, '»': 79, 'Н': 80, 'СС': 81, 'ХХ': 82, 'а': 83, 'б': 84, 'в': 85, 'г': 86, 'д': 87, 'е': 88, 'ж': 89, 'з': 90, 'и': 91, 'й': 92, 'к': 93, 'л': 94, 'м': 95, 'н': 96, 'о': 97, 'п': 98, 'р': 99, 'с': 100, 'т': 101, 'у': 102, 'ф': 103, 'х': 104, 'ц': 105, 'ч': 106, 'ш': 107, 'щ': 108, 'ъ': 109, 'ы': 110, 'ь': 111, 'э': 112, 'ю': 113, 'я': 114, 'ё': 115}
# int_to_label = {0: '!', 1: '##', 2: '(', 3: '()', 4: ')', 5: '*', 6: '+', 7: ',', 8: '-', 9: '.', 10: '..', 11: '/', 12: '/0', 13: '/1', 14: '/2', 15: '/3', 16: '/4', 17: '/5', 18: '/6', 19: '/7', 20: '/8', 21: '/9', 22: '0', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: ':', 33: '::', 34: ';', 35: '=', 36: '>>', 37: '?', 38: 'CC', 39: 'XX', 40: 'a', 41: 'b', 42: 'c', 43: 'd', 44: 'e', 45: 'en', 46: 'f', 47: 'i', 48: 'k', 49: 'l', 50: 'm', 51: 'n', 52: 'o', 53: 'q', 54: 'r', 55: 's', 56: 't', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z', 62: '{', 63: '|', 64: '}', 65: '~1236', 66: '~12456', 67: '~13456', 68: '~1456', 69: '~3', 70: '~34', 71: '~346', 72: '~4', 73: '~46', 74: '~5', 75: '~56', 76: '~6', 77: '§', 78: '«', 79: '»', 80: 'Н', 81: 'СС', 82: 'ХХ', 83: 'а', 84: 'б', 85: 'в', 86: 'г', 87: 'д', 88: 'е', 89: 'ж', 90: 'з', 91: 'и', 92: 'й', 93: 'к', 94: 'л', 95: 'м', 96: 'н', 97: 'о', 98: 'п', 99: 'р', 100: 'с', 101: 'т', 102: 'у', 103: 'ф', 104: 'х', 105: 'ц', 106: 'ч', 107: 'ш', 108: 'щ', 109: 'ъ', 110: 'ы', 111: 'ь', 112: 'э', 113: 'ю', 114: 'я', 115: 'ё'}

# model-7, 127
label_to_int = {'!': 0, '##': 1, '(': 2, '()': 3, ')': 4, '*': 5, '+': 6, ',': 7, '-': 8, '.': 9, '..': 10, '/': 11, '/0': 12, '/1': 13, '/2': 14, '/3': 15, '/4': 16, '/5': 17, '/6': 18, '/7': 19, '/8': 20, '/9': 21, '0': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, ':': 32, '::': 33, ';': 34, '=': 35, '>>': 36, '?': 37, 'CC': 38, 'XX': 39, 'a': 40, 'b': 41, 'background': 42, 'c': 43, 'd': 44, 'e': 45, 'en': 46, 'f': 47, 'i': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'v': 59, 'w': 60, 'x': 61, 'y': 62, 'z': 63, '{': 64, '|': 65, '}': 66, '~1236': 67, '~12456': 68, '~13456': 69, '~1456': 70, '~3': 71, '~34': 72, '~346': 73, '~4': 74, '~46': 75, '~5': 76, '~56': 77, '~6': 78, '§': 79, '«': 80, '»': 81, 'В': 82, 'И': 83, 'Л': 84, 'М': 85, 'Н': 86, 'О': 87, 'П': 88, 'С': 89, 'СС': 90, 'Т': 91, 'Ф': 92, 'ХХ': 93, 'а': 94, 'б': 95, 'в': 96, 'г': 97, 'д': 98, 'е': 99, 'ж': 100, 'з': 101, 'и': 102, 'й': 103, 'к': 104, 'л': 105, 'м': 106, 'н': 107, 'о': 108, 'п': 109, 'р': 110, 'с': 111, 'т': 112, 'у': 113, 'ф': 114, 'х': 115, 'ц': 116, 'ч': 117, 'ш': 118, 'щ': 119, 'ъ': 120, 'ы': 121, 'ь': 122, 'э': 123, 'ю': 124, 'я': 125, 'ё': 126}
int_to_label = {0: '!', 1: '##', 2: '(', 3: '()', 4: ')', 5: '*', 6: '+', 7: ',', 8: '-', 9: '.', 10: '..', 11: '/', 12: '/0', 13: '/1', 14: '/2', 15: '/3', 16: '/4', 17: '/5', 18: '/6', 19: '/7', 20: '/8', 21: '/9', 22: '0', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: ':', 33: '::', 34: ';', 35: '=', 36: '>>', 37: '?', 38: 'CC', 39: 'XX', 40: 'a', 41: 'b', 42: 'background', 43: 'c', 44: 'd', 45: 'e', 46: 'en', 47: 'f', 48: 'i', 49: 'k', 50: 'l', 51: 'm', 52: 'n', 53: 'o', 54: 'p', 55: 'q', 56: 'r', 57: 's', 58: 't', 59: 'v', 60: 'w', 61: 'x', 62: 'y', 63: 'z', 64: '{', 65: '|', 66: '}', 67: '~1236', 68: '~12456', 69: '~13456', 70: '~1456', 71: '~3', 72: '~34', 73: '~346', 74: '~4', 75: '~46', 76: '~5', 77: '~56', 78: '~6', 79: '§', 80: '«', 81: '»', 82: 'В', 83: 'И', 84: 'Л', 85: 'М', 86: 'Н', 87: 'О', 88: 'П', 89: 'С', 90: 'СС', 91: 'Т', 92: 'Ф', 93: 'ХХ', 94: 'а', 95: 'б', 96: 'в', 97: 'г', 98: 'д', 99: 'е', 100: 'ж', 101: 'з', 102: 'и', 103: 'й', 104: 'к', 105: 'л', 106: 'м', 107: 'н', 108: 'о', 109: 'п', 110: 'р', 111: 'с', 112: 'т', 113: 'у', 114: 'ф', 115: 'х', 116: 'ц', 117: 'ч', 118: 'ш', 119: 'щ', 120: 'ъ', 121: 'ы', 122: 'ь', 123: 'э', 124: 'ю', 125: 'я', 126: 'ё'}

RUSSIAN_TO_ITALIAN = {
    'а': 'a',
    'б': 'b',
    'в': 'v',
    'г': 'g',
    'д': 'd',
    'е': 'e',
    'ё': 'e',
    'ж': 'z',
    'з': 'z',
    'и': 'i',
    'й': 'i',
    'к': 'k',
    'л': 'l',
    'м': 'm',
    'н': 'n',
    'о': 'o',
    'п': 'p',
    'р': 'r',
    'с': 's',
    'т': 't',
    'у': 'u',
    'ф': 'f',
    'х': 'h',
    'ц': 'c',
    'ч': 'c',
    'ш': 's',
    'щ': 's',
    'ъ': '',
    'ы': 'y',
    'ь': '',
    'э': 'e',
    'ю': 'u',
    'я': 'ya',
    ' ': ' ',
    '.': '.',
    ',': ',',
    ':': ':',
    ';': ';',
    '!': '!',
    '?': '?',
    '-': '-',
    '(': '(',
    ')': ')',
    '"': '"',
    "'": "'",
    '#': '#',
    '№': '№',
    '%': '%',
    '&': '&',
    '*': '*',
    '/': '/',
    '\\': '\\',
    '@': '@',
    '^': '^',
    '+': '+',
    '=': '=',
    '$': '$',
    '€': '€',
    '£': '£',
    '¥': '¥',
    '<': '<',
    '>': '>',
    '{': '{',
    '}': '}',
    '[': '[',
    ']': ']',
    '~': '~',
    '`': '`',
    '_': '_',
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9'
}

def russian_to_italian(text):
    return ''.join(RUSSIAN_TO_ITALIAN.get(c, c) for c in text)


model = retinanet_resnet50_fpn_v2(num_classes=127, score_thresh=0.45, detections_per_img=840)
# model.load_state_dict(torch.load('weights/model-9-0.862.pth'))
model.load_state_dict(torch.load('weights/model-12-0.865.pth'))
# model.load_state_dict(torch.load('weights/model-19-0.937.pth'))

# Put the model in inference mode
model.eval()

mean=[0.5749533646009656, 0.5758692075743113, 0.5564374772810018]
std=[0.12675546510063618, 0.13864833881922706, 0.14966126335877825]

# Define multiple transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize([1376, 1024]),
    # transforms.RandomCrop([1376, 1024]),
    transforms.Normalize(
        mean=mean,
        std=std
    )
])
# Apply the transforms to the image
img = transform(image)

batch = [img]
prediction = model(batch)[0]

labels = [int_to_label[i] for i in prediction["labels"].tolist()]

# Reshaping the mean and std
mean = torch.tensor(mean).view(-1, 1, 1)
std = torch.tensor(std).view(-1, 1, 1)
img = img * std + mean

# Get the boxes, labels, and scores from the prediction results
boxes = prediction["boxes"]
labels = prediction["labels"]
scores = prediction["scores"]

print("Num detections:", len(labels))
 
# Create a figure and axes
fig, ax = plt.subplots(1)
ax.imshow(to_pil_image(img))

# Iterate over the boxes, labels, and scores
for box, label, score in zip(boxes, labels, scores):
    # Get the coordinates of the bounding box
    xmin, ymin, xmax, ymax = box.detach().numpy()

    # Create a rectangle patch
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the axes
    ax.add_patch(rect)

    label_text = russian_to_italian(int_to_label[int(label.detach().numpy())])

    # Add the label and score as text
    # plt.text(xmin, ymin-10, f"{label_text}: {score:0.2f}", fontsize=10, color='r')
    plt.text(xmin, ymin-2, f"{label_text}", fontsize=14, color='r')

plt.show()
