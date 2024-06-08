import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

from utils.bbox_regressor import ObjectDetector
from utils.custom_tensor_dataset import CustomTensorData
from utils import config


print('[INFO] Loading dataset...')
data = []
labels = []
bboxes = []
image_paths = []

for csv_file in paths.list_files(config.ANNOTS_PATH, '.csv'):
    rows = open(csv_file).read().strip().split('\n')
    for row in rows:
        row = row.split(',')
        file_name, x_start, y_start, x_end, y_end, label = row

        img_path = os.path.sep.join([config.IMAGES_PATH, label, file_name])
        img = cv.imread(img_path)
        h, w = img.shape[:2]

        x_start = float(x_start) / w
        x_end = float(x_end) / w

        y_start = float(y_start) / h
        y_end = float(y_end) / h

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (244, 244))

        data.append(img)
        labels.append(label)
        bboxes.append((x_start, y_start, x_end, y_end))
        image_paths.append(img_path)

data = np.array(data, dtype=np.float32)
labels = np.array(labels)
bboxes = np.array(bboxes, dtype=np.float32)
image_paths = np.array(image_paths)

encoder = LabelEncoder()
labels = encoder.fit(labels)

data = train_test_split(data, labels, bboxes, image_paths,
                        test_size=.2, random_state=42)

train_images, test_images = data[:2]
train_labels, test_labels = data[2:4]
train_bboxes, test_bboxes = data[4:6]
train_paths, test_paths = data[6:]

train_images, test_images = torch.tensor(
    train_images), torch.tensor(test_images)
train_labels, test_labels = torch.tensor(
    train_labels), torch.tensor(test_labels)
train_bboxes, test_bboxes = torch.tensor(
    train_bboxes), torch.tensor(test_bboxes)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD)
])


train_dataset = CustomTensorData(
    (train_images, train_labels, train_bboxes),
    transform
)

test_dataset = CustomTensorData(
    (test_images, test_labels, test_bboxes),
    transform
)

print(f'[INFO] Total training samples: {len(train_dataset)}...')
print(f'[INFO] Total test samples: {len(test_dataset)}...')

train_steps = len(train_dataset) // config.BATCH_SIZE
test_steps = len(test_dataset) // config.BATCH_SIZE

train_loader = DataLoader(
    train_dataset,
    config.BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=config.PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    config.BATCH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=config.PIN_MEMORY
)

print('[INFO] Saving testing image paths...')
f = open(config.TEST_PATHS, 'w')
f.write('\n'.join(test_paths))
f.close()

resnet = resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False


object_detector = ObjectDetector(resnet, len(encoder.classes_))
object_detector = object_detector.to(config.DEVICE)

class_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.MSELoss()

optimi = torch.optim.Adam(object_detector.parameters(), config.INIT_LR)
print(object_detector)

history = {
    'total_train_loss': [],
    'total_val_loss': [],
    'train_class_acc': [],
    'val_class_acc': []
}

print('[INFO] Training the network...')
start_time = time.time()
for epoch in tqdm(range(1, config.NUM_EPOCHS + 1)):
    # Set the model in training mode
    object_detector.train()

    total_train_loss = 0
    total_val_loss = 0

    train_correct = 0
    val_correct = 0

    for batch in train_loader:
        images, labels, bboxes = map(lambda x: x.to(config.DEVICE), batch)

        predictions = object_detector(images)
        class_loss = class_loss_fn(predictions[1], labels)
        bbox_loss = bbox_loss_fn(predictions[0], bboxes)
        total_loss = (config.BBOX * bbox_loss) + (config.LABELS * class_loss)

        optimi.zero_grad()
        total_loss.backward()
        optimi.step()

        total_train_loss += total_loss
        train_correct += (predictions[1].argmax(1)
                          == labels).type(torch.float).sum().item()

    with torch.no_grad():
        # Set the model in evaluation mode
        object_detector.eval()

        for batch in test_loader:
            images, labels, bboxes = map(lambda x: x.to(config.DEVICE), batch)

            predictions = object_detector(images)
            class_loss = class_loss_fn(predictions[1], labels)
            bbox_loss = bbox_loss_fn(predictions[0], bboxes)
            total_loss = (config.BBOX * bbox_loss) + \
                (config.LABELS * class_loss)

            optimi.zero_grad()
            total_loss.backward()
            optimi.step()

            total_val_loss += total_loss
            val_correct += (predictions[1].argmax(1)
                            == labels).type(torch.float).sum().item()

    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / test_steps

    train_correct = train_correct / len(train_dataset)
    val_correct = val_correct / len(test_dataset)

    h['total_train_loss'].append(avg_train_loss.cpu().detach().numpy())
    h['total_val_loss'].append(total_val_loss.cpu().detach().numpy())

    h['train_class_acc'].append(train_correct)
    h['val_class_acc'].append(val_correct)

    print(f'[INFO] EPOCH: {epoch}/{config.NUM_EPOCHS}')
    print(
        f'Train loss: {avg_train_loss:.6f}, Train accuracy: {train_correct:.4f}')
    print(f'Val loss: {avg_val_loss:.6f}, Val accuracy: {val_correct:.4f}')

end_time = time.time()
print(
    f'[INFO] Total time taken to train the model: {end_time - start_time:.2f}s')

print('[INFO] Saving the object detector model...')
torch.save(object_detector.state_dict(), config.MODEL_PATH)

print('[INFO] Saving the label encoder...')
f = open(config.LE_PATH, 'wb')
f.write(pk.dumps(encoder))
f.close


plt.style.use('ggplot')
plt.figure()
plt.plot(h['total_train_loss'], label='total_train_loss')
plt.plot(h['total_val_loss'], label='total_val_loss')
plt.plot(h['train_class_acc'], label='train_class_acc')
plt.plot(h['val_class_acc'], label='val_class_acc')
plt.title('Total Training Loss and Classification Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plot_path = os.path.sep.join([config.PLOTS_PATH, 'training_plot.png'])
plt.savefig(plot_path)
