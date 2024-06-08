import argparse
import mimetypes
import pickle

import torch
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

import cv2 as cv
import imutils

from utils import config


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to input image/text file of image paths')
opt = vars(parser.parse_args())

file_type = mimetypes.guess_type(opt['input'])[0]
image_paths = [opt['input']]

if file_type == 'text/plain':
    image_paths = open(opt['input']).read().strip().split('\n')

print('[INFO] Loading object detector...')
model = torch.load(config.MODEL_PATH).to(config.DEVICE)
model.eval()

encoder: LabelEncoder = pickle.loads(open(config.LE_PATH, 'rb').read())

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

for img_path in image_paths:
    img = cv.imread(img_path)
    orginal = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (244, 244))
    img = img.transpose((2, 0, 1))

    img = torch.from_numpy(img)
    img = transforms(img).to(config.DEVICE)
    img = img.unsqueeze(0)

    bbox_prediction, class_prediction = model(img)
    x_start, y_start, x_end, y_end = bbox_prediction[0]

    class_prediction = torch.nn.Softmax(-1)(class_prediction)
    label = class_prediction.argmax(dim=-1).cpu()
    label = encoder.inverse_transform(label)[0]

    orginal = imutils.resize(orginal, 600)
    h, w = orginal.shape[:2]

    x_start, x_end = x_start * w, x_end * w
    y_start, y_end = y_start * h, y_end * h

    y = y_start - 10 if y_start - 10 > 10 else y_start + 10
    cv.putText(orginal, label, (x_start, y), cv.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
    cv.rectangle(orginal, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv.imshow('Output', orginal)
    cv.waitKey(0)
