import cv2
import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


import tensorflow as tf 
SM_FRAMEWORK=tf.keras

import segmentation_models as sm 

tf.keras.backend.set_image_data_format('channels_last')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from segmentation_models.metrics import iou_score
from tensorflow.keras.models import load_model

dice_loss = sm.losses.cce_dice_loss

# loading trained model
model = load_model('model_1.h5', custom_objects={'categorical_crossentropy_plus_dice_loss': dice_loss, 'iou_score': iou_score})

# frame capturing from a video
cam = cv2.VideoCapture('traffic.webm')
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')

currentframe = 0

while(True):
    # reading from frame 
    ret, frame = cam.read()
    if ret:
        name = './data/frame' + str(currentframe) + '.jpg'
        # image writing
        cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()

x_path = []
root_dir = "data"
for root, dirs, files in os.walk(root_dir):
    for name in files:
        if name.endswith((".jpg")):
            x_path.append(os.path.join(root, name).replace('\\','/'))

# https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order
x_path.sort(key=lambda f: int(re.sub('\D', '', f)))

plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

for test_img in tqdm(x_path):
    name = test_img.split('/')[-1]
    image = cv2.imread(test_img, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (256,256), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(name, image)

    pred_mask  = model.predict(image[np.newaxis,:,:,:])
    pred_mask = tf.argmax(pred_mask, axis=-1)

    plt.imshow(pred_mask[0])  
    plt.savefig(name, bbox_inches='tight',pad_inches = 0)


direc = 'Original'

files = os.listdir(direc)
all_files = []

for f in files:
    all_files.append(os.path.join(direc, f))


# https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order
all_files.sort(key=lambda f: int(re.sub('\D', '', f)))

images = []

for f in all_files:
    images.append(cv2.imread(f))

height,width,layers=images[1].shape
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
out = cv2.VideoWriter('video.mp4', fourcc, 10, (height, width), isColor=True)

for f in images:
    out.write(f)
out.release()

direc = 'Masked'

files = os.listdir(direc)
all_files = []

for f in files:
    all_files.append(os.path.join(direc, f))

# https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order
all_files.sort(key=lambda f: int(re.sub('\D', '', f)))

# print(all_files[:10])
images = []

for f in all_files:
    images.append(cv2.imread(f))

height,width,layers=images[1].shape
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
out = cv2.VideoWriter('mask.mp4', fourcc, 10, (height, width), isColor=True)

for f in images:
    out.write(f)
out.release()