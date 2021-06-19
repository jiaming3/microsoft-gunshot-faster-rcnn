# this file is used to detect gunshot from any wav files(time need to be greater than 12s).
# the filename will be save as current second + xmin + ymin + xmax + ymax,
# such name will help user to obtain the false positive and retrain the model
# only detected gunshot image will be saved.
from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure
import json
from PIL import Image
import pandas as pd
from pydub import AudioSegment

sys.setrecursionlimit(40000)
guncheck = False
config_output_filename = "dataset/config.pickle"
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
C.num_rois = 10
C.model_path = "dataset/retrain.hdf5"
#img_path = "hn/hard negatives"  # options.test_path


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier_only.load_weights(C.model_path, by_name=True)

all_imgs = []

classes = {}

bbox_threshold = 0.80

##########################################################################################################################

sound_path = 'TRY THESE GUNS'
mylist = os.listdir(sound_path)
num_detected = 0
overlap = 12 * 1000
twelve = 12 * 1000
fig = figure.Figure()
ax = fig.subplots(1)
hn_count = 0
gunshot_cout = 0
shotgun_cout = 0
data = {
    "label": [],
    "confidence": [],
    "filename": [],
    "offset (s)": [],

}

##########################################################################################################################

for i in range(len(mylist)):

    print(i, ':', mylist[i])
    print("Loading soundfile")

    start = time.time()
    clip = AudioSegment.from_wav(sound_path + "/" + mylist[i])
    end = time.time()

    print("Took", int(end - start), "seconds to load file")

    current_s = 0

    for j in range(int(clip.duration_seconds / (overlap / 1000))):

        hit = False

        st = time.time()

        fig = figure.Figure()
        ax = fig.subplots(1)

        if ((current_s + twelve) / 1000 > clip.duration_seconds):
            print("Breaking, clip not in range")
            break

        short = clip[current_s:current_s + twelve]

        short.export("cache/short.wav", format="wav")

        y, sr = librosa.load("cache/short.wav")

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=1500)

        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr,
                                       fmax=1500, ax=ax)

        save_to = 'cache/images/' + "detected_" + mylist[i] + "_" + str((current_s / 1000)) + '.png'

        fig.savefig(save_to)
        plt.cla()
        plt.close(fig)

        os.remove("cache/short.wav")

        ##########################################################################################################################
        img = cv2.imread(save_to)
        X, ratio = format_img(img, C)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        print("R", R.shape)
        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            # ROI(1, 10, 4)
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
            target_shape1 = (ROIs.shape[0], C.num_rois, ROIs.shape[2])
            ROIs_padded1 = np.zeros(target_shape1).astype(ROIs.dtype)
            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            hit = True
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):

                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              2)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))

                all_dets.append((key, 100 * new_probs[jk]))

                data['filename'].append(mylist[i])
                data['offset (s)'].append(str(current_s / 1000))
                data['label'].append(str(key))
                data['confidence'].append(str(100 * new_probs[jk]))
                if (str(key) == 'gunshot' or str(key) == 'shotgun'):
                    guncheck = 'True'
                    if (str(key) == 'gunshot'):
                        gunshot_cout += 1
                        guncheck == True

                    else:
                        shotgun_cout += 1
                        guncheck == True

                    (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    textOrg = (real_x1, real_y1 - 0)

                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                                  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                    cv2.imwrite('dataset/detected/{}.png'.format(
                        str(current_s) + ' ' + str(real_x1) + ' ' + str(real_y1) + ' ' + str(real_x2) + ' ' + str(
                            real_y2)), img)
                if (str(key) == 'hn'):
                    hn_count += 1
        print('Elapsed time = {}'.format(time.time() - st))
        print('result at ', current_s, ':', all_dets)
        print('hn:', hn_count, 'gunshot:', gunshot_cout, 'shotgun', shotgun_cout, 'guncheck:', guncheck)

        if (guncheck):
            print('checking', guncheck)
            guncheck = False
            print('reset', guncheck)
        else:
            os.remove(save_to)

        current_s = current_s + twelve

    print("Took ", str((time.time() - start) / 60), " mins, for ", str(clip.duration_seconds / 60), "mins of audio")

with open("detected.json", "w") as store:
    json.dump(data, store, indent=3)

df = pd.read_json("detected.json")
df.to_csv("detected_csv.csv")
