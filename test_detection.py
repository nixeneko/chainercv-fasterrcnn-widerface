# coding: utf-8

import os
import numpy as np
import scipy.io
import cv2

MATFILE = "wider_face_split/wider_face_train.mat"
IMGDIR = "WIDER_train/images"

mat = scipy.io.loadmat(MATFILE)

#dict_keys(['pose_label_list', 'event_list', 'file_list', '__header__', '__version__', 'invalid_label_list', 'illumination_label_list', '__globals__', 'occlusion_label_list', 'face_bbx_list', 'blur_label_list', 'expression_label_list'])

MODELFILE = 'result/snapshot_model.npz'

import chainer
from chainer import iterators
from chainercv.links import FasterRCNNVGG16
from chainercv import utils

model = FasterRCNNVGG16(
    n_fg_class=1,
    pretrained_model=MODELFILE)

chainer.cuda.get_device(0).use()
model.to_gpu()

for i in range(len(mat['event_list'])):
    event = mat['event_list'][i,0][0]
    for j in range(len(mat['file_list'][i,0])):
        file = mat['file_list'][i,0][j,0][0]
        filename = "{}.jpg".format(file)
        filepath = os.path.join(IMGDIR, event, filename)
        # bounding boxes and labels of the picture file
        bboxs = mat['face_bbx_list'][i,0][j,0]
        invalid_labels = mat['invalid_label_list'][i,0][j,0].ravel()
        pose_labels = mat['pose_label_list'][i,0][j,0].ravel()
        illum_labels = mat['illumination_label_list'][i,0][j,0].ravel()
        occlusion_labels = mat['occlusion_label_list'][i,0][j,0].ravel()
        blur_labels = mat['blur_label_list'][i,0][j,0].ravel()
        expression_labels = mat['expression_label_list'][i,0][j,0].ravel()
        
        #if not '36_Football_Football_36_63' in filename:
        #    continue
            
        img = cv2.imread(filepath)
        a, = np.where(invalid_labels==1)
        #print(invalid_labels)
        #print(np.where(invalid_labels==1))
        #if a.sum() > 0:
            #print(bboxs)
        for k, bbox in enumerate(bboxs):
            #color = (0,0,255) if invalid_labels[k] else (0,255,0)
            color = (255,255,0)
            pt1 = tuple(bbox[:2])
            pt2 = tuple(np.add(bbox[:2], bbox[2:]))
            cv2.rectangle(img, pt1, pt2, color, 1)
            
        imgpred = utils.read_image(filepath, color=True)
        bboxes, labels, scores = model.predict([imgpred])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        print(score)
        for k in np.where(score>=0.7)[0]:
            #color = (0,0,255) if invalid_labels[k] else (0,255,0)
            bbx = bbox[k]
            color = (0,int(255*(1-score[k])),255)
            pt1 = tuple(bbx[1::-1])
            pt2 = tuple(bbx[:1:-1])
            cv2.rectangle(img, pt1, pt2, color, 1)
        
        cv2.imshow('test', img)
        print(img.shape)
        key = cv2.waitKey()
        if key == 27:
            quit()
        print('next_loop')

#import pdb; pdb.set_trace()

