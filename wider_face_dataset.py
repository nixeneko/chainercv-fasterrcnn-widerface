# coding: utf-8

import numpy as np
import os
import warnings
import logging
#import xml.etree.ElementTree as ET
import scipy.io

import chainer

from chainercv.utils import read_image


class WIDERFACEDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir, label_mat_file,
                 use_difficult=False, return_difficult=False, 
                 exclude_file_list=None, logger=None):
        
        # id_list_file = os.path.join(
            # data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        # self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.label_mat_file = label_mat_file
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        
        self.logger = logger #for 
        # list up files
        mat = scipy.io.loadmat(self.label_mat_file)
        self.ids = []
        self.bboxs = {}
        self.labels = {}
        self.difficult = {}
        for i in range(len(mat['event_list'])):
            event = mat['event_list'][i,0][0]
            for j in range(len(mat['file_list'][i,0])):
                file = mat['file_list'][i,0][j,0][0]
                filename = "{}.jpg".format(file)
                filepath = os.path.join(data_dir, 'images', event, filename)
                if exclude_file_list != None and filename in exclude_file_list:
                    continue
                # bounding boxes and labels of the picture file
                bboxs = mat['face_bbx_list'][i,0][j,0]
                # convert from (x, y, w, h) to (y1, x1, y2, x2)
                swapped_bbxs = bboxs[:, [1,0,3,2]] #  (y,x,h,w)
                swapped_bbxs[:,2:4] = swapped_bbxs[:,2:4] + swapped_bbxs[:,0:2]
                
                invalid_labels = mat['invalid_label_list'][i,0][j,0].ravel()
                pose_labels = mat['pose_label_list'][i,0][j,0].ravel()
                illum_labels = mat['illumination_label_list'][i,0][j,0].ravel()
                occlusion_labels = mat['occlusion_label_list'][i,0][j,0].ravel()
                blur_labels = mat['blur_label_list'][i,0][j,0].ravel()
                expression_labels = mat['expression_label_list'][i,0][j,0].ravel()
                
                self.ids.append(filepath)
                self.bboxs[filepath] = swapped_bbxs.astype(np.float32)
                self.labels[filepath] = np.zeros(len(bboxs), dtype=np.int32) #dummy, always 0
                self.difficult[filepath] = invalid_labels

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        bbox = self.bboxs[id_].astype(np.float32)
        label = self.labels[id_].astype(np.int32)
        difficult = self.difficult[id_].astype(np.bool)
        if not self.use_difficult:
            bbox = bbox[np.where(difficult==False)]
            label = label[np.where(difficult==False)]
            difficult = difficult[np.where(difficult==False)]

        # Load a image
        img_file = id_
        img = read_image(img_file, color=True)
        #print(img_file)
        if self.logger:
            self.logger.debug(img_file)
        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label
        
def test():
    a = WIDERFACEDataset('WIDER_train', 'wider_face_split/wider_face_train.mat')
    id_ = r'WIDER_train\images\36--Football\36_Football_americanfootball_ball_36_571.jpg'
    bbx = a.bboxs[id_]
    lbl = a.labels[id_]
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    test()