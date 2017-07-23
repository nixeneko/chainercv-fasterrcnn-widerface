from __future__ import division

import argparse
import numpy as np
import sys
import time

import chainer
from chainer import iterators

from chainercv.datasets import voc_detection_label_names
from chainercv.evaluations import eval_detection_voc_ap
from chainercv.links import FasterRCNNVGG16
from chainercv.utils import apply_prediction_to_iterator

from  wider_face_dataset import WIDERFACEDataset

class ProgressHook(object):

    def __init__(self, n_total):
        self.n_total = n_total
        self.start = time.time()
        self.n_processed = 0

    def __call__(self, imgs, pred_values, gt_values):
        self.n_processed += len(imgs)
        fps = self.n_processed / (time.time() - self.start)
        sys.stdout.write(
            '\r{:d} of {:d} images, {:.2f} FPS'.format(
                self.n_processed, self.n_total, fps))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='result/snapshot_model.npz')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    model = FasterRCNNVGG16(
        n_fg_class=1,
        pretrained_model=args.pretrained_model)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    #dataset = VOCDetectionDataset(
    #    year='2007', split='test', use_difficult=True, return_difficult=True)
    dataset = WIDERFACEDataset('WIDER_val', 'wider_face_split/wider_face_val.mat',
        use_difficult=True, return_difficult=True)
    iterator = iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterator explicitly
    del imgs

    pred_bboxes, pred_labels, pred_scores = pred_values
    gt_bboxes, gt_labels, gt_difficults = gt_values

    ap = eval_detection_voc_ap(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    map_ = np.nanmean(ap)

    print()
    print('mAP: {:f}'.format(map_))
    for l, name in enumerate(('face',)):
        if ap[l]:
            print('{:s}: {:f}'.format(name, ap[l]))
        else:
            print('{:s}: -'.format(name))


if __name__ == '__main__':
    main()
