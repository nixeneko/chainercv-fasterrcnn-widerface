from __future__ import division

try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    pass

import argparse
import numpy as np
import warnings 
import logging

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger

from chainercv.datasets import TransformDataset
#from chainercv.datasets import voc_detection_label_names
#from chainercv.datasets import VOCDetectionDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainercv import transforms

from wider_face_dataset import WIDERFACEDataset

# dataset paths
WIDER_TRAIN_DIR = 'WIDER_train'
WIDER_TRAIN_ANNOTATION_MAT = 'wider_face_split/wider_face_train.mat'
WIDER_VAL_DIR = 'WIDER_val'
WIDER_VAL_ANNOTATION_MAT = 'wider_face_split/wider_face_val.mat'

BLACKLIST_FILE = 'blacklist.txt'

faster_rcnn = FasterRCNNVGG16(n_fg_class=1,
                              pretrained_model='imagenet')
                              #min_size=600, max_size=1000,)
def transform(in_data):
    img, bbox, label = in_data
    _, H, W = img.shape
    img = faster_rcnn.prepare(img)
    _, o_H, o_W = img.shape
    scale = o_H / H
    bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

    # horizontally flip
    img, params = transforms.random_flip(
        img, x_random=True, return_param=True)
    bbox = transforms.flip_bbox(
        bbox, (o_H, o_W), x_flip=params['x_flip'])

    return img, bbox, label, scale

def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: Faster R-CNN')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--train_data_dir', '-t', default=WIDER_TRAIN_DIR,
                        help='Training dataset (WIDER_train)')
    parser.add_argument('--train_annotation', '-ta', default=WIDER_TRAIN_ANNOTATION_MAT,
                        help='Annotation file (.mat) for training dataset')
    parser.add_argument('--val_data_dir', '-v', default=WIDER_VAL_DIR,
                        help='Validation dataset (WIDER_train)')
    parser.add_argument('--val_annotation', '-va', default=WIDER_VAL_ANNOTATION_MAT,
                        help='Annotation file (.mat) for validation dataset')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # for logging pocessed files
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename='filelog.log')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    blacklist = []
    with open(BLACKLIST_FILE, 'r') as f:
        for line in f:
            l = line.strip()
            if l:
                blacklist.append(line.strip())
    
    # train_data = VOCDetectionDataset(split='trainval', year='2007')
    # test_data = VOCDetectionDataset(split='test', year='2007',
                                    # use_difficult=True, return_difficult=True)
    train_data = WIDERFACEDataset(args.train_data_dir, args.train_annotation, 
        logger=logger, exclude_file_list=blacklist)
    test_data = WIDERFACEDataset(args.val_data_dir, args.val_annotation)
    # faster_rcnn = FasterRCNNVGG16(n_fg_class=len(voc_detection_label_names),
                                  # pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))


    train_data = TransformDataset(train_data, transform)
    #import pdb; pdb.set_trace()
    #train_iter = chainer.iterators.MultiprocessIterator(
    #    train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=1)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn, 'snapshot_model.npz'),
        trigger=(args.iteration, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'iteration'))

    log_interval = 20, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/map',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model.faster_rcnn, use_07_metric=True,
            label_names=('face',)),
        trigger=ManualScheduleTrigger(
            [args.step_size, args.iteration], 'iteration'))

    trainer.extend(extensions.dump_graph('main/loss'))

    #try:
        # warnings.filterwarnings('error', category=RuntimeWarning)
    trainer.run()
    #except RuntimeWarning as w:
    #    logger.debug(w)
    #    quit()

if __name__ == '__main__':
    main()
