import argparse
import matplotlib.pyplot as plot

import chainer

from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='trained_model/snapshot_model.npz')
    parser.add_argument('image')
    args = parser.parse_args()

    model = FasterRCNNVGG16(
        n_fg_class=1,
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=('face',))
    plot.show()


if __name__ == '__main__':
    main()
