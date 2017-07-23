# Face detection based on ChainerCV Faster-RCNN implementation

Modified ChainerCV example to train on WIDER FACE dataset for face detection instead on PASCAL VOC.

The original ChainerCV example code is: 
https://github.com/chainer/chainercv/tree/master/examples/faster_rcnn

## Usage
Tested on Windows 10 (64 bit), Python 3.5.3 (installed by Anaconda)

This code depends on:
- Chainer (https://chainer.org/)
- CuPy (https://cupy.chainer.org/)
- ChainerCV (https://github.com/chainer/chainercv)
- SciPy (https://www.scipy.org/; for importing MATLAB .mat file)

Library version: 
- Chainer 2.0.0
- CuPy 1.0.0.1
- ChainerCV 0.5.1

### Demo
If a path to pretorained model is not given, `tained_moded/snapshot_model.npz` is used.

    python demo.py [--gpu <gpu>] [--pretrained_model <model_path>] <image>.jpg

### Train code
First, download the WIDER FACE dataset from the website: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/ 

Download these files and extract:
- Wider Face Training Images
- Wider Face Validation Images
- Face annotations

Execute with specifing dataset directories and annotation files (.mat) for training and validation sets, such as:

    python train.py --gpu=0 --train_data_dir="WIDER_train" --train_annotation="wider_face_split/wider_face_train.mat" --val_data_dir="WIDER_val" --val_annotation="wider_face_split/wider_face_val.mat"

Or edit codes specifying dataset paths in `train.py` and execute:

    python train.py --gpu 0

### test_dataset.py
Show ground truth and predicted result on the images from validation dataset.
Requires OpenCV Python bindings to execute.

Edit `test_detection.py` about dataset paths and run:

    python test_detection.py --gpu 0

- Green rectangles: ground truth.
- Blue rectangles: predicted result (the less score has the lighter color)

Esc key to exit. Any other keys to the next image.

### eval.py
Returns average precision. (0.362059 for `trained_model/snapshot_model.npz`)

    python eval.py [--gpu <gpu>] [--pretrained_model <model_path>]


