# coding: utf-8

#model url: http://nixeneko.2-d.jp/hatenablog/20170724_facedetection_model/snapshot_model.npz

import urllib.request
import os

def download_model(url, dest):
    destdir = os.path.dirname(dest)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    print("Downloading {}... \nThis may take several minutes.".format(dest))
    urllib.request.urlretrieve(url, dest)
