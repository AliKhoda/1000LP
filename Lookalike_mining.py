# Part1: Arcface feature extraction

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import sys
import numpy as np
import mxnet as mx
import os

from scipy import misc
import random
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans
import matplotlib.pyplot as plt
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

import progressbar
import ipdb
from multiprocessing.dummy import Pool as ThreadPool

# Load model
def get_model(ctx, model):
    image_size = (112,112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model
  
# Preprocess input image
def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    # Assert input shape
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2
        assert image_size[0]==112
        assert image_size[0]==112 or image_size[1]==96
    
    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped
    
    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
      
# Face alignment
def get_input(detector,face_img):
    # Pass input images through face detector
    ret = detector.detect_face(face_img, det_type = 0)
    if ret is None:
        return None
    bbox, points = ret
    if bbox.shape[0]==0:
        return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    # Call preprocess() to generate aligned images
    nimg = preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned
  
# Get embedding
def get_feature(model,aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding
  
# Get feat for image file
def get_feat_img(img):
    global detector
    global model
    
    image = cv2.imread(img)
    # Preprocess image
    pre = get_input(detector,image)
    # Get embedding of image
    return get_feature(model,pre)

# Calculate feat pair similarity
def calc_dist_sim(feat1, feat2):
    # Compute squared distance between embeddings
    dist = np.sum(np.square(feat1-feat2))
    # Compute cosine similarity between embedddings
    sim = np.dot(feat1, feat2.T)
    return (dist, sim)
  
# Dump feat to text file
def dump_feat(args):
    img = args[0]
    featfile = args[1]
    global detector
    global model
    
    try:
        feat = get_feat_img(img)
        if not os.path.exists(os.path.split(featfile)[0]):
            os.makedirs(os.path.split(featfile)[0])
        np.savetxt(featfile, feat, fmt='%f')
    except:
        pass
      
# Load feature file
def load_feat(featfile):
    return np.loadtxt(featfile, dtype=float)

# Dump feature for all '.jpg' files in a directory
def dump_feat_dir(imgdir, featdir):
    
    global detector
    global model
    
    tlist = []
    
    for root, dirs, files in os.walk(os.path.join(imgdir)):
        pb = progbar(root)
        for file in pb(files):
            if file.endswith(".jpg"):
                jpg = os.path.join(root,file)
                txt = jpg.replace(imgdir,featdir).replace('.jpg','.txt')

                if not os.path.exists(txt):
                    dump_feat((jpg,txt))
  
# Download onnx model
mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx')
# Path to ONNX model
model_name = 'resnet100.onnx'

# Configure face detector
det_threshold = [0.6,0.7,0.8]
mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)

# Load ONNX model
model = get_model(ctx , model_name)

base = 'dataloc' # Location of the image folder
images = os.path.join(base,'image')
feats = os.path.join(base,'feat')

# Dump features for all files in VGGFace
dump_feat_dir(images, feats)











# Part2: Lookalike pair extraction

import os
import progressbar
import numpy as np
import pandas as pd
import collections
import cv2
import glob
import multiprocessing

from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# Check if VGGID is in Voxceleb
def isinVC(vggid):
    global voxtab
    
    return not voxtab[voxtab['VGGFace2 ID '].str.contains(vggid)].empty

# Metadata for VGGface2 and Voxceleb2 datasets
metadata = os.path.join(base,'identity_meta.csv')
metavox = os.path.join(base,'vox2_meta.csv')

voxtab = pd.read_csv(metavox)
table = pd.read_csv(metadata)

# Extract median embedding per ID in VGGface
feat = {}
count = 0
for root, dirs, files in os.walk(os.path.join(featdir)):
    if files:
        count += 1
        # Extract ID from path
        faceid = os.path.split(root)[1]
        
        # Load features
        matrix = []
        pb = progbar('{}: {}'.format(count,faceid))
        for file in pb(files):
            if file.endswith(".txt"):
                matrix.append(np.loadtxt(os.path.join(root,file)))
        
        # Calculate median and fill dictionary
        vals = np.stack(matrix, axis = 0)
        feat[faceid] = np.median(vals, axis = 0)

ofeat = collections.OrderedDict(sorted(feat.items()))

names = list(ofeat.keys())
vals = np.stack(list(ofeat.values()), axis = 0)

# Pairwise distance matrix between identities
mat = cdist(vals,vals,'cosine')
mat = (mat + mat.transpose())/2
np.fill_diagonal(mat,2)

# Output file
outfile = os.path.join(base, 'lookalikes.csv')

# Sort scores
sortedvals = np.sort(mat, axis=None)

df = pd.DataFrame(columns=['TargetID', 'TargetName', 
                           'LookalikeID', 'LookalikeName'])

# How many pairs to find
tofind = 10*len(mat)

found = 0
pointer = 0
pb = progbar("Finding Lookalikes",tofind)
pb.start()
while found<tofind:
    # Find location of nth highest similarity in mat
    ids = np.where(mat == sortedvals[pointer*2])
    pair = ids[0]
    
    # Extract name and gender
    g1 = list(table[table['Class_ID'].str.contains(names[pair[0]])][' Gender'])[0]
    n1 = list(table[table['Class_ID'].str.contains(names[pair[0]])][' Name'])[0]
    g2 = list(table[table['Class_ID'].str.contains(names[pair[1]])][' Gender'])[0]
    n2 = list(table[table['Class_ID'].str.contains(names[pair[1]])][' Name'])[0]
    
    pointer += 1
    
    # if genders match and both identities are in VoxCeleb2 dataset
    if (g1 is g2) and isinVC(names[pair[0]]) and isinVC(names[pair[1]]):
        
        # insert into pairs dataframe
        df.loc[found] = [names[pair[0]], n1, 
                         names[pair[1]], n2]
        
        found += 1
        
        pb.update(found)
        
pb.finish()

# save outputfile
df.to_csv(outfile)
