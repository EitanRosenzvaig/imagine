import os
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import numpy as np
import scipy.sparse as sp
import logging

logger = logging.getLogger('similarity')

class ImageEncoder:

    def __init__(self, files):
        self.idx_to_mid = {}
        self.files = files
        self.total_max = len(self.files)
        self.batch_size = 10
        self.min_idx = 0
        self.max_idx = self.min_idx + min(self.batch_size, self.total_max)
        self.n_dims = 25088
        self.px = 224
        self.preds = sp.lil_matrix((self.total_max, self.n_dims))
        self.model = VGG16(include_top=False, weights='imagenet')

    def get_encodings(self):
        logger.info("Starting encoding of %s images", self.total_max)
        while self.min_idx < self.total_max - 1:
            X = np.zeros(((self.max_idx - self.min_idx), self.px, self.px, 3))
            
            # For each file in batch, 
            # load as row into X
            for i in range(self.min_idx, self.max_idx):
                fname = self.files[i]
                mid = os.path.basename(fname)
                self.idx_to_mid[i] = mid
                try:
                    img = kimage.load_img(fname, target_size=(self.px, self.px))
                    img_array = kimage.img_to_array(img)
                    X[i - self.min_idx, :, :, :] = img_array
                except:
                    logger.error('Error loading %s', fname)
            self.max_idx = i
            X = preprocess_input(X)
            
            these_preds = self.model.predict(X)
            shp = ((self.max_idx - self.min_idx) + 1, self.n_dims)
            
            # Place predictions inside full preds matrix.
            self.preds[self.min_idx:self.max_idx + 1, :] = these_preds.reshape(shp)
            
            self.min_idx = self.max_idx
            self.max_idx = np.min((self.max_idx + self.batch_size, self.total_max))
        return self.preds.tocsr(), self.idx_to_mid