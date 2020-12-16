import os
root_dir = os.path.join(os.path.dirname(__file__), '..')

import numpy as np
import pandas as pd
import PIL.Image as Image
from tqdm import tqdm
import pickle

from src import Logger as Log
from src import Visualize

class PlantPathology:
    def __init__(self, resize=True, use_cache=True):
        self.data_dir = os.path.join(root_dir, 'data')
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.train_csv_filename = os.path.join(self.data_dir, 'train.csv')
        self.test_csv_filename = os.path.join(self.data_dir, 'test.csv')
        self.train_cache_filename = os.path.join(self.data_dir, 'train.pkl')
        self.test_cache_filename = os.path.join(self.data_dir, 'test.pkl')
        self.resize_size = (224, 224)
        self.trainX = None
        self.trainY = None
        self.train_len = None
        self.testX = None
        self.testY = None
        self.test_len = None

        train_image_meta = pd.read_csv(self.train_csv_filename)
        test_image_meta = pd.read_csv(self.test_csv_filename)

        if use_cache and os.path.exists(self.train_cache_filename) and os.path.exists(self.test_cache_filename):
            Log.log(Log.INFO, 'Loading training set from cache...')
            with open(self.train_cache_filename, 'rb') as fin:
                self.trainX, self.trainY, self.mean = pickle.load(fin)
            Log.log(Log.INFO, 'Loading test set from cache...')
            with open(self.test_cache_filename, 'rb') as fin:
                self.testX, self.testY = pickle.load(fin)
        else:
            Log.log(Log.INFO, 'Loading training set...')
            trainX = []
            trainY = []
            for record in tqdm(train_image_meta.values):
                filename, Ys = record[0], record[1:]
                img_filename = os.path.join(self.img_dir, '{}.jpg'.format(filename))
                img = Image.open(img_filename)
                if resize:
                    img = np.array(img.resize(self.resize_size))
                else:
                    img = np.array(img)
                trainX.append(img)
                trainY.append(Ys)
            self.trainX = np.array(trainX, dtype=np.uint8)
            self.trainY = np.array(trainY, dtype=np.float16)

            Log.log(Log.INFO, 'Loading test set...')
            testX = []
            testY = []
            for record in tqdm(test_image_meta.values):
                filename, Ys = record[0], record[1:]
                img_filename = os.path.join(self.img_dir, '{}.jpg'.format(filename))
                img = Image.open(img_filename)
                if resize:
                    img = np.array(img.resize(self.resize_size))
                else:
                    img = np.array(img)
                testX.append(img)
                testY.append(Ys)
            self.testX = np.array(testX)
            self.testY = np.array(testY)

            Log.log(Log.INFO, 'Calculating mean value...')
            self.mean = np.sum(self.trainX, axis=(0, 1, 2)) / np.prod(self.trainX.shape[:-1]) / 255

            Log.log(Log.INFO, 'Dumping data...')
            with open(self.train_cache_filename, 'wb') as fout:
                pickle.dump([self.trainX, self.trainY, self.mean], fout)
            with open(self.test_cache_filename, 'wb') as fout:
                pickle.dump([self.testX, self.testY], fout)

        Log.log(Log.INFO, 'Applying normalization...')
        self.trainX = np.array(self.trainX / 255 - self.mean, dtype=np.float32)
        self.testX = np.array(self.testX / 255 - self.mean, dtype=np.float32)

        self.train_len = self.trainX.shape[0]
        self.test_len = self.testX.shape[0]

        Log.log(Log.INFO, 'Loading datasets success.')

    def trainLen(self):
        return self.train_len

    def testLen(self):
        return self.test_len

    def trainData(self, index=None):
        if index is None:
            return self.trainX, self.trainY
        assert index >= 0 and index < self.trainLen()
        return self.trainX[index], self.trainY[index]

    def testData(self, index=None):
        if index is None:
            return self.testX, self.testY
        assert index >= 0 and index < self.testLen()
        return self.testX[index], self.testY[index]
