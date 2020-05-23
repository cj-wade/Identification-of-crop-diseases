# coding=utf-8
import keras
import math
import os
import cv2
import numpy as np
import json
import random
import tensorflow as tf
from keras.preprocessing import image


class DataGenerator(keras.utils.Sequence):

    def __init__(self, json_path,data_dir,is_train=True,target_size=(224,224,3),batch_size=32,):
        self.batch_size = batch_size
        self.data =  json.load(open(json_path,'r'))
        self.indexes = np.arange(len(self.data))
        self.data_dir = data_dir
        self.target_size = target_size
        self.is_train = is_train
        self.on_epoch_end()


    def __len__(self):
        # 计算每一个epoch的迭代次数 向上取整

        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        # 每次数据的范围
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.data[k] for k in batch_indexs]
        # 生成数据
        x, y = self.data_generation(batch_datas)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.is_train:
#             print('change')
            np.random.shuffle(self.indexes)


    def data_generation(self, data):
        images = np.empty((len(data),self.target_size[0],self.target_size[1],self.target_size[2]))
        labels = []
        # 读取图片 并且设置为标签
        for i in range(len(data)):
            _class = data[i]['disease_class']
            _id = data[i]['image_id']
            _class_dir = os.path.join(self.data_dir, str(_class))
            image_path = os.path.join(_class_dir, _id)
            images[i,] = self.read_img(image_path)
            labels.append(_class)

        labels = keras.utils.to_categorical(labels,61)
        return images,labels
#         return images,labels

    def read_img(self,path):
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            img = self.preprocess(img)
        except Exception as e:
            print(e)
        else:
            return img

    def preprocess(self,img):
        # >0 水平翻转
        # =0 垂直翻转
        # <0 水平垂直翻转

        if self.is_train:
            # 随机224*224 缩放
            t = np.random.randint(4)
            if t == 0:
                img = cv2.resize(img, (self.target_size[0], self.target_size[1]), cv2.INTER_NEAREST)
            elif t == 1:
                img = cv2.resize(img, (self.target_size[0], self.target_size[1]), cv2.INTER_LINEAR)
            elif t == 2:
                img = cv2.resize(img, (self.target_size[0], self.target_size[1]), cv2.INTER_AREA)
            else:
                img = cv2.resize(img,(self.target_size[0],self.target_size[1]),cv2.INTER_CUBIC)

            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
        else:
            img = cv2.resize(img,(self.target_size[0],self.target_size[1]))
        img = img / 255
        return img


# import time
# start = time.time()
# images = []
# for i in range(len(valid)):
#     _class = valid[i]['disease_class']
#     _id = valid[i]['image_id']
#     _class_dir = os.path.join(valid_dir, str(_class))
#     image_path = os.path.join(_class_dir, _id)
#     if not  read_img(image_path):
#         images.append(image_path)
# end = time.time()
#
#
# def read_img(path):
#     try:
#         img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
#         img = cv2.resize(img,(224,224,3))
#         img = img / 255
#         preprocess(img)
#     #             if self.is_train:
#     #                 img = self.preprocess(img)
#     except Exception as e:
#         print(e)
#     else:
#         return img
# #
# def preprocess(img):
#     # >0 水平翻转
#     # =0 垂直翻转
#     # <0 水平垂直翻转
#     if random.random()>0.5:
#         cv2.flip(img,1)
#     if random.random()>0.5:
#         cv2.flip(img,0)