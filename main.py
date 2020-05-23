import keras
from utils import *
from keras.callbacks import ModelCheckpoint
from model import *
from DataGenerator import DataGenerator
from saveModel import saveModel
from predict import *
import os
import math
import keras.backend as K
from InceptionResNetV2_Att import InceptionResNetV2_Att
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
CWD = os.getcwd()
BASE = 0
BEST = 0
BATCH_SIZE = 32
EPOCHS = 25 - BASE
VERBOSE = 1
TRAIN_SIZE = 31718
VALID_SIZE = 4540
DATA_DIR = os.path.join(CWD, 'data')
SAVE_DIR = os.path.join(CWD, 'save')
MONITOR='val_accuracy'


WEIGHT_PATH = os.path.join(DATA_DIR,'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
train_dir = os.path.join(DATA_DIR,'train')
train_json = os.path.join(DATA_DIR,'ai_challenger_pdr2018_trainingset_20181023','AgriculturalDisease_trainingset','AgriculturalDisease_train_annotations.json')
valid_dir = os.path.join(DATA_DIR,'valid')
valid_json = os.path.join(DATA_DIR,'ai_challenger_pdr2018_validationset_20181023','AgriculturalDisease_validationset','AgriculturalDisease_validation_annotations.json')


train_data = DataGenerator(train_json,train_dir,batch_size=BATCH_SIZE,is_train=True)
valid_data = DataGenerator(valid_json,valid_dir,batch_size=BATCH_SIZE,is_train=False)


input =  keras.Input(shape=[224, 224, 3])

def test_lenet():
    
    modelpath = os.path.join(SAVE_DIR, "lenet5.hdf5")
    historypath = os.path.join(SAVE_DIR, "lenet5.history.txt")
    
    
    lenet5 = LeNet5(input)
    def scheduler(epoch):
        lr = lenet5.optimizer.lr
        K.set_value(lenet5.optimizer.lr, lr * 0.94)
        return K.get_value(lenet5.optimizer.lr)
    
    reduce_lr = LearningRateScheduler(scheduler)
    savemodel = saveModel(modelpath,2)
    callbacks_list = [reduce_lr,savemodel]

    
    h=lenet5.fit_generator(
                    train_data,
                    steps_per_epoch = math.ceil(TRAIN_SIZE / BATCH_SIZE),
                    verbose=VERBOSE,
                    epochs=EPOCHS,
                    validation_data=valid_data,
                    max_queue_size=128,
                    workers=8,
                    shuffle=True,
                    use_multiprocessing =True,
                    callbacks=callbacks_list)

    saveHistory(h,historypath)


def test_in_att():
    modelpath = os.path.join(SAVE_DIR, "InceptionResNetV2_Att.hdf5")
    historypath = os.path.join(SAVE_DIR, "InceptionResNetV2_Att.history.txt")

    model = InceptionResNetV2_Att(weights_path=WEIGHT_PATH,input_tensor=input)
    def scheduler(epoch):
        lr = model.optimizer.lr
        K.set_value(model.optimizer.lr, lr * 0.98)
        return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)
    savemodel = saveModel(modelpath,5,BASE)
    callbacks_list = [reduce_lr,savemodel]
    print('loading weight')
    
    h=model.fit_generator(
        train_data,
        steps_per_epoch=math.ceil(TRAIN_SIZE / BATCH_SIZE),
        verbose=VERBOSE,
        epochs=EPOCHS,
        validation_data=valid_data,
        max_queue_size=64,
        workers=2,
        shuffle=True,
        use_multiprocessing=True,
        callbacks=callbacks_list)
    
    saveHistory(h,historypath)
    
def test_predict():

    modelpath = os.path.join(SAVE_DIR, "InceptionResNetV2_Att.hdf5")
    savepath = os.path.join(SAVE_DIR, "InceptionResNetV2_Att-result.txt")
    labelspath = 'validlabels.txt'
    print('start predict')
    predict(modelpath,valid_data,savepath)
    valid(labelspath,savepath)
    
def load_ing():
    modelpath =  os.path.join(SAVE_DIR, "epoch:20-InceptionResNetV2_Att.hdf5")
    historypath = os.path.join(SAVE_DIR, "InceptionResNetV2_Att.history.txt")
    print('loading model')
    model = load_model(modelpath)
    model.summary()
    def scheduler(epoch):
        lr = model.optimizer.lr
        K.set_value(model.optimizer.lr, lr * 0.98)
        return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)
    
    save = saveModel(modelpath,5,BASE,BEST)
    callbacks_list = [reduce_lr,save]
    
    print('start training')
    h = model.fit_generator(
        train_data,
        steps_per_epoch=math.ceil(TRAIN_SIZE / BATCH_SIZE),
        verbose=VERBOSE,
        epochs=EPOCHS,
        validation_data=valid_data,
        max_queue_size=64,
        workers=2,
        shuffle=True,
        use_multiprocessing=True,
        callbacks=callbacks_list )
    saveHistory(h,historypath)
    
    
if __name__ == '__main__':

#     test_lenet()
#     test_AlexNet()
    test_in_att()
#     test_in()
#     load_ing()
#     test_predict()
