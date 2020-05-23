import keras
import time
import os
class saveModel(keras.callbacks.Callback):
    def __init__(self,filepath,epoch=None,base=0,best=0):
        self.best = best
        self.filepath = filepath
        self.epoch = epoch
        self.base = base

    def sleep(self,times):
        for i in range(times):
            print('休眠还剩{}秒'.format(times-i))
            time.sleep(1)

    def save(self,filepath):
        print('saving to:{}'.format(filepath))
        try:
            self.model.save(filepath, overwrite=True)
        except Exception as e:
            print(e)
            self.sleep(15)
            self.model.save(filepath, overwrite=True)
            pass

    def on_epoch_end(self,epoch,logs={}):
        
        now_acc = logs.get('val_accuracy')
        t = epoch + 1 + self.base
        if now_acc > self.best:
            self.best = now_acc
            self.save(self.filepath)

        if self.epoch and t > 1 and t % self.epoch == 0:
            (dir,name) = os.path.split(self.filepath)
            name = 'epoch:{:0>2d}-'.format(t) + name
            path = os.path.join(dir,name)
            self.save(path)

