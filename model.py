import keras

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Dense, Flatten,Dropout
from keras.models import Sequential,Model

adam = keras.optimizers.Adam(lr=0.0001)
sgd = keras.optimizers.SGD(lr=0.01)
def LeNet5(input):

    conv1=Conv2D(32, kernel_size=(5, 5), activation='relu',padding='same')(input)
    at1 = cbam(conv1)
    maxpool1=MaxPooling2D(pool_size=(2,2))(at1)

    # dropout1=Dropout(0.2)(maxpool1)

    conv2=Conv2D(64, kernel_size=(5, 5), activation='relu',padding='same')(maxpool1)
    at2 = cbam(conv2)
    maxpool2=MaxPooling2D(pool_size=(2,2))(at2)


    flatten=Flatten()(maxpool2)
    dense1=Dense(500,activation='relu')(flatten)
    output=Dense(61,activation='softmax')(dense1)

    model = Model(input,output,name='lenet5-cbam')
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model