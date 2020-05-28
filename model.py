import keras

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPool1D,GlobalMaxPool2D
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

def AlexNet(input):
    # Define the converlutional layer 1
    conv1 = Conv2D(filters=96, kernel_size=[11, 11], strides=[4, 4], activation='relu',
                   use_bias=True, padding='valid')(input)
    pooling1 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=[5, 5], strides=[1, 1], activation='relu',
                   use_bias=True, padding='valid')(pooling1)
    pooling2 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv2)

    conv3 = Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], activation='relu',
                   use_bias=True, padding='valid')(pooling2)
    conv4 = Conv2D(filters=192, kernel_size=[3, 3], strides=[1, 1], activation='relu',
                   use_bias=True, padding='valid')(conv3)

    # pooling3 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv4)
#     pooling3 = MaxPooling2D(pool_size=[4, 4], strides=[1, 1], padding='valid')(conv4)
    conv5 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], activation='relu',
                   use_bias=True, padding='valid')(conv4)

    pooling3 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv5)

    flatten = Flatten()(pooling3)
    fc1 = Dense(2048, activation='relu', use_bias=True)(flatten)
    drop1 = Dropout(0.5)(fc1)
    fc2 = Dense(2048, activation='relu', use_bias=True)(drop1)
    drop2 = Dropout(0.5)(fc2)
    output = Dense(61, activation='softmax', use_bias=True)(drop2)
    
#     pooling4 = GlobalMaxPool2D()(conv5)
#     output = Dense(61, activation='softmax', use_bias=True)(pooling4)
    model = keras.Model(input, output,name='alexnet')
    # 编译模型
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

