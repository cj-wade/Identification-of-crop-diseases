from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda
from keras import backend as K


def channel_attention(input_feature, ratio=2, name='cha_att'):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature]), cbam_feature

def mini_channel_attention(input_feature):
    # 得出channel大小
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    # 平均池化特征
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # 最大池化特征
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # 分别1*1卷积获取简单非线性特征，use_bias=True
    avg_attention = Conv2D(filters=1,
                          kernel_size=1,
                          activation='relu',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=True)(avg_pool)
    max_attention = Conv2D(filters=1,
                          kernel_size=1,
                          activation='relu',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=True)(max_pool)
    # 两个注意力加和
    channel_attention = Add()([avg_attention, max_attention])
    # 再对合并的注意力做一次1*1卷积
    channel_attention = Conv2D(filters=1,
                          kernel_size=1,
                          activation='relu',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=True)(channel_attention)
    # 做一次hard_sigmoid回到童年，不要太飘
    channel_attention = Activation('hard_sigmoid')(channel_attention)

    if K.image_data_format() == "channels_first":
        channel_attention = Permute((3, 1, 2))(channel_attention)

    return multiply([input_feature, channel_attention])

def spatial_attention(input_feature, name='spa_att'):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature]), cbam_feature


def cbam(cbam_feature, ratio=2, name='cbam'):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in CBAM: Convolutional Block Attention Module.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def r_cbam(cbam_feature, ratio=2):
    cbam_feature = spatial_attention(cbam_feature, )
    cbam_feature = channel_attention(cbam_feature, ratio)
    return cbam_feature


def i_cbam(input_feature, ratio=2):
    # channel attention(input:input_feature, output:ct)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    ct = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        ct = Permute((3, 1, 2))(ct)

    # spatial attention(input:input_feature, output:st)
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    st = Conv2D(filters=1,
                kernel_size=kernel_size,
                activation='hard_sigmoid',
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                use_bias=False)(concat)
    assert st._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        st = Permute((3, 1, 2))(st)

    cbam_feature = multiply([st, input_feature, ct])
    return cbam_feature

def mini_i_cbam(input_feature):
    # mini channel attention(input:input_feature, output:ct)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    # 平均池化特征
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # 最大池化特征
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # 分别1*1卷积获取简单非线性特征，use_bias=True
    avg_attention = Conv2D(filters=1,
                           kernel_size=1,
                           activation='relu',
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal',
                           use_bias=True)(avg_pool)
    max_attention = Conv2D(filters=1,
                           kernel_size=1,
                           activation='relu',
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal',
                           use_bias=True)(max_pool)
    # 两个注意力加和
    ct = Add()([avg_attention, max_attention])
    # 再对合并的注意力做一次1*1卷积
    ct = Conv2D(filters=1,
                               kernel_size=1,
                               activation='relu',
                               strides=1,
                               padding='same',
                               kernel_initializer='he_normal',
                               use_bias=True)(ct)
    # 做一次hard_sigmoid回到童年，不要太飘
    ct = Activation('hard_sigmoid')(ct)

    if K.image_data_format() == "channels_first":
        ct = Permute((3, 1, 2))(channel_attention)

    # spatial attention(input:input_feature, output:st)
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    st = Conv2D(filters=1,
                kernel_size=kernel_size,
                activation='hard_sigmoid',
                strides=1,
                padding='same',
                kernel_initializer='he_normal',
                use_bias=False)(concat)
    assert st._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        st = Permute((3, 1, 2))(st)

    cbam_feature = multiply([st, input_feature, ct])
    return cbam_feature