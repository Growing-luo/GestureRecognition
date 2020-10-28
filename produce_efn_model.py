import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout
from tensorflow import keras
from tensorflow.keras import optimizers, Sequential
# from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

import efficientnet.tfkeras as efn

# from efficientnet.tfkeras import EfficientNetB0 as Net

# Hyper parameters 超参数设置


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_x = range(len(acc))

    plt.plot(epochs_x, acc, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs_x, loss, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    batch_size = 16

    width = 200
    height = 200
    epochs = 50
    dropout_rate = 0.6
    input_shape = (height, width, 3)
    num_classes =10

    train_dir = r'G:\shuzituxiang\train'
    validation_dir = r'G:\shuzituxiang\val'

    # 图像数据增强
    train_datagen = ImageDataGenerator(

        rescale=1. / 255,  # 归一化
        horizontal_flip=False,
        )

    validation_datagen = ImageDataGenerator(rescale=1. / 255, )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        shuffle=False,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        shuffle=False,
        batch_size=batch_size,
        class_mode='categorical')

    print(train_generator.class_indices)
    print(validation_generator.class_indices)


    base_model = efn.EfficientNetB0(input_shape=(200, 200, 3),
                                    weights=None,
                                    include_top=False,
                                    pooling='avg')
    #base_model.summary()
    model = efn.model.models.Sequential()
    model.add(base_model)
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax', name="fc_out"))

    # 输出网络模型参数
    model.summary()

    # 卷积层参与训练
    base_model.trainable = True

    base_learning_rate = 0.001

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=base_learning_rate),
                  metrics=['acc'])

    # 看看基础模型有多少层
    print("Number of layers in the base model: ", len(base_model.layers))

    # 从此层开始微调
    fine_tune_at = 196

    # 冻结‘fine_tune_at’层之前的所有层
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    history_tl = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )

    from keras.models import load_model

    plot_training(history_tl)

    model.save('my_efficientb0test_model.h5')

#     用于训练后输出Training和Validation的accuracy及loss图

