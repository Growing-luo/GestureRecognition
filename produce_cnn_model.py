# 导入基本依赖包
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, Activation, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout
from tensorflow import keras
from keras import optimizers, Sequential
# from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    epochs = 1
    NUM_TRAIN = 256
    NUM_TEST = 64
    dropout_rate = 0.6
    input_shape = (height, width, 3)
    num_classes = 10
    train_dir = r"G:\test\sunshoushi_train"
    validation_dir = r"G:\test\sunshoushi_val"


    # 图像数据增强
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 归一化
        horizontal_flip=True,
        fill_mode='nearest')

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


    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))  # 这里和下面for循环range里的值要匹配
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


    augemted_images = [train_generator[0][0][0] for i in range(5)]
    plotImages(augemted_images)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(200, 200, 3), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalMaxPooling2D())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()
    print("Number of layers in the base model: ", len(model.layers))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    history_tl = model.fit_generator(
        train_generator,
        #steps_per_epoch=NUM_TRAIN,
        epochs=epochs,
        validation_data=validation_generator,
        #validation_steps=NUM_TEST
    )

    from keras.models import load_model

    plot_training(history_tl)

    model.save('my_model0823.h5')
