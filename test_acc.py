import tensorflow as tf
import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
model=load_model("my_model.h5")


def predict(dir_path):
    if not os.path.exists(dir_path):
        print('路径不存在')
    else:
        dirs = os.listdir(dir_path)
        sum = 0
        m = 0
        for subdir in dirs:
            sum = sum+1
            img_path = dir_path + '/' + subdir
            img = image.load_img(img_path, target_size=(200, 200))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            pred_class = model.predict(img)
            # print(pred_class)
            if pred_class[0][1] > 0.8:
                m = m+1
            max_index = np.argmax(pred_class, axis=-1)
            acc = pred_class[0][int(max_index)] * 100
            rubbish_list = ['B', 'C', 'D', 'J', 'L', 'M', 'O','V','M','Y']
            print("结果为{}".format(rubbish_list[int(max_index)]))
        print(m/sum)

#predict(r"G:\shuzituxiang\test\C")
predict(r"G:\v587l\Desktop\train\3")