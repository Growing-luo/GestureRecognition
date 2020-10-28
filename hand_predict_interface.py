import os
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import NW
from PIL import Image, ImageTk
from yuchuli import *
import cv2
import easygui

import tensorflow as tf
import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
model=load_model("my_model.h5")


# 全局变量
sFilePath = 'start'
nFilePath = 'start'
# 定义要创建的目录
mkpath = r"G:\ges_v1\\"
# 将要预测的视频图片保存路径
predict_path = r"G:\ges_v1\predict\\"
# 录制好的视频存储路径
video_path = r"G:\ges_v1\video\\"
# 识别视频图像的地址
dir_path = r"G:\ges_v1\predict"

# 基本窗口设置
window = Tk()
window.title('手势识别')
window.geometry('1800x700+80+50')


# 录制识别视频
def shibie():
    global i, j
    i = 0  # i控制多少帧存储一张图片
    j = 1  # j代表当前已经录了多少张图片
    m=0
    cap = cv2.VideoCapture(0)  # 获取摄像头设备或打开摄像头
    if cap.isOpened():

        while True:
            #roi=cv2.resize(frame,(300,300))
            ret, img = cap.read()
            skinMask = HSVBin(img)
            # cv2.imshow('skin', skinMask)
            res = cv2.GaussianBlur(skinMask, (3, 3), 0)
            dst = cv2.Canny(res, 50, 100)
            # cv2.imshow('dst', dst)
            max_idx, contours = getContours(skinMask)
            # if cv2.waitKey(10)==32:
            if max_idx != -1:
                result = get_rectangle(img, max_idx, contours)
                result = cv2.flip(result, 1)


            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_pre = Image.fromarray(img)
            img_pre = ImageTk.PhotoImage(img_pre)

            result_dir = result
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            result = cv2.resize(result,(300,300))
            result = Image.fromarray(result)
            result = ImageTk.PhotoImage(result)

            c1.create_image(0, 0, anchor=NW, image=img_pre)

            c2.create_image(0, 0, anchor=NW, image=result)



            key = cv2.waitKey(20)

            if i % 5 == 0:
                # 设置保存路径
                #print(result)
                cv2.imwrite(predict_path + str(j) + ".jpg", result_dir)
                j = j + 1
            i = i + 1
            m = m+1
            #key = cv2.waitKey(1) & 0xFF
            if m == 200:
                print("得到" + str(j) + "张图片")
                messagebox.showinfo("提示", "已经完成预测视频的处理")
                break
            window.update_idletasks()
            window.update()
    #cap.release()
    #cv2.destroyAllWindows()  # 销毁所有窗口

# 处理录制好的视频
def shipin():
    global i, j
    i = 0  # i控制多少帧存储一张图片
    j = 1  # j代表当前已经录了多少张图片
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                if i % 5 == 0:
                    # 设置保存路径
                    cv2.imwrite(predict_path + str(j) + ".jpg", frame)
                    print("得到" + str(j) + "张图片")
                    j = j + 1
                i = i + 1
            else:
                break
    else:
        print("打开视频文件失败")


# 创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    os.makedirs(path)
    print(path + ' 创建成功')
    return True

#读取播放avi
def disvideo(num,videocanvas):
    flag=0

    cap = cv2.VideoCapture('avis/'+num+'.avi')
    n=0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            roi=cv2.resize(frame,(500,500))
            img_pre = Image.fromarray(roi)
            img_pre = ImageTk.PhotoImage(img_pre)
            videocanvas.create_image(0, 0, anchor=NW, image=img_pre)
            window.update_idletasks()
            window.update()
            cv2.waitKey(20)
        else:
            break
    cap.release()



def predict():
    list = []
    if not os.path.exists(dir_path):
        print('路径不存在')
    else:
        dirs = os.listdir(dir_path)
        print("开始识别")
        for subdir in dirs:
            img_path = dir_path + '/' + subdir
            # print(img_path)
            img = image.load_img(img_path, target_size=(200, 200))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            pred_class = model.predict(img)
            max_index = np.argmax(pred_class, axis=-1)
            acc = pred_class[0][int(max_index)] * 100
            rubbish_list = ['B', 'C', 'F', 'O', 'R', 'W', 'Y','J','L','M']
            #print("结果为{}的准确率是{:.2f}%".format(rubbish_list[int(max_index)], acc))
            if max_index==0:
                list.append('B')
            if max_index==1:
                list.append('C')
            if max_index==2:
                list.append('F')
            if max_index==3:
                list.append('O')
            if max_index==4:
                list.append('R')
            if max_index==5:
                list.append('W')
            if max_index==6:
                list.append('Y')
            if max_index==7:
                list.append('J')
            if max_index==8:
                list.append('L')
            if max_index==9:
                list.append('M')

        list1 = []
        flag = 1
        count = 1
        index = 0
        for j, i in enumerate(list):
            if j == 0:
                continue
            if i == list[j - 1]:
                count = count + 1
                if len(list) == j + 1 and count >= 20:
                    list1.append(i)
            elif count >= 20:
                count = 1
                list1.append(list[j - 1])
        print(list1)
        num = list1[0]
        disvideo(num,c3)


b2 = Button(window, text="录制识别手势", command=shibie, width=15, height=2).place(x=10, y=160)


b6 = Button(window, text="识别动态手势", command=predict, width=15, height=2).place(x=10, y=260)

b8 = Button(window, text="退出", command=shipin, width=15, height=2).place(x=10, y=360)



c1 = Canvas(window, height=480, width=640, bg='white')
#c1.create_image(200,200,anchor=NW,image=im)
c1.place(x=200, y=35)
c2 = Canvas(window, height=300, width=300, bg='white')
c2.place(x=860, y=35)
c3 = Canvas(window, height=500, width=500, bg='white')
c3.place(x=1200, y=35)


l1 = Label(window, text='原始相机',font=20)
l1.place(x=250, y=5)
l1 = Label(window, text='截取图片',font=20)
l1.place(x=900, y=5)
l1 = Label(window, text='MAYA演示',font=20)
l1.place(x=1350, y=5)
# l1 = Label(window, text='识别后MAYA演示',font=20)
# l1.place(x=1500, y=5)

window.mainloop()

list = [1,2,3]

disvideo(list,c3)