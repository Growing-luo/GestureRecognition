import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

def get_rectangle(img, max_idx, contours):
    # rect = cv2.minAreaRect(contours[max_idx])  # 得到最小外接矩形的（中心（x,y)，（宽，高），旋转角度）
    x, y, w, h = cv2.boundingRect(contours[max_idx])
    #print(x)
    #print(y)
    #print(w)
    if w>=h:
        h=w
    else:
        w=h
    #print(h)
    cv2.rectangle(img, (int(x-w*0.1), int(y-h*0.1)), (int(x + w*1.1), int(y + h*1.1)), (0, 255, 0), 2)
    #result = img[x:x+w, y:y+h]
    result = img[int(y - h*0.1+2):int(y + h*1.1-2),int(x-w*0.1+2):int(x + w*1.1-2)]
    #result = cv2.flip(result, 1)  # 第二个参数大于0表示沿y轴翻转

    #print(result)
    return result


def centroid(max_contour):  # 求最大连通域的中心坐标
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def center(contours, frame, max_idx):  #在中心点周围画矩形

    cnt_centroid = centroid(contours[max_idx])
    cv2.circle(contours[max_idx], cnt_centroid, 5, [255, 0, 255], -1)
    cv2.rectangle(frame, (cnt_centroid[0] - 125, cnt_centroid[1] - 175),
                  (cnt_centroid[0] + 125, cnt_centroid[1] + 125), (255, 0, 0), 2)  # 画矩形
    print("Centroid : " + str(cnt_centroid))


def getContours(img):

    contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    #print(area)
    if len(area):
        max_idx = np.argmax(area)
        return max_idx, contours
    else:
        print("没有手势")
        return -1,contours




def HSVBin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #lower_skin = np.array([0, 90, 71])  # ([100, 50, 0])
    #upper_skin = np.array([160, 180, 130])  # ([125, 200, 200])
    lower_skin = np.array([100, 50, 0])
    upper_skin = np.array([125, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # res = cv2.bitwise_and(img,img,mask=mask)
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((6, 6), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel2)
    # closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel3)
    return mask

def HSVBin2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #lower_skin = np.array([0, 90, 71])  # ([100, 50, 0])
    #upper_skin = np.array([160, 180, 130])  # ([125, 200, 200])
    lower_skin = np.array([100, 50, 0])
    upper_skin = np.array([125, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # res = cv2.bitwise_and(img,img,mask=mask)
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((6, 6), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel2)
    # closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel3)
    return mask

