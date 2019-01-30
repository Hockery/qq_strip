#!/bin/bash

import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('images/screen2.png')
image =image[image.shape[0]//4:,:]
imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(imgray.shape)
plt.figure("Image") # 图像窗口名称
# print(image[image.shape[0]//4:,:,:].shape)
# for bit in image[:,50:52]:
#     print(bit)

# plt.imshow(imgray[imgray.shape[0]//4:,:])#-image[100,100])
# plt.axis('on') # 关掉坐标轴为 off
# plt.title('image') # 图像题目
# plt.show()
ret,thresh = cv2.threshold(imgray,127,255,0)
imagec, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(image.shape[:2], dtype = np.uint8)+255
print(type(contours[3]))
cv2.polylines(mask, [contours[3]], True, 125)
cv2.fillPoly(mask, [contours[3]], 0)
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(image, image, mask=mask_inv)

img1gray = cv2.cvtColor(img1_bg,cv2.COLOR_BGR2GRAY)
edge_img1 = cv2.Canny(img1gray, 10, 50)
imagec, contours, hierarchy = cv2.findContours(edge_img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
lines = cv2.HoughLinesP(edge_img1, 0.8, np.pi / 180, 90,
                        minLineLength=50, maxLineGap=10)
print(len(lines))

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
# ret1,thresh1 = cv2.threshold(img1gray,30,255,0)


# cv2.drawContours(image, contours, -1, (0, 155, 0), thickness=5)
cv2.imwrite("abc2.jpg", image)
# print(imagec.shape, contours)
# for i in range(0,len(contours)):  
#     x, y, w, h = cv2.boundingRect(contours[i])   
#     cv2.rectangle(image, (x,y), (x+w,y+h), (153,153,0), 5) 
#     newimage=image[y+2:y+h-2,x+2:x+w-2] # 先用y确定高，再用x确定宽
#     nrootdir=("cut_image/")
#     if not os.path.isdir(nrootdir):
#         os.makedirs(nrootdir)
#     cv2.imwrite( nrootdir+str(i)+".jpg",newimage) 
#     # print (i)
cv2.imwrite("abc.jpg", edge_img1)