import cv2
import os, sys

import math
import numpy as np

from matplotlib import pyplot as plt

maior_dif=0
i_img=-1
imagens = []
gray_images = []
for i in range(0,4):
    imagens.append(cv2.imread('/home/leticia/Documentos/ComputacionalVision/ComputerVision/imgs/composite_0'+str(i+1)+'.png'))
    # cv2.namedWindow('Janela'+str(i))
    # cv2.imshow('Janela'+str(i), img[i])

    height, width = imagens[i].shape[:2]
    norm=np.zeros((height,width))
    final=cv2.normalize(imagens[i], norm, 0, 255, cv2.NORM_MINMAX)

    # cv2.namedWindow('JanelaNORMALIZADO'+str(i))
    # cv2.imshow('JanelaNORMALIZADO'+str(i), final)

    B, G, R = cv2.split(final)
    gray_images.append(np.full((height, width), R-B, dtype=np.uint8))
    # cv2.namedWindow('JanelaGRAY'+str(i))
    # cv2.imshow('JanelaGRAY'+str(i), gray_images[i])

cv2.waitKey()

print("QUESTÃO 1 FEITA")
cv2.waitKey()

# HISTOGRAMAS
for i in range(0,4):
    img_hist = gray_images[i]
    # # max_value= np.average(img_hist)
    # # max_value= 2*int(max_value)
    # max_value=np.amax(img_hist)

    # print(max_value)

    # ax = plt.gca()
    # # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([0, 100])

    # plt.hist(img_hist.ravel(), max_value ,[0, max_value])
    # plt.show()

    # FONTE: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    
    # global thresholding
    ret1,th1 = cv2.threshold(img_hist,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img_hist,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_hist,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img_hist, 0, th1,
            img_hist, 0, th2,
            blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        ax = plt.gca()
        # ax.set_xlim([xmin, xmax])
        ax.set_ylim([0, 100])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


cv2.waitKey()
