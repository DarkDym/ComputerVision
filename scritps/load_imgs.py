import cv2
import os, sys

import math
import numpy as np

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
    cv2.namedWindow('JanelaGRAY'+str(i))
    cv2.imshow('JanelaGRAY'+str(i), gray_images[i])

cv2.waitKey()

print("QUESTÃO 1 FEITA")
cv2.waitKey()
