import cv2
import os, sys

import math
import numpy as np

maior_dif=0
i_img=-1
imagens = []
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
    diferenca=np.average(R-B)
    # print(diferenca)
    if diferenca > maior_dif:
        maior_dif = diferenca
        i_img=i
# cv2.waitKey()

print('Escolhido '+str(i_img+1)+' com '+str(maior_dif))

B, G, R = cv2.split(imagens[i_img])
height, width = imagens[i_img].shape[:2]
gray_image = np.full((height, width), R-B, dtype=np.uint8)

cv2.namedWindow('ESCOLHIDA')
cv2.imshow('ESCOLHIDA', gray_image)
print("QUESTÃO 1 FEITA")
cv2.waitKey()
