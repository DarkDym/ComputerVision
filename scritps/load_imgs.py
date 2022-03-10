import cv2
import os, sys

import math
import numpy as np

maior_dif=0
i_img=-1
img = []
for i in range(0,4):
    img.append(cv2.imread('/home/leticia/Documentos/ComputacionalVision/ComputerVision/imgs/composite_0'+str(i+1)+'.png'))
    # cv2.namedWindow('Janela'+str(i))
    # cv2.imshow('Janela'+str(i), img[i])

    B, G, R = cv2.split(img[i])
    diferenca=np.average(R-B)
    # print(diferenca)
    if diferenca > maior_dif:
        maior_dif = diferenca
        i_img=i
# cv2.waitKey()

print('Escolhido '+str(i_img+1)+' com '+str(maior_dif))
