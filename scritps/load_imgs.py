import cv2
import sys


img = cv2.imread('/home/leticia/Documentos/ComputacionalVision/ComputerVision/imgs/composite_01.png')

cv2.namedWindow('Janela')
cv2.imshow('Janela', img)
cv2.waitKey()