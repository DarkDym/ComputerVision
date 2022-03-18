import cv2
import os, sys
import glob

import math
import numpy as np

from matplotlib import pyplot as plt

PATH_TO_IMGS = "./imgs/"
PATH_TO_IMGS_Q1 = "./questao1/"

class Questao1:
    def __init__(self):

        files = self.openDir()

        for img_file in files:
            # Carrega imagem
            img = cv2.imread(img_file)
            print("Usando imagem: "+img_file)
            img_file_out=img_file.split("/")[-1]
            # cv2.imshow('Imagem '+img_file_out, img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Normaliza para ajudar no processamento posterior
            height, width = img.shape[:2]
            norm=np.zeros((height,width))
            final=cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)
            # cv2.imshow('Imagem NORMALIZADA '+img_file_out, final)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Faz a diferença entre os canais e salva
            B, G, R = cv2.split(final)
            gray_image=np.full((height, width), R-B, dtype=np.uint8)
            cv2.imwrite(PATH_TO_IMGS_Q1+ img_file_out, gray_image)
            # cv2.imshow('Imagem GRAY '+img_file_out, gray_image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        
        print("questao1 executada")


    def openDir(self):
        files = glob.glob(PATH_TO_IMGS+"*.png")
        # print(files)
        return files


if __name__ == "__main__":
    
    Questao1().openDir()


