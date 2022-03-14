import cv2
import os, sys
import glob

import math
import numpy as np

from matplotlib import pyplot as plt

PATH_TO_IMGS = "./imgs/"
PATH_TO_IMGS_Q1 = "./questao1/"
PATH_TO_IMGS_Q2 = "./questao2/"

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


class Questao2:
    def __init__(self):
        files = self.openDir()

        for img_file in files:
            # Carrega imagem
            img = cv2.imread(img_file)
            img_file_out=img_file.split("/")[-1]
            height, width = img.shape[:2]
            norm=np.zeros((height,width))
            final=cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)
            B, G, R = cv2.split(final)
            gray_image=np.full((height, width), R-B, dtype=np.uint8)

            ################################################### A
            # Estratégia proposta por nós
            initial_th=np.mean(gray_image)
            ret_first,img_first = cv2.threshold(gray_image,initial_th,255,cv2.THRESH_BINARY)
            # cv2.imshow('Janela', img_first)
            # cv2.waitKey()
            second_th=np.mean(img_first)
            ret_second,img_our_method = cv2.threshold(gray_image,second_th,255,cv2.THRESH_BINARY)

            # plot all the images and their histograms
            images = [gray_image, 0, img_our_method]
            titles = ['Original Gray Image','Histogram','Global Thresholding '+str(int(second_th))]

            # Plot da imagem original
            plt.subplot(1,2,1),plt.imshow(gray_image,'gray')
            plt.title(titles[0]), plt.xticks([]), plt.yticks([])

            # Plot da imagem resultante do nosso método
            plt.subplot(1,2,2),plt.imshow(img_our_method,'gray')
            plt.title(titles[2]), plt.xticks([]), plt.yticks([])

            print(img_file_out.split(".")[0])
            plt.savefig(PATH_TO_IMGS_Q2+img_file_out.split(".")[0]+"a.png")
            plt.close()

            # Histograma com eixo y limitado para facilitar a visualização
            plt.hist(gray_image.ravel(),256)
            plt.title(titles[1]), plt.xticks([]), plt.yticks([])
            ax = plt.gca()
            ax.set_ylim([0, 100])
            # Primeiro limiar em preto
            plt.axvline(int(initial_th), color='k', linestyle='dashed', linewidth=2)
            # Segundo e limiar final em vermelho
            plt.axvline(int(second_th), color='r', linestyle='dashed', linewidth=2)

            plt.savefig(PATH_TO_IMGS_Q2+img_file_out.split(".")[0]+"aHIST.png")
            plt.close()

            ################################################### B
            #  Otsu's thresholding
            ret_otsu,img_otsu = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # plot all the images and their histograms
            images = [gray_image, 0, img_otsu]
            titles = ['Original Gray Image','Histogram',"Otsu's Thresholding"]

            # Plot da imagem original
            plt.subplot(1,2,1),plt.imshow(gray_image,'gray')
            plt.title(titles[0]), plt.xticks([]), plt.yticks([])

            # Plot da imagem resultante do Otsu
            plt.subplot(1,2,2),plt.imshow(img_our_method,'gray')
            plt.title(titles[2]), plt.xticks([]), plt.yticks([])

            plt.savefig(PATH_TO_IMGS_Q2+img_file_out.split(".")[0]+"b.png")
            plt.close()          

            ################################################### C



        print("questao2 executada")

    def openDir(self):
        files = glob.glob(PATH_TO_IMGS+"*.png")
        # print(files)
        return files      

if __name__ == "__main__":
    
    # Questao1().openDir()

    Questao2().openDir()


