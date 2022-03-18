import cv2
import os, sys
import glob

import math
import numpy as np

from matplotlib import pyplot as plt

PATH_TO_IMGS = "./imgs/"
PATH_TO_IMGS_Q2 = "./questao2/"

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
            titles = ['Original Gray Image','Histogram','Automatic Thresholding '+str(int(second_th))]

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

            titles = ['Original Gray Image','Histogram',"Otsu's Thresholding"]

            # Plot da imagem original
            plt.subplot(1,2,1),plt.imshow(gray_image,'gray')
            plt.title(titles[0]), plt.xticks([]), plt.yticks([])

            # Plot da imagem resultante do Otsu
            plt.subplot(1,2,2),plt.imshow(img_otsu,'gray')
            plt.title(titles[2]), plt.xticks([]), plt.yticks([])

            plt.savefig(PATH_TO_IMGS_Q2+img_file_out.split(".")[0]+"b.png")
            plt.close()          

            ################################################### C

            # print("\nEstatísticas nosso método")
            # connected_component_label(img_our_method)
            print("\nEstatísticas Otsu")
            connected_component_label(img_otsu)
            print("-------------------------------------------")

        print("questao2 executada")

    def openDir(self):
        files = glob.glob(PATH_TO_IMGS+"*.png")
        # print(files)
        return files      

def connected_component_label(img):

    num_labels, labels = cv2.connectedComponents(img)
    
    # Definindo as cores para os labels
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Definindo cor do fundo como preto
    labeled_img[label_hue==0] = 0
    
    # Mostrando a imagem original
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("Imagem original")
    # plt.show()
    
    # Imagem depois dos labels
    final_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)

    print("Tamanho em pixels de cada célula")
    pixels_cores_dif=np.unique(final_img.reshape(-1, final_img.shape[2]), axis=0, return_counts = True)[-1]
    pixels_cores_dif= np.delete(pixels_cores_dif, 0) # tira o preto
    print(pixels_cores_dif)
    print("Tamanho médio de uma célula")
    print(np.mean(pixels_cores_dif))
    print("Desvio padrão de tamanho de uma célula")
    print(np.std(pixels_cores_dif))
    print("Total de labels")
    print(num_labels-1)

    # plt.imshow(final_img)
    # plt.axis('off')
    # plt.title("Imagem depois dos labels")
    # plt.show()


if __name__ == "__main__":

    Questao2().openDir()


