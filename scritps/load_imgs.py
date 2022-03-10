import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt

PATH_TO_IMGS = "./imgs/"

class LoadImage():
    def __init__(self):

        files = self.openDir()
        # sizeOfImgs = files.size

        images = []

        for img_file in files:

            #Leitura das imagens e separação dos canais
            img = cv2.imread(img_file)
            [imB,imG,imR] = cv2.split(img)

            #Subtração do Canal Vermelho pelo Canal Azul
            imgRmB = imR - imB

            #Normalização da subtração
            h,w = imgRmB.shape[:2]
            RmBBinary = cv2.normalize(imgRmB, np.zeros((h,w)), 0, 255, cv2.NORM_MINMAX)

            #Filtragem dos ruídos da imagem
            imgGaussian = cv2.GaussianBlur(imgRmB, (5,5), 0)

            #Aplicando Otsu na imagem filtrada 
            ret, threshold = cv2.threshold(imgGaussian, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            #If = imgRmB | Ib = threshold
            #UTILIZANDO SOBEL NESTA PARTE
            grad_sobel_x = cv2.Sobel(imgRmB, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_sobel_y = cv2.Sobel(imgRmB, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            # cv2.imshow("SOBEL X",grad_sobel_x)
            # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            # cv2.imshow("SOBEL Y",grad_sobel_y)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            abs_grad_x = cv2.convertScaleAbs(grad_sobel_x)
            abs_grad_y = cv2.convertScaleAbs(grad_sobel_y)

            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            cv2.imshow("SOBEL",cv2.resize(grad,(800,600)))
            # cv2.imwrite(PATH_TO_IMGS+"sobel_"+str(os.path.basename(img_file)), grad)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #UTILIZANDO LAPLACIANO NESTA PARTE
            grad_laplace = cv2.Laplacian(imgRmB, cv2.CV_16S, ksize=3)

            abs_grad = cv2.convertScaleAbs(grad_laplace)

            cv2.imshow("LAPLACIAN",cv2.resize(abs_grad,(800,600)))
            # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #UTILIZANDO CANNY
            canny = cv2.Canny(imgRmB,100,200)
            cv2.imshow("CANNY",cv2.resize(canny,(800,600)))
            # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #TESTE SOMA CANNY SOBEL
            sobel_canny = canny + grad
            cv2.imshow("CANNY+SOBEL",cv2.resize(sobel_canny,(800,600)))
            # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #TESTE SOMA LAPLACIAN SOBEL
            sobel_laplacian = abs_grad + grad
            cv2.imshow("LAPLACIAN+SOBEL",cv2.resize(sobel_laplacian,(800,600)))
            # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def openDir(self):
        files = glob.glob(PATH_TO_IMGS+"*.png")
        print(files)
        return files
        

        

if __name__ == "__main__":
    
    # images = []
    LoadImage().openDir()

    titles = ['Composite 02 - Sobel','Composite 03 - Sobel', 'Composite 04 - Sobel', 'Composite 01 - Sobel']

    plt.subplot(4,1,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])