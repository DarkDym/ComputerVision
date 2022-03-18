import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import imutils

#TESTE DO SCIKIT PARA WATERSHEDS
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import color
from skimage.measure import regionprops
from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray


PATH_TO_IMGS = "./imgs/"

SHOW_IMGS = 1

QNT_CELLS_EYECOUNTED = [11, 20, 12, 12]

class LoadImage():
    def __init__(self):
        print("=================================STRATING PROGRAM=================================")
        self.count = 0

    def water_transform(self):
        files = self.openDir()

        images_raw = []
        watersheds = []
        result_imgs = []

        for img_file in files:

            #Leitura das imagens e separação dos canais
            print("============================ARQUIVO: "+str(os.path.basename(img_file))+"============================")
            print("============================QNT CELLS EYECOUNTED:" + str(QNT_CELLS_EYECOUNTED[self.count]) + "============================")
            self.count = self.count + 1
            img = cv2.imread(img_file)
            imgcp2 = img.copy()
            images_raw.append(img)
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
            otsu = threshold

            #If = imgRmB | Ib = threshold
            #UTILIZANDO SOBEL NESTA PARTE
            grad_sobel_x = cv2.Sobel(imgGaussian, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_sobel_y = cv2.Sobel(imgGaussian, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

            abs_grad_x = cv2.convertScaleAbs(grad_sobel_x)
            abs_grad_y = cv2.convertScaleAbs(grad_sobel_y)

            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


            kernel = np.ones((3,3), np.uint8)
            opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

            back = cv2.dilate(opened,kernel,iterations=3)     

            #=======================================WATERSHED OPENCV=======================================
            #=======================================WATERSHED OPENCV=======================================

            dist_transform = cv2.distanceTransform(opened,cv2.DIST_L2,5)
            ret2, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(back,sure_fg)
            ret3, markers = cv2.connectedComponents(sure_fg)
            markers = markers+10
            markers[unknown==255] = 0

            markers = cv2.watershed(img,markers)

            img[markers == -1] = [0,255,255]  
            img2 = color.label2rgb(markers, bg_label=0)
            
            connectivity=4
            imgT = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            nb_comp,output,sizes,centroids=cv2.connectedComponentsWithStats(imgT,connectivity)
            nb_comp-=1; sizes=sizes[0:,-1]; centroids=centroids[1:,:]
            print("=======================QNT CELLS WATERSHED OPENCV: "+str(len(np.unique(markers))-2)+"===================================")
            for label in np.unique(markers):
                if label == 0:
                    continue
                if label > -1 and label != 10:
                    mask = np.zeros(imgT.shape, dtype="uint8")
                    mask[markers == label] = 255

                    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    c = max(cnts, key=cv2.contourArea)

                    ((x, y), r) = cv2.minEnclosingCircle(c)
                    cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
                    cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            #=======================================WATERSHED OPENCV=======================================
            #=======================================WATERSHED OPENCV=======================================

            #=======================================QUESTÂO 4=======================================
            #Para executar a verificação da questão 4, descomentar um por vez as linhas 134 e 135. 
            #Para executar a verificação do min_dist mencionado no relatório, descomentar as linhas 138 a 141 uma por execução
            canny = cv2.Canny(imgGaussian,50,50)
            sobel_canny = cv2.add(canny,grad)
            sobel_canny_filtered = cv2.GaussianBlur(sobel_canny, (3,3), 0)
            grad_laplace = cv2.Laplacian(imgGaussian, cv2.CV_16S, ksize=3)
            abs_grad = cv2.convertScaleAbs(grad_laplace)
            sobel_laplacian = cv2.add(abs_grad,grad)
            sobel_laplacian_filtered = cv2.GaussianBlur(sobel_laplacian, (3,3), 0)
            #=======================================QUESTÂO 4=======================================

            #=======================================WATERSHED SCIKIT=======================================
            #=======================================WATERSHED SCIKIT=======================================
            otsu = cv2.dilate(otsu,kernel,iterations=3)
            otsu = cv2.erode(otsu,kernel,iterations=3)

            sciImg = ndi.distance_transform_edt(grad)
            # sciImg = ndi.distance_transform_edt(sobel_canny_filtered) #Para a questão 4
            # sciImg = ndi.distance_transform_edt(sobel_laplacian_filtered) #Para a questão 4
            localMax = peak_local_max(sciImg, indices=False, min_distance=20, labels=otsu)
            # localMax = peak_local_max(sciImg, indices=False, min_distance=10, labels=otsu) #Para a questão 4
            # localMax = peak_local_max(sciImg, indices=False, min_distance=30, labels=otsu) #Para a questão 4
            # localMax = peak_local_max(sciImg, indices=False, min_distance=40, labels=otsu) #Para a questão 4
            # localMax = peak_local_max(sciImg, indices=False, min_distance=50, labels=otsu) #Para a questão 4
            markersSci = ndi.label(localMax, structure=np.ones((3,3)))[0]
            labels = watershed(-sciImg, markersSci, mask=otsu)
            ws = len(np.unique(labels)) - 1
            print("=======================QNT CELLS SCIKIT: "+str(ws)+" ================================")
            for label in np.unique(markersSci):
                if label == 0:
                    continue
                mask = np.zeros(imgT.shape, dtype="uint8")
                mask[markersSci == label] = 255

                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)

                    # draw a circle enclosing the object
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(imgcp2, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.putText(imgcp2, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            regions = regionprops(labels)
            count_region = 0
            total_area = 0
            total_std = 0
            for r in regions:
                print("+++++++++++++++++++LABEL: "+str(r.label)+"+++++++++++++++++++++++++++")
                # print("+++++++++++++++++++AREA: "+str(r.area)+"+++++++++++++++++++++++++++")
                print("+++++++++++++++++++STD: "+str(np.std(r.image.astype(float)))+"+++++++++++++++++++++++++++")
                count_region = count_region + 1
                total_area = total_area + r.area
                total_std = total_std + np.std(r.image.astype(float))
            print("+++++++++++++++++++MEAN AREA: "+str(total_area/count_region)+"+++++++++++++++++++++++++++")
            print("+++++++++++++++++++MEAN_STD: "+str(total_std/count_region)+"+++++++++++++++++++++++++++")

            result_imgs.append(imgcp2)
            #=======================================WATERSHED SCIKIT=======================================
            #=======================================WATERSHED SCIKIT=======================================
            
            #Para a visualização de todas as imagens geradas dentro do script, deixar a variável 'SHOW_IMGS = 1', caso contrário deixar ela '0''.
            if SHOW_IMGS:

                cv2.imshow("SOBEL",cv2.resize(grad,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("BACKGROUND",cv2.resize(back,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("FOREGROUND",cv2.resize(sure_fg,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("UNKNOW",cv2.resize(unknown,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("BINARIZACAO OTSU",cv2.resize(otsu,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("NEW WATERSHED",cv2.resize(img,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("NEW WATERSHED 2",cv2.resize(img2,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("OPENED",cv2.resize(opened,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("COUNTED WATERSHED OPENCV",cv2.resize(img,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow("SCIKIT WATERSHED",cv2.resize(imgcp2,(800,600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("\n")

        return images_raw,watersheds,result_imgs

    def openDir(self):
        files = glob.glob(PATH_TO_IMGS+"*.png")
        print(files)
        return files
              

if __name__ == "__main__":
    
    ir,ws,ri = LoadImage().water_transform()