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
SHOW_IMGS2 = 1
SAVE_IMG = True

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



            #=======================================WATERSHED SCIKIT=======================================
            #=======================================WATERSHED SCIKIT=======================================
            otsu = cv2.dilate(otsu,kernel,iterations=3)
            otsu = cv2.erode(otsu,kernel,iterations=3)

            sciImg = ndi.distance_transform_edt(grad)
            localMax = peak_local_max(sciImg, indices=False, min_distance=20, labels=otsu)
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
            
            #Para a visualização de todas as imagens geradas dentro do script, deixar a variável 'SHOW_IMGS = True', caso contrário deixar ela falsa.
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

#Exercise3() é uma função que foi utilizada para o desenvolvimento dos exercícios propostos, os resultados obtidos foram transferidos para a função water_transform()
    def Exercise3(self):

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
            images_raw.append(img)
            [imB,imG,imR] = cv2.split(img)

            #Subtração do Canal Vermelho pelo Canal Azul
            imgRmB = imR - imB

            #Normalização da subtração
            h,w = imgRmB.shape[:2]
            RmBBinary = cv2.normalize(imgRmB, np.zeros((h,w)), 0, 255, cv2.NORM_MINMAX)

            #Filtragem dos ruídos da imagem
            imgGaussian = cv2.GaussianBlur(RmBBinary, (5,5), 0)

            #Aplicando Otsu na imagem filtrada 
            ret, threshold = cv2.threshold(imgGaussian, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            otsu = threshold

            #If = imgRmB | Ib = threshold
            #UTILIZANDO SOBEL NESTA PARTE
            grad_sobel_x = cv2.Sobel(imgGaussian, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_sobel_y = cv2.Sobel(imgGaussian, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            # if SHOW_IMGS:
            #     cv2.imshow("SOBEL X",grad_sobel_x)
            #     cv2.waitKey(0)
            #     # cv2.destroyAllWindows()
            #     cv2.imshow("SOBEL Y",grad_sobel_y)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            abs_grad_x = cv2.convertScaleAbs(grad_sobel_x)
            abs_grad_y = cv2.convertScaleAbs(grad_sobel_y)

            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            if SHOW_IMGS:
                cv2.imshow("SOBEL",cv2.resize(grad,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"sobel_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #UTILIZANDO LAPLACIANO NESTA PARTE
            grad_laplace = cv2.Laplacian(imgGaussian, cv2.CV_16S, ksize=3)

            abs_grad = cv2.convertScaleAbs(grad_laplace)

            if SHOW_IMGS:
                cv2.imshow("LAPLACIAN",cv2.resize(abs_grad,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), abs_grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #UTILIZANDO CANNY
            canny = cv2.Canny(imgGaussian,100,200)
            if SHOW_IMGS:
                cv2.imshow("CANNY",cv2.resize(canny,(800,600)))
                cv2.imwrite(PATH_TO_IMGS+"canny_"+str(os.path.basename(img_file)), canny)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #TESTE SOMA CANNY SOBEL
            sobel_canny = cv2.add(canny,grad)
            if SHOW_IMGS:
                cv2.imshow("CANNY+SOBEL",cv2.resize(sobel_canny,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #TESTE SOMA LAPLACIAN SOBEL
            sobel_laplacian = cv2.add(abs_grad,grad)
            if SHOW_IMGS:
                cv2.imshow("LAPLACIAN+SOBEL",cv2.resize(sobel_laplacian,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #PREPARAÇÂO PARA O WATERSHED
            if SHOW_IMGS:
                cv2.imshow("OTSU",cv2.resize(threshold,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            kernel = np.ones((3,3), np.uint8)
            opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
            if SHOW_IMGS:
                cv2.imshow("OPENED",cv2.resize(opened,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #BACKGROUND
            back = cv2.dilate(opened,kernel,iterations=3)
            if SHOW_IMGS:
                cv2.imshow("BACK_DILATE",cv2.resize(back,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            ret,bg = cv2.threshold(back,1,128,1)
            if SHOW_IMGS:
                cv2.imshow("BACK_DILATE_THRESHOLD",cv2.resize(bg,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #FOREGROUND
            fore = cv2.erode(opened,kernel,iterations=1)
            if SHOW_IMGS:
                cv2.imshow("FORE_ERODE",cv2.resize(fore,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #MARKER
            marker = cv2.add(fore,bg)
            if SHOW_IMGS:
                cv2.imshow("MARKER",cv2.resize(marker,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #CONVERT TO MARKER32
            marker32 = np.int32(marker)
            ret = cv2.watershed(img,marker32)
            # ret = cv2.watershed(grad,marker32)
            imgcp = img.copy()
            imgcp2 = img.copy()
            imgcp[ret==-1] = [255,0,0]
            
            if SHOW_IMGS2:
                cv2.imshow("TESTE",cv2.resize(imgcp,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            m = cv2.convertScaleAbs(marker32)
            if SHOW_IMGS2:
                cv2.imshow("WATERSHED",cv2.resize(m,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            watersheds.append(m)

            #WATERSHED APPLIED TO IMAGE
            ret, threshold = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            res = cv2.bitwise_and(img,img,mask = threshold)
            th = cv2.inRange(imgcp,(255,0,0),(255,0,0)).astype(np.uint8)
            restst = cv2.bitwise_not(th)
            if SHOW_IMGS:
                cv2.imshow("RESTST",cv2.resize(th,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_"+str(os.path.basename(img_file)), res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if SHOW_IMGS:
                cv2.imshow("RESULTADO",cv2.resize(res,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_"+str(os.path.basename(img_file)), res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            result_imgs.append(res)


            # print("================================QNT CELLS: " + str(output[0]) + "================================")

            dist_transform = cv2.distanceTransform(opened,cv2.DIST_L2,5)
            ret2, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(back,sure_fg)
            ret3, markers = cv2.connectedComponents(sure_fg)
            markers = markers+10
            markers[unknown==255] = 0

            # watershed 
            markers = cv2.watershed(img,markers)

            img[markers == -1] = [0,255,255]  
            img2 = color.label2rgb(markers, bg_label=0)
            if SHOW_IMGS:
                cv2.imshow("SURE_FG",cv2.resize(sure_fg,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_"+str(os.path.basename(img_file)), res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if SHOW_IMGS:
                cv2.imshow("UNKNOW",cv2.resize(unknown,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_"+str(os.path.basename(img_file)), res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if SHOW_IMGS2:
                cv2.imshow("NEW WATERSHED",cv2.resize(img,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_"+str(os.path.basename(img_file)), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if SHOW_IMGS2:
                cv2.imshow("NEW WATERSHED 2",cv2.resize(img2,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_contrast_"+str(os.path.basename(img_file)), img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            connectivity=4
            imgT = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if SHOW_IMGS:
                cv2.imshow("GRAY",cv2.resize(imgT,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_"+str(os.path.basename(img_file)), res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            nb_comp,output,sizes,centroids=cv2.connectedComponentsWithStats(imgT,connectivity)
            nb_comp-=1; sizes=sizes[0:,-1]; centroids=centroids[1:,:]
            # print("===================================NB_COMP: "+str(nb_comp)+"===================================")
            # print(np.unique(markers))
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
            if SHOW_IMGS2:
                cv2.imshow("COUNTED",cv2.resize(img,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_counted"+str(os.path.basename(img_file)), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            #=======================================WATERSHED SCIKIT=======================================
            #=======================================WATERSHED SCIKIT=======================================
            otsu = cv2.dilate(otsu,kernel,iterations=3)
            otsu = cv2.erode(otsu,kernel,iterations=3)
            if SHOW_IMGS2:
                cv2.imshow("OTSU",cv2.resize(otsu,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"laplacian_"+str(os.path.basename(img_file)), grad)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            sciImg = ndi.distance_transform_edt(grad)
            localMax = peak_local_max(sciImg, indices=False, min_distance=20, labels=otsu)
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
            if SHOW_IMGS2:
                cv2.imshow("SCIIMG",cv2.resize(imgcp2,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_counted"+str(os.path.basename(img_file)), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            regions = regionprops(labels)
            for r in regions:
                print("+++++++++++++++++++LABEL: "+str(r.label)+"+++++++++++++++++++++++++++")
                print("+++++++++++++++++++AREA: "+str(r.area)+"+++++++++++++++++++++++++++")
                print("+++++++++++++++++++STD: "+str(np.std(r.image.astype(float)))+"+++++++++++++++++++++++++++")
            if SHOW_IMGS:
                cv2.imshow("SCIIMG",cv2.resize(imgcp2,(800,600)))
                # cv2.imwrite(PATH_TO_IMGS+"watershed_SCIKIT"+str(os.path.basename(img_file)), imgcp2)
                # cv2.imshow(str(r.label),cv2.resize(r.image.astype(float),(50,50)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            #=======================================WATERSHED SCIKIT=======================================
            #=======================================WATERSHED SCIKIT=======================================
            
            print("\n")

        return images_raw,watersheds,result_imgs

    def openDir(self):
        files = glob.glob(PATH_TO_IMGS+"*.png")
        print(files)
        return files
              

if __name__ == "__main__":
    
    # ir,ws,ri = LoadImage().water_transform()
    ir,ws,ri = LoadImage().Exercise3()

    # titles = ['Composite 02','Watershed 02','Resultado 02',
    #           'Composite 03','Watershed 03','Resultado 03',
    #           'Composite 04','Watershed 04','Resultado 04',
    #           'Composite 01','Watershed 01','Resultado 01']

    # plt.subplot(1,3,1),plt.imshow(ir[0])
    # plt.title(titles[0]), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,2),plt.imshow(ws[0])
    # plt.title(titles[1]), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,3),plt.imshow(ri[0])
    # plt.title(titles[2]), plt.xticks([]), plt.yticks([])
    # plt.show()
    
    # plt.subplot(1,3,1),plt.imshow(ir[1])
    # plt.title(titles[3]), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,2),plt.imshow(ws[1])
    # plt.title(titles[4]), plt.xticks([]), plt.yticks([])
    # plt.subplot(4,1,3),plt.imshow(ri[1])
    # plt.title(titles[5]), plt.xticks([]), plt.yticks([])
    # plt.show()
    
    # plt.subplot(1,3,1),plt.imshow(ir[2])
    # plt.title(titles[6]), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,2),plt.imshow(ws[2])
    # plt.title(titles[7]), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,3),plt.imshow(ri[2])
    # plt.title(titles[8]), plt.xticks([]), plt.yticks([])
    # plt.show()

    # plt.subplot(1,3,1),plt.imshow(ir[3])
    # plt.title(titles[9]), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,2),plt.imshow(ws[3])
    # plt.title(titles[10]), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,3),plt.imshow(ri[3])
    # plt.title(titles[11]), plt.xticks([]), plt.yticks([])
    # plt.show()

    # for i in range(4):
    #     print(i)
    #     plt.subplot(4,3,i*3+1),plt.imshow(ir[i])
    #     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(4,3,i*3+2),plt.imshow(ws[i])
    #     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(4,3,i*3+3),plt.imshow(ri[i])
    #     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    # plt.show()