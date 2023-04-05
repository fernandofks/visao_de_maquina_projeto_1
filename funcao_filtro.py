import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *

#img_in = cv2.imread("Tratamento_blobs\Figuras\Pecas_Variadas.png", cv2.IMREAD_GRAYSCALE)
#img_bin = np.where(img_in>128, 255, 0).astype("uint8")

#kernel = np.ones((3,3),np.uint8)

def close(img,kernel):
    erosion = cv2.erode(img, kernel, iterations = 1)
    dilation = cv2.dilate(erosion,kernel, iterations = 1) 
    return dilation

def open(img,kernel):
    dilation = cv2.dilate(img,      kernel, iterations = 1) 
    erosion  = cv2.erode (dilation, kernel, iterations = 1)
    return dilation


#img_out = close(img_bin,kernel)


#pneuzinhos - inercia_alta e convexidade alta
#covids     - inercia_alta e convexidade média
#parafusos  - inercia_baixa e convexidade alta
#fusosgran  - inercia_baixa e convexidade baixa

# Set up the detector_pneus with default parameters.

def filtro(img_in, byColor, byArea, byCircularity, byConvexity,byInertia,
           Color = 0, Area_min= 0.01, Area_max= 1.0,
           Circularity_min= 0.01, Circularity_max= 1.0,
           Convexity_min= 0.01,Convexity_max= 1.0,
           Inertia_min= 0.01,Inertia_max= 1.0):

    filtrar = cv2.SimpleBlobDetector_Params()
    
    filtrar.filterByColor = byColor
    if byColor == True:
        filtrar.blobColor = Color
    
    filtrar.filterByArea = byArea
    if byArea == True:
        filtrar.minArea = Area_min
        filtrar.maxArea = Area_max
        
    filtrar.filterByCircularity = byCircularity
    if byCircularity == True:
        filtrar.minCircularity = Circularity_min
        filtrar.maxCircularity = Circularity_max
    
    filtrar.filterByConvexity = byConvexity
    if byConvexity == True:
        filtrar.minConvexity = Convexity_min
        filtrar.maxConvexity = Convexity_max
        
    filtrar.filterByInertia = byInertia
    if byInertia == True:
        filtrar.minInertiaRatio = Inertia_min
        filtrar.maxInertiaRatio = Inertia_max    
    
    detector = cv2.SimpleBlobDetector_create(filtrar)
    filtrado = detector.detect(img_in)
    
    return filtrado

#pneus = filtro(img_out, True, False, False, True, True, Color = 0, 
#               Convexity_min=0.87,Convexity_max=1, Inertia_min=0.8,Inertia_max=1)

# List parameters (X,Y,size,ang) of each detected keypoints

#img1_text = cv2.cvtColor(img_out,cv2.COLOR_GRAY2RGB)#pneus

#i=1
#for KPi in pneus:
    #print("Blob_", i, ": X= ", KPi.pt[0], " Y= ", KPi.pt[1], " size=", KPi.size**2, " ang=", KPi.angle)
#    img1_text = cv2.putText(img1_text, str(i), (int(KPi.pt[0]),int(KPi.pt[1])), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)
#    i=i+1
#display image (with text)

#cv2.imshow("Img1 with texts", img1_text)
#cv2.waitKey(0)

