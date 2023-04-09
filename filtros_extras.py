import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *

#Conforme visto na aula 3 desafio 2
#aceita apenas imagens coloridas
def grayscale_especial (img_in,const_red=1.0,const_green=1.0,const_blue=1.0):
    [B,G,R] = cv2.split(img_in)

    (height,width) = R.shape

    R_bin = np.zeros((height,width), dtype = 'int32')
    G_bin = np.zeros((height,width), dtype = 'int32')
    B_bin = np.zeros((height,width), dtype = 'int32')

    for i in range (height-1):
        for j in range (width-1):
            R_bin[i,j] = R[i,j]*const_red
            G_bin[i,j] = G[i,j]*const_green
            B_bin[i,j] = B[i,j]*const_blue

    Merged = R_bin+G_bin+B_bin
    Merged = np.clip(Merged, 0, 255).astype(np.uint8)
    return Merged

#Conforme visto na aula 7 exercício 6
#aceita apenas inputs de imagens grayscale
def filtro_nitidez (img_in):
    #img_in = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    (height,width) = img_in.shape
    
    Nitidez_kernel = [[0,-1,0],[-1,5,-1],[0,-1,0]]
    filt_nitidez = np.zeros((height,width), dtype="int32")
    m = 3 # tamanho da matriz gaussiana
    d = int((m-1)/2)
    
    for i in range (d, height-d):
        for j in range (d, width-d):
            
            secao_img = img_in[i-d:i+d+1, j-d:j+d+1]
            prod_imag_Nitidez_kernel = Nitidez_kernel * secao_img + Nitidez_kernel
            filt_nitidez[i,j] = np.clip(abs(prod_imag_Nitidez_kernel.sum()), 0, 255).astype(np.uint8)
    
    return filt_nitidez

#Conforme visto na aula 11 exercício 1
#acho que aceita apenas imagens gratscale (canal único)
def filtro_blobs(img_in, byColor, byArea, byCircularity, byConvexity,byInertia,
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

def close(img,kernel = np.ones((3,3),np.uint8)):
    dilation = cv2.dilate(img,      kernel, iterations = 1) 
    erosion  = cv2.erode (dilation, kernel, iterations = 1)
    return erosion

def open(img,kernel= np.ones((3,3),np.uint8)):
    erosion = cv2.erode(img, kernel, iterations = 1)
    dilation = cv2.dilate(erosion,kernel, iterations = 1) 
    return dilation