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

def ajuste_brilho_contraste(img_in,brilho,contraste):
    (height,width) = img_in.shape
    img_in = img_in.astype(np.int32)
    img_out = np.zeros((height,width), dtype = "uint8")

    for i in range (height-1):
        for j in range (width-1):
            intens32 = img_in[i,j]*contraste + brilho       
            img_out[i,j] = np.clip(intens32, 0, 255).astype(np.uint8)
    return img_out

#código fonte: https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv
def CorrecaoGamma(img_in, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    img_out = cv2.LUT(img_in, table)
    return img_out

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

def super_filtro (img_in,type):
    (height,width) = img_in.shape
    
    m = 3 # tamanho da matriz gaussiana
    d = int((m-1)/2)
    sigma = 2 
    
    PrewittX_kernel = [[-1,0,1],[-1,0,1],[-1,0,1]]
    PrewittY_kernel = [[-1,-1,-1],[0,0,0],[1,1,1]]
    SobelX_kernel = [[-1,0,1],[-2,0,2],[-1,0,1]]
    SobelY_kernel = [[-1,-2,-1],[0,0,0],[1,2,1]]
    Laplacian_kernel = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    Nitidez_kernel = [[0,-1,0],[-1,5,-1],[0,-1,0]]
    
    
    filt_sobelX = np.zeros((height,width), dtype="int32")
    filt_sobelY = np.zeros((height,width), dtype="int32")
    filt_prewittX = np.zeros((height,width), dtype="int32")
    filt_prewittY = np.zeros((height,width), dtype="int32")
    filt_sobelConv = np.zeros((height,width), dtype="int32")
    filt_derivada_segunda = np.zeros((height,width), dtype="int32")
    filt_nitidez = np.zeros((height,width), dtype="int32")
    
    for i in range (d, height-d):
        for j in range (d, width-d):
            
            secao_img = img_in[i-d:i+d+1, j-d:j+d+1] 
            
            prod_imag_SobX_kernel    = SobelX_kernel * secao_img
            prod_imag_SobY_kernel    = SobelY_kernel * secao_img
            prod_imag_PreX_kernel    = PrewittX_kernel * secao_img
            prod_imag_PreY_kernel    = PrewittY_kernel * secao_img
            prod_imag_d2_kernel      = Laplacian_kernel * secao_img
            prod_imag_Nitidez_kernel = Nitidez_kernel * secao_img + Nitidez_kernel
            

            filt_sobelX[i,j]           = np.clip( abs(prod_imag_SobX_kernel.sum()), 0, 255).astype(np.uint8)
            filt_sobelY[i,j]           = np.clip(abs(prod_imag_SobY_kernel.sum()), 0, 255).astype(np.uint8)
            filt_prewittX[i,j]             = np.clip(abs(prod_imag_PreX_kernel.sum()), 0, 255).astype(np.uint8)
            filt_prewittY[i,j]             = np.clip(abs(prod_imag_PreY_kernel.sum()), 0, 255).astype(np.uint8)
            filt_sobelConv[i,j]        = np.clip(filt_sobelX[i,j] + filt_sobelY[i,j], 0, 255).astype(np.uint8)
            filt_derivada_segunda[i,j] = np.clip(abs(prod_imag_d2_kernel.sum()), 0, 255).astype(np.uint8)
            filt_nitidez[i,j]          = np.clip(abs(prod_imag_Nitidez_kernel.sum()), 0, 255).astype(np.uint8)
            
    if type == "derivada":
        return filt_derivada_segunda
    
    elif type == "nitidez":
        return filt_nitidez
    
    else:
        return "filter not specified on function, go to filtros_extras.py and work it out"

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