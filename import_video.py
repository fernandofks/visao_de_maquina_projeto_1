import cv2
import os
 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd # para desenhar/plotar as tabelas informando os status das peças em cada teste

from filtros_extras import *

# Load the video file
cap = cv2.VideoCapture('Video1_Vedacao.mp4')
count =  0
#cap.set(cv2.CAP_PROP_POS_MSEC,180000)

# Loop through the frames in the video
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()
        
    if ret:
        # Convert the frame to grayscale
        [B,G,R] = cv2.split(frame)
        (height,width) = R.shape
        
        # binarizando a imagem
        returns,thresh=cv2.threshold(B,70,255,cv2.THRESH_BINARY_INV)
        thresh = close(thresh)
        
        contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
        img1_text = cv2.cvtColor(B,cv2.COLOR_GRAY2RGB)
        
        if len(contours) != 0:
            Peca_no_frame = True
            perimetro_max = -1
            contorno_out = contours[0]
            for j in range(len(contours)):
                #achar o perimetro perimetro_maximo para gerar esse contorno
                perimetro = cv2.arcLength(contours[j],True)
                if perimetro > perimetro_max:
                    perimetro_max = perimetro
                    contorno_out = contours[j]

            cv2.drawContours(img1_text,contorno_out,-1,(0,0,255),4)
            hull = cv2.convexHull(contorno_out)
            cv2.drawContours(img1_text,hull,-1,(0,255,0),8)
    
            area_contorno = cv2.contourArea(contorno_out)
            area_hull = cv2.contourArea(hull)
            if area_hull >= 1: #evitar que o video crashe por conta de divisão por 0
                convexidade = area_contorno/area_hull
        else:
            Peca_no_frame = False
            
        # Display the resulting image
        cv2.imshow('Blob Detection', img1_text)
        # Wait for a key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()