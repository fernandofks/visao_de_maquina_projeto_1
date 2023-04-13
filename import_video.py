import cv2
import os
 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd # para desenhar/plotar as tabelas informando os status das peças em cada teste

from filtros_extras import *
from fillHoles import fillHoles


# Load the video file
cap = cv2.VideoCapture('Video1_Vedacao.mp4')
first_appearance = True
cont_pecas = 0 

f = plt.figure(figsize=(20,10))
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
        #thresh = close(thresh)
        thresh = fillHoles(thresh)
        
        contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
        img1_text = cv2.cvtColor(B,cv2.COLOR_GRAY2RGB)
        
        if len(contours) != 0:
            perimetro_max = -1
            contorno_out=max(contours, key=len)
            if len(contorno_out)>=5: #necessita de no mínimo 5 para fazer fit ellipse
                ellipse = cv2.fitEllipse(contorno_out)
                (x, y), (MA, ma),angle =  ellipse

                area_contorno = cv2.contourArea(contorno_out)
                #print (area_contorno)
                    
                if y > 200 and y< 300 and first_appearance and area_contorno>30000:
                    cont_pecas += 1
                    print ([cont_pecas], y)
                        
                    first_appearance = False
                    ax = f.add_subplot (4,6,cont_pecas)
                        #plt.imshow(B, cmap='gray')
                        #plt.imshow(thresh, cmap='gray')
                    plt.imshow(img1_text, cmap='gray')
                if area_contorno<30000 and y>400:
                    first_appearance = True

                     
                #M = cv2.moments(contours[j])
                #
                #if M["m00"] >=1: #evitar que o video crashe por conta de divisão por 0
                #    cX = (M["m10"] / M["m00"])
                #    cY = (M["m01"] / M["m00"])

            cv2.drawContours(img1_text,contorno_out,-1,(0,0,255),4)
            hull = cv2.convexHull(contorno_out)
            cv2.drawContours(img1_text,hull,-1,(0,255,0),8)
            #cv2.circle(img1_text, (cX, cY), 7, (255, 255, 255), -1)
    
            area_contorno = cv2.contourArea(contorno_out)
            area_hull = cv2.contourArea(hull)
            if area_hull >= 1: #evitar que o video crashe por conta de divisão por 0
                convexidade = area_contorno/area_hull
        else:
            first_appearance = True
            
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

plt.show()