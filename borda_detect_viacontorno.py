import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd # para desenhar/plotar as tabelas informando os status das peças em cada teste
import cv2

from filtros_extras import *

img_in=cv2.imread("NOK_borda\Fig_NOK_12.jpg",cv2.IMREAD_COLOR)
if img_in is None:
    print("File not found. Bye!")
    exit(0) #Essa linha "crasha" o notebook, caso ocorra reinicar o kernel
    
#gray=cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
gray = grayscale_especial (img_in,0.02,0.21,0.77)
gray_nit = filtro_nitidez(gray)

returns,thresh=cv2.threshold(gray,97,255,cv2.THRESH_BINARY)

contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    for i in range(len(contours)):
        if len(contours[i]) >= 800:
            # Draw the contour on the original image in green color
            cv2.drawContours(img_in,[contours[i]],-1,(0,255,0),2)
            
            # Draw a circle at the centroid
            #cv2.circle(img_in, (cx, cy), 5, (0, 0, 255), -1)
            
            # Compute the bounding box of the contour
            x,y,w,h = cv2.boundingRect(contours[i])

            # Draw the bounding box on the original image in blue color
            cv2.rectangle(img_in,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow('Convexity Detection', img_in)
cv2.waitKey(0)
cv2.destroyAllWindows()