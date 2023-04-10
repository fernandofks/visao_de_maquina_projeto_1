import cv2
import os
 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd # para desenhar/plotar as tabelas informando os status das peças em cada teste

from filtros_extras import *

path = "OK"
#path = "NOK_borda"

dir_list = os.listdir(path)

for i in range(0,len(dir_list)):
    #encontrar uma maneira mais bonita de fazer a soma das strings abaixo
    img_in = cv2.imread(path + "/" + str(dir_list[i]), cv2.IMREAD_COLOR)
    #img_in = cv2.imread("OK\Fig_OK_15.jpg", cv2.IMREAD_COLOR)
    
    if img_in is None:
        print("File not found. Bye!")
        exit(0)
    
    img_in = grayscale_especial(img_in,0.0,0.13,0.87)
    img_in = ajuste_brilho_contraste(img_in, -70, 1)
    #img_in = filtro_nitidez(img_in)
        
    img_bin = np.where(img_in >77, 255, 0).astype("uint8")
    #img_bin = close(img_bin)

    #Quando os parametros de convexidade estão configurados para min = 0.995 e max = 1 Peças com defeitos na borda são todas rejeitadas
    #no entanto peças sem defitos nas bordas também são
    aro_exterior = filtro_blobs(img_bin,True,True,False,True,False,Color = 0,Area_min=10000,Area_max=200000,Convexity_min = 0.99,Convexity_max=1)
    img_text = cv2.cvtColor(img_in,cv2.COLOR_GRAY2BGR)
    
    j=1
    for KPi in aro_exterior:
        print("Blob_", i, ": X= ", KPi.pt[0], " Y= ", KPi.pt[1], " size=", KPi.size**2, " ang=", KPi.angle)
        img_text = cv2.putText(img_text, str(i), (int(KPi.pt[0]),int(KPi.pt[1])), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)
        j=j+1

    img1_with_KPs = cv2.drawKeypoints(img_text, aro_exterior, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #cv2.imshow('output',img_bin)
    cv2.imshow('output',img1_with_KPs)
    cv2.waitKey(0)