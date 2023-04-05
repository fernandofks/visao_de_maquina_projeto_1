import numpy as np
import cv2

import matplotlib.pyplot as plt
from math import *
from funcao_filtro import filtro


img_in = cv2.imread("visao_de_maquina_projeto_1\img 2\Fig_NOK_01.jpg", cv2.IMREAD_GRAYSCALE)
if img_in is None:
    print("File not found. Bye!")
    exit(0)


img_bin = np.where(img_in >97, 255, 0).astype("uint8")

cv2.imshow('graycsale image',img_in)
cv2.waitKey(0)

aro_exterior = filtro(img_bin,True,True,False,True,False,Color = 0,Area_min = 10000,Area_max = 200000,Convexity_max=1)
img1_text = cv2.cvtColor(img_in,cv2.COLOR_GRAY2RGB)#pneus

i=1
for KPi in aro_exterior:
    print("Blob_", i, ": X= ", KPi.pt[0], " Y= ", KPi.pt[1], " size=", KPi.size**2, " ang=", KPi.angle)
    img1_text = cv2.putText(img1_text, str(i), (int(KPi.pt[0]),int(KPi.pt[1])), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)
    i=i+1

cv2.imshow('graycsale image',img1_text)
cv2.waitKey(0)

img1_with_KPs = cv2.drawKeypoints(img_in, aro_exterior, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('graycsale image',img1_with_KPs)
cv2.waitKey(0)

plt.imshow(img_bin, cmap='gray')
plt.show()