import numpy as np
import cv2

import matplotlib.pyplot as plt
from math import *
from filtros_extras import *


img_in = cv2.imread("NOK_borda\Fig_NOK_15.jpg", cv2.IMREAD_COLOR)
if img_in is None:
    print("File not found. Bye!")
    exit(0)

gray = grayscale_especial(img_in,0.0,0.13,0.87)
gray_nit = filtro_nitidez(gray)

plt.imshow(gray, cmap='gray')
plt.show()
plt.imshow(gray_nit, cmap='gray')
plt.show()

img_bin = np.where(gray_nit >97, 255, 0).astype("uint8")

cv2.imshow('grayscale image',gray)
cv2.waitKey(0)

aro_exterior = filtro_blobs(img_bin,True,True,False,True,False,Color = 0,Area_min = 10000,Area_max = 200000,Convexity_max=0.99)
img1_text = img_in

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

