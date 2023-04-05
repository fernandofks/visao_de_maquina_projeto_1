import numpy as np
import cv2

import matplotlib.pyplot as plt
from math import *
from funcao_filtro import filtro

flag0 = cv2.IMREAD_GRAYSCALE
#Load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag
img_gray = cv2.imread("img/Fig_OK_01.jpg", flag0) #or simply (path, 0)
#Ler forma da figura (altura e largura)
(h, w) = img_gray.shape

cv2.imshow('graycsale image',img_gray)
cv2.waitKey(0)

circulo=np.where(img_gray>150,255,0).astype(np.uint8)