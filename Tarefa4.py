import numpy as np
import cv2

import matplotlib.pyplot as plt
from math import *


img_in = cv2.imread("visao_de_maquina_projeto_1\img\Fig_OK_01.jpg", cv2.IMREAD_COLOR)
img_bin = np.where(img_in>80, 255, 0).astype("uint8")

cv2.imshow('graycsale image',img_bin)
cv2.waitKey(0)