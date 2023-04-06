import cv2
import numpy as np

def grayscale_especial (fig_in,const_red=1.0,const_green=1.0,const_blue=1.0):
    [B,G,R] = cv2.split(fig_in)

    (height,width) = R.shape
    print ("Height = ", height)
    print ("Width = ", width)

    R_bin = np.zeros((height,width), dtype = 'uint8')
    G_bin = np.zeros((height,width), dtype = 'uint8')
    B_bin = np.zeros((height,width), dtype = 'uint8')

    for i in range (height-1):
        for j in range (width-1):
            R_bin[i,j] = R[i,j]*const_red
            G_bin[i,j] = G[i,j]*const_green
            B_bin[i,j] = B[i,j]*const_blue

    Merged = R_bin+G_bin+B_bin
    return Merged