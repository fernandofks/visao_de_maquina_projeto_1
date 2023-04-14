import cv2
import os
 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd # para desenhar/plotar as tabelas informando os status das peças em cada teste

from filtros_extras import *
from fillHoles import fillHoles
from random import randint # used to generate borders to off-limits cropped images

from tensorflow import keras

# Load the video file
cap = cv2.VideoCapture('Video2_Vedacao.mp4')

#Flag that checks if the rubber in the image is new or not
first_appearance = True

cont_pecas = 0 
conjunto_NOK_video=[]
x_des,y_des = (int(508),int(486))

df = pd.DataFrame()
#test_result lists, for every rubber the test result will be appended here and they will be added as a dataSeries to a dataFrame
Numero_da_peca    = []
Convexidade_valor = [] 
Teste_borda       = []
Teste_superficie  = []
Diametro_Peca     = []
Status_Diametro   = []
Relacao_AB        = []
Status_Raio_AB    = []


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
        img1_text_R = R
        if len(contours) != 0:
            perimetro_max = -1
            contorno_out=max(contours, key=len)
            if len(contorno_out)>=5: #necessita de no mínimo 5 para fazer fit ellipse
                ellipse = cv2.fitEllipse(contorno_out)
                (x, y), (MA, ma),angle =  ellipse

                area_contorno = cv2.contourArea(contorno_out)
                
                
                #print (area_contorno)
                
                #runs the tests and lowers the first_appearance flag    
                if y > 200 and y< 300 and first_appearance and area_contorno>30000:
                    
                    #generating cropped images in order to make the model work
                    #documentation used for this part: https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
                    #documentation used for this part: https://stackoverflow.com/questions/55733086/opencv-how-to-overcrop-an-image
                    src = img1_text_R
                    borderType = cv2.BORDER_REPLICATE
                    boarderSize = .1
                    top = int(boarderSize * src.shape[0])  # shape[0] = rows
                    bottom = int(boarderSize * src.shape[0])
                    left = int(boarderSize * src.shape[1])  # shape[1] = cols
                    right = left    
                    value = [randint(0,255), randint(0,255), randint(0,255)]
                    dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
                    
                    #actually cropping the image (before making border it actually works)
                    #cv2.rectangle(src, (int(x - x_des/2), int(y - y_des/2)), (int(x + x_des/2),int(y + y_des/2)), (255, 0, 0), 3)
                    #cropped_image = src[int(x - x_des/2):int(x + x_des/2), int(y - y_des/2):int(y + y_des/2)]
                    
                    #actually actually cropping the image
                    #dst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
                    border_height = dst.shape[0]
                    border_width = dst.shape[1]
                    x_offset = 0
                    y_offset = -70
                    x_min_border = int((border_width - x_des)/2) + x_offset
                    y_min_border = int((border_height - y_des)/2) + y_offset
                    x_max_border = int((border_width + x_des)/2) + x_offset
                    y_max_border = int((border_height  + y_des)/2) + y_offset
                    
                    #x_min_border = int(left + x_offset)
                    #y_min_border = int(x_min_border + x_des)
                    #x_max_border = int(left + x_des + x_offset)
                    #y_max_border = int(y_min_border +y_des)
                    
                    dst_rect = cv2.rectangle(dst, (x_min_border, y_min_border), (x_max_border, y_max_border), (255, 0, 0), 3)
                    #cv2.rectangle(dst, (int((border_width - x_des)/2), int((border_height - y_des)/2)+20), (int((border_width + x_des)/2),int((border_height + y_des)/2)+20), (255, 0, 0), 3)
                    
                    #o comando de crop image funciona na base de (y,x) e não (x,y) e se passaram 30 min até eu descobrir isso
                    #documentação dessa merda: https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
                    cropped_image = dst[y_min_border: y_max_border, x_min_border: x_max_border]
                    cv2.imshow("cropped", cropped_image)
                    # Display cropped image
                    #cv2.imshow("cropped", cropped_image)
                    cv2.imwrite('imagens_classificar/'+str(cont_pecas)+'_2.jpg', cropped_image)
                    cont_pecas += 1
                    #print ([cont_pecas], y)
                        
                    first_appearance = False
                    
                    ax = f.add_subplot (4,6,cont_pecas)
                    #plt.imshow(B, cmap='gray')
                    #plt.imshow(thresh, cmap='gray')
                    #plt.imshow(img1_text, cmap='gray')
                    plt.imshow(dst_rect, cmap='gray')
                    conjunto_NOK_video.append(img1_text_R)
                    
                    #### Teste de contornos
                    
                    cv2.drawContours(img1_text,contorno_out,-1,(0,0,255),4)
                    hull = cv2.convexHull(contorno_out)
                    cv2.drawContours(img1_text,hull,-1,(0,255,0),8)
                    #cv2.circle(img1_text, (cX, cY), 7, (255, 255, 255), -1)
            
                    area_contorno = cv2.contourArea(contorno_out)
                    area_hull = cv2.contourArea(hull)
                    if area_hull >= 1: #evitar que o video crashe por conta de divisão por 0
                        convexidade = area_contorno/area_hull
                    #print ([cont_pecas], y, convexidade)
                    
                    Numero_da_peca.append(cont_pecas)
                    Convexidade_valor.append(convexidade)
                    #Teste_borda
                    
                
                #raises the first_appearance flag
                if area_contorno<30000 and y>400:
                    first_appearance = True

                     
            M = cv2.moments(contorno_out)
            
            if M["m00"] >=1: #evitar que o video crashe por conta de divisão por 0
                cX = (M["m10"] / M["m00"])
                cY = (M["m01"] / M["m00"])

            cv2.drawContours(img1_text,contorno_out,-1,(0,0,255),4)
            hull = cv2.convexHull(contorno_out)
            cv2.drawContours(img1_text,hull,-1,(0,255,0),8)
            cv2.circle(img1_text, (int(cX), int(cY)), 7, (255, 255, 255), -1)
            
            #area_contorno = cv2.contourArea(contorno_out)
            #area_hull = cv2.contourArea(hull)
            #if area_hull >= 1: #evitar que o video crashe por conta de divisão por 0
            #    convexidade = area_contorno/area_hull
        
        #also, if there are no countours detected in the image it raises the first appearance flag
        else:
            first_appearance = True
            
        # Display the resulting image
        cv2.imshow('Blob Detection', img1_text)
        # Wait for a key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
conjunto_NOK_video=np.stack(conjunto_NOK_video, axis=0)
# Release the video capture object and close all windows
#model=keras.models.load_model('./superficie.h5')
#print(model.predict(conjunto_NOK_video))
cap.release()
cv2.destroyAllWindows()

plt.show()

df["Numero da Peca"]     = Numero_da_peca
df["Convexidade medida"] = Convexidade_valor

print(df)