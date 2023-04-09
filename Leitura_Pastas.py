import cv2
import os
 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd # para desenhar/plotar as tabelas informando os status das peças em cada teste

from filtros_extras import *

#importar imagens de todas as pastas
#rodar todos os testes com todas as imagens
#ordem dos testes: Borda, Superfície, Diâmetro e Status, Relação A/B (raios de elipse) e Status
#imprimir os resultados dos testes em uma planilha gerada pelo pandas (excel?)

#Parte 1: Importando e lendo todos os arquivos em uma pasta:

path = "OK"
dir_list = os.listdir(path)

#tamanho da esteira em mm (usado para obter a conversão px para mm e assim calcular o diâmetro da peça)
#o tamanho da esteira foi fornecido pelo Dinho no arquivo de rubricas disponível no BB
#nas imagens a largura da esteira é de 65mm
#nos vídeos a largura da esteira é de 75.6mm
largura_da_esteira = 65 # mm  

f = plt.figure(figsize=(10,5))
for i in range(0,len(dir_list)):
    #encontrar uma maneira mais bonita de fazer a soma das strings abaixo
    img_in = cv2.imread(path + "/" + str(dir_list[i]), cv2.IMREAD_GRAYSCALE)
    
    #O filtro de nitidez especial está apresentando erros, provavelmente um overshoot do range (0,255), verificar se há astype("int32") 
    #img_in = grayscale_especial(img_in)
    
    (height,width) = img_in.shape
    
    if img_in is None:
        print("File not found. Bye!")
        exit(0)
        
    #plotar a imagem em um gráfico
    ax = f.add_subplot (3,5,i+1)
    plt.imshow(img_in, cmap='gray')
    ##### Parte 1 resultados: Funcionando OK até aqui
    
    ##### Parte 2: Rodar um dos testes com a imagem (teste escolhido: Contorno externo)
    
    returns,thresh=cv2.threshold(img_in,97,255,cv2.THRESH_BINARY)
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    img1_text = cv2.cvtColor(img_in,cv2.COLOR_GRAY2RGB)
    
    if len(contours) != 0:
        for j in range(len(contours)):
            if len(contours[j]) >= 1000:
                # Draw the contour on the original image in green color
                cv2.drawContours(img1_text,[contours[j]],-1,(0,255,0),2)
                
                # Draw a circle at the centroid
                #cv2.circle(img_in, (cx, cy), 5, (0, 0, 255), -1)
                
                # Compute the bounding box of the contour
                x,y,w,h = cv2.boundingRect(contours[j])

                # Draw the bounding box on the original image in blue color
                #cv2.rectangle(img_in,(x,y),(x+w,y+h),(255,0,0),2)
                convex = cv2.isContourConvex(contours[j])
                
                # o trecho abaixo é provisório e vai ser substituido por um append no pandas
                if convex:
                    print (i, "is convex")
                else:
                    print (i, "is not convex")
                    
            else:
                # optional to "delete" the small contours
                cv2.drawContours(thresh,contours,-1,(0,0,0),-1)
                
    cv2.imshow('Convexity Detection', img1_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
plt.show()


#Parte 1(Funcionando Ok)
#Parte 2(Gerando um contorno extra por conta de um "blob" no topo da tela) 