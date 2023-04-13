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

#path = "OK"
path = "NOK_superficie"

dir_list = os.listdir(path)


f = plt.figure(figsize=(10,5))
for i in range(0,len(dir_list)):
    #encontrar uma maneira mais bonita de fazer a soma das strings abaixo
    img_in = cv2.imread(path + "/" + str(dir_list[i]), cv2.IMREAD_COLOR)
    if img_in is None:
        print("File not found. Bye!")
        exit(0)
        
    #Separando os canais de cor da imagem original
    [B,G,R] = cv2.split(img_in)
    (height,width) = R.shape
        
    edges = cv2.Canny(img_in,100,200) #edges não é um bom filtro pra essa aplicação
    #B = CorrecaoGamma(B,.5)
    #laplacian = cv2.Laplacian(img_in,cv2.CV_8U)

    ax = f.add_subplot (3,5,i+1)
    #plt.imshow(B, cmap='gray')
    #plt.imshow(thresh, cmap='gray')
    plt.imshow(edges, cmap='gray')
plt.show()

