import cv2
import os
 
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd # para desenhar/plotar as tabelas informando os status das peças em cada teste

from filtros_extras import *
from fillHoles import fillHoles

#importar imagens de todas as pastas
#rodar todos os testes com todas as imagens
#ordem dos testes: Borda, Superfície, Diâmetro e Status, Relação A/B (raios de elipse) e Status
#imprimir os resultados dos testes em uma planilha gerada pelo pandas (excel?)

#Parte 1: Importando e lendo todos os arquivos em uma pasta:

#path = "OK"
path = "NOK_borda"

dir_list = os.listdir(path)
print(dir_list)
#tamanho da esteira em mm (usado para obter a conversão px para mm e assim calcular o diâmetro da peça)
#o tamanho da esteira foi fornecido pelo Dinho no arquivo de rubricas disponível no BB
#nas imagens a largura da esteira é de 65mm
#nos vídeos a largura da esteira é de 75.6mm

largura_da_esteira = 65 # mm
diametro_ideal_mm = 50



#Criando um dataframe para registrar e imprimir os resultados dos testes (e possivelmente exportar isso para uma planilha do Excel)
df = pd.DataFrame()
  
Nome_arquivo     = []
Convexidade_valor= [] 
Teste_borda      = []
Teste_superficie = []
Diametro_Peca    = []
Status_Diametro  = []
Relacao_AB       = []
Status_Raio_AB   = []

 
f = plt.figure(figsize=(10,5))
for i in range(0,len(dir_list)):
    #encontrar uma maneira mais bonita de fazer a soma das strings abaixo
    contours = []
    img_in = cv2.imread(path + "/" + str(dir_list[i]), cv2.IMREAD_COLOR)
    if img_in is None:
        print("File not found. Bye!")
        exit(0)
    
    #Separando os canais de cor da imagem original
    [B,G,R] = cv2.split(img_in)
    (height,width) = B.shape
            
    Nome_arquivo.append(str(dir_list[i]))
    
    returns,thresh=cv2.threshold(B,90,255,cv2.THRESH_BINARY_INV)
    #thresh =  close (thresh)
    thresh = fillHoles(thresh)
    
    # Convexity is calculated by the ratio of blob area/blob's convex hull area
     
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img1_text = cv2.cvtColor(B,cv2.COLOR_GRAY2RGB)
    
    perimetro_max = -1
    contorno_out = contours[0]
    
    # Find the max perimeter and plot its contour:
    for j in range(len(contours)):
        perimetro = cv2.arcLength(contours[j],True)
        if perimetro > perimetro_max:
            perimetro_max = perimetro
            contorno_out = contours[j]
            
    # Get the convex hull of the shape and plot its contour:
    hull = cv2.convexHull(contorno_out)

    #find the radius of contour
    if len(contorno_out)>=5: #necessita de no mínimo 5 para fazer fit ellipse
        ellipse = cv2.fitEllipse(contorno_out)
        (x, y), (MA, ma),angle =  ellipse
                
    M = cv2.moments(contorno_out)
                
    cX = (M["m10"] / M["m00"])
    cY = (M["m01"] / M["m00"])
    
    
    
    diametro_ideal_px = diametro_ideal_mm*(width/largura_da_esteira)
    raio_ideal_px = int(diametro_ideal_px/2)
    
    area_contorno = cv2.contourArea(contorno_out)
    area_hull = cv2.contourArea(hull)
    area_circulo = int(pi*pow((MA+ma)/4,2))
    area_circulo_ideal = int(pi*pow(raio_ideal_px,2))
    
    convexidade = area_contorno/area_circulo_ideal
    
    
    #cv2.circle(img1_text, (int(cX), int(cY)), int((MA+ma)/4), (255,0, 0), 8)
    cv2.circle(img1_text, (int(cX), int(cY)), raio_ideal_px, (255,0, 0), 8)
    cv2.drawContours(img1_text,contorno_out,-1,(0,0,255),4)
    cv2.drawContours(img1_text,hull,-1,(0,255,0),8)
    

    
    if convexidade < 0.95:
        Teste_borda.append("Reprovado")
    else:
        Teste_borda.append("Aprovado")
            
    #print ("Perimetro",[i], perimetro_max)
    print ("Area_Circulo_fit", [i], area_circulo)
    print ("Area_Circulo_ideal", [i], area_circulo_ideal)
    print ("Area_Contorno", [i], area_contorno)
    print ("Area_Hull", [i], area_hull ,"\n")
    
    
    print ("Convexidade",[i], convexidade ,"\n")
    Convexidade_valor.append(convexidade)

    ax = f.add_subplot (3,5,i+1)
    #plt.imshow(B, cmap='gray')
    plt.imshow(thresh, cmap='gray')
    #plt.imshow(img1_text, cmap='gray')
                
plt.show()

df["Nome do Arquivo"] = Nome_arquivo
df["Convexidade medida"] = Convexidade_valor
df["Teste de borda"] = Teste_borda
print(df)
