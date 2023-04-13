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

path = "OK"
#path = "NOK_borda"

dir_list = os.listdir(path)

#tamanho da esteira em mm (usado para obter a conversão px para mm e assim calcular o diâmetro da peça)
#o tamanho da esteira foi fornecido pelo Dinho no arquivo de rubricas disponível no BB
#nas imagens a largura da esteira é de 65mm
#nos vídeos a largura da esteira é de 75.6mm
largura_da_esteira = 65 # mm

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
    
    #testes de filtro para melhorar output
    #B = CorrecaoGamma(B,1)
    #B = cv2.bilateralFilter(B,4,130,75)
    
    #O filtro de grayscale especial está apresentando erros, provavelmente um overshoot do range (0,255), verificar se há astype("int32") 
    #img_in = grayscale_especial(img_in,const_red=0.00,const_green=1.00,const_blue=0.00)
            
    Nome_arquivo.append(str(dir_list[i]))
    
    #plotar a imagem em um gráfico
    #ax = f.add_subplot (3,5,i+1)
    #plt.imshow(B, cmap='gray')
    ##### Parte 1 resultados: Funcionando OK até aqui
    
    ##### Parte 2: Rodar um dos testes com a imagem (teste escolhido: Contorno externo)
    returns,thresh=cv2.threshold(B,70,255,cv2.THRESH_BINARY_INV)
    #thresh =  close (thresh)
    thresh = fillHoles(thresh)
    
    #usado para mostrar os contornos binarizados
    #ax = f.add_subplot (3,5,i+1)
    #plt.imshow(thresh, cmap='gray')

    #convexity is calculated by the ratio of blob area/blob's convex hull area 
    # Get the convex hull of the contour
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img1_text = cv2.cvtColor(B,cv2.COLOR_GRAY2RGB)
    
    #print("numero_contornos", len(contours))
    
    perimetro_max = -1
    contorno_out = contours[0]
    
    for j in range(len(contours)):
        #achar o perimetro perimetro_maximo para gerar esse contorno
        perimetro = cv2.arcLength(contours[j],True)
        if perimetro > perimetro_max:
            perimetro_max = perimetro
            contorno_out = contours[j]
    
        #lista_perimetro.append(Perimetro)
        #Substituir por perimetro de contorno ao invés de comprimento da lista de contornos
    
    #if perimetro >= perimetro_max:  
    cv2.drawContours(img1_text,contorno_out,-1,(0,0,255),4)
    hull = cv2.convexHull(contorno_out)
    cv2.drawContours(img1_text,hull,-1,(0,255,0),8)
    
    area_contorno = cv2.contourArea(contorno_out)
    area_hull = cv2.contourArea(hull)
    convexidade = area_contorno/area_hull
    
    if convexidade < 0.9812:
        Teste_borda.append("Reprovado")
    else:
        Teste_borda.append("Aprovado")
            
    print ("Perimetro",[i], perimetro_max)
    print ("Area_Contorno", [i], area_contorno)
    print ("Area_Hull", [i], area_hull)
    print ("Convexidade",[i], convexidade ,"\n")
    Convexidade_valor.append(convexidade)
            
    #aro_exterior = filtro_blobs(G,True,True,False,True,False,Color = 0,Area_min=10000,Area_max=200000,Convexity_min = 0.95,Convexity_max=1)
    #img_text = img_in
    
    #j=1
    #for KPi in aro_exterior:
        #print("Blob_", i, ": X= ", KPi.pt[0], " Y= ", KPi.pt[1], " size=", KPi.size**2, " ang=", KPi.angle)
        #img_text = cv2.putText(img_text, str(i), (int(KPi.pt[0]),int(KPi.pt[1])), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)
        #j=j+1

    #img1_with_KPs = cv2.drawKeypoints(img_text, aro_exterior, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ax = f.add_subplot (3,5,i+1)
    #plt.imshow(B, cmap='gray')
    #plt.imshow(thresh, cmap='gray')
    plt.imshow(img1_text, cmap='gray')
                
    #cv2.imshow('Convexity Detection', img1_text)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

plt.show()

df["Nome do Arquivo"] = Nome_arquivo
df["Convexidade medida"] = Convexidade_valor
df["Teste de borda"] = Teste_borda
print(df)

#Parte 1(Funcionando Ok)
