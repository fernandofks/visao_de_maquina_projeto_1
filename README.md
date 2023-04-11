# visao_de_maquina_projeto_1

Problemas encontrados a serem consertados:

- Contornos:
As funções de detecção de contorno do openCV funcionam, no entanto a binarização das imagens que é utilizada para traçar o contorno 
não é adequada para essas funções, o código roda e em alguns casos/ocasioes até acerta os resultados mas ainda é insuficiente, isso
ocorre por um simples motivo, dentro da peça de borracha tem-se multiplas intensidades de cinza, para obter melhores resultados deve-se
aplicar um filtro que mantenha os contornos mas que "borre" os preenchimentos (Qual filtro faz isso? Gaussiano? Mediana?)

-Reflexão da luz na borracha:
Como resolver? Ajuste de brilho por gama? Falar com o Dinho

Filtro de Nitidez:
Ao aplicar o filtro de nitidez e pedir pras funções rodarem com a imagem nítida como input há um erro por conta da "profundidade da imagem"

Classificação da superfície:
Há a necessidade de destacar os contornos e ignorar os preenchimentos?
Feita com o uso de filtro derivada?
