# visao_de_maquina_projeto_1

Objetiva-se com esse projeto: 
A classificação da qualidade de borrachas de vedação, com o uso de conhecimentos desenvolvidos ao longo da disciplina de visão de máquina e do curso

## Os arquivos para correção do projeto são:

### Arquivo_final.ipynb
Para correção dos dois vídeos: alterar o nome do vídeo na linha 2 da segunda célula do arquivo
Nesse arquivo importa-se o vídeo com as peças passando sobre a esteira e as peças são classificadas de acordo com o defito que possuem
O resultado então é expotado em um arquivo .csv que pode ser aberto pelo bloco de notas ou excel

### Resultados.csv
Esse é o arquivo em .csv gerado pelo Arquivo_final.ipynb

## Arquivos adicionais para a execução:

### fillHoles.py
Arquivo disponibilizado pelo professor para preencher buracos nas imagens geradas pelo comando 
cv2.threshold

### borda_ML.ipynb
Arquivo utilizado para gerar modelo de ML que classifica a borda das peças nas imagens como OK ou NOK
gerando um modelo salvo no arquivo borda_ML.h5

### borda_video_ML.ipynb
Arquivo utilizado para gerar modelo de ML que classifica a borda das peças nos vídeos como OK ou NOK
gerando um modelo salvo no arquivo borda_video_ML.h5

### superficie.ipynb
Arquivo utilizado para gerar modelo de ML que classifica a superfície das peças nas imagens como OK ou NOK gerando um modelo salvo no arquivo superficie_ML.h5

### superficie_video.ipynb
Arquivo utilizado para gerar modelo de ML que classifica a superfície das peças nos vídeos como OK ou NOK gerando um modelo salvo nos arquivos superficie_video.h5 e superficie_video_ML.h5

### filtros_extras.py (obsoleto)
Arquivo de funções utilizado principalmente nos protótipos para realização de testes antes da descoberta de funções nativas ao próprio OpenCV, o seu uso se tornou obsoleto nas versões atuais



