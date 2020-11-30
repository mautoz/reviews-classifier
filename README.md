![EACH-USP](./img/each.png)

# ACH2018 - PSGII

Este repositório faz parte do projeto de ACH2018,  funcionamdo em conjunto com o [review-automation-scripts](https://github.com/mautoz/reviews-automation-scripts) e [python-webhook](https://github.com/mautoz/python-webhook).
Aqui estão os códigos utilizados para o machine learing.

## Requisitos

- Postgres >=10.14
- Python 3.7

Bibliotecas
- Scikit-learn 
- NLTK
- WordCloud
- psycopg2 2.8.6
- Unicode 1.1.1 (Se for usar o pip, existe unicode e Unicode, escolha o último)

## Configurações do PC

- SO: Ubuntu 18.04.5 LTS
- Memória: 8 GB
- Processador: Intel i5

## Instruções

1. A primeira coisa a fazer é criar um 'database' chamado 'reviews-classifier'. Então criar as tabelas do [reviews_data.sql](./sql/reviews_data.sql).

2. É possível começar o treino de máquina com os reviews do Kaggle. No [README](./csv/README.md) da pasta csv é possível encontrar a fonte dos três bancos de dados usado. É só baixar e colocar os arquivos nas pastas correspondentes. São arquivos grandes, por essa razão, a inserção pode demorar bastante.

3. 