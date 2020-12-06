![EACH-USP](./img/each.png)

# ACH2018 - PSGII

Este repositório faz parte do projeto de ACH2018, funcionando em conjunto com o [review-automation-scripts](https://github.com/mautoz/reviews-automation-scripts) e [python-webhook](https://github.com/mautoz/python-webhook).
Aqui estão os códigos utilizados para o machine learing, também estão as funções que criaram alguns dos gráficos usados.

## Requisitos

- Postgres >= 10.14
- Python >= 3.7

Bibliotecas
- Scikit-learn 
- NLTK
- WordCloud
- psycopg2 == 2.8.6
- Unicode == 1.1.1 (Se for usar o pip, existe unicode e Unicode, escolha o último)

## Configurações do PC

- SO: Ubuntu 18.04.5 LTS
- Memória: 8 GB
- Processador: Intel i5

## Instruções

1. A primeira coisa a fazer é criar um 'database' chamado 'reviews-classifier'. Então criar as tabelas do [reviews_data.sql](./sql/reviews_data.sql).

2. É possível começar o treino de máquina com os reviews do Kaggle. No [README](./csv/README.md) da pasta csv estão as fonte dos três bancos de dados usado. É só baixar e colocar os arquivos nas pastas correspondentes. São arquivos grandes, por essa razão, a inserção pode demorar bastante.

3. Se você for utilizar os datasets do item 2, então execute os códigos: google_apps_reader.py, python3 google_apps_extended_reader.py, python3 android_app_dataset_reader.py. É só "descomentar" no [run](./run.sh).

4. Ainda no [run](./run.sh), preencha as variáveis do Postgres com as informações da sua máquina!

5. Com o dataset já carregados no bd, agora é só executar o unsupervised.py ou supervised.py, para obter os resultados. Execute-os, assim como os códigos do item 3, pelo [run](./run.sh), que carregará as variáveis. Exemplo:
```
$ bash run
```
ou
```
$ chmod +x run
$ ./run
```

Observação: é importante destacar que independente de utilizar ou não os dataset do Kaggle, os dados do bd estarão "crus", isto é, não terão sido avaliados por humanos! É preciso utilizar o [python-webhook](https://github.com/mautoz/python-webhook) para classificar manualmente para fornecer dados para o treino da máquina!
Caso queira testar com alguns dados já classificados, mais especificamente com os que serviram de base para a monografia, baixe o dump da seção abaixo e utilize o comando de carregar no postgres, dentro do database 'reviews-classifier'. Deve ficar mais ou menos assim:
```
psql -h hostname -d databasename -U username -f reviews_classifier_dump.sql
```

## DUMP do banco de dados

Após algum tempo, a classificação manual dos reviews foi encerrada e então foram feitos o cálculos apresentados na monografia. Para que os testes possam ser repetidos, foi feito um dump do db. O link está abaixo.

[DUMP_reviews_data](https://drive.google.com/file/d/1F93ycz0607EPHUfGM3PwSn-izSD76Cll/view?usp=sharing)