#! /bin/bash

# Preencha os dados abaixo com as especificações do seu postgres.
# O padrão pode ser:
# export POSTGRES_HOST=localhost
# export POSTGRES_DATABASE=reviews-classifier
# export POSTGRES_USER=postgres
export POSTGRES_HOST=
export POSTGRES_DATABASE=
export POSTGRES_USER=
export POSTGRES_PASSWORD=
# Padrão pode ser:
# export POSTGRES_PORT=5432
export POSTGRES_PORT=5432

# Três datasets do Kaggle que podem ser usados para o treinamento da máquina.
# Basta retirar os comentários.
# python3 google_apps_reader.py
# python3 google_apps_extended_reader.py
# python3 android_app_dataset_reader.py

python3 supervised.py

# python3 unsupervised.py

unset postgres_host
unset postgres_db
unset postgres_user
unset postgres_password
unset postgres_port
