#! /bin/bash

export POSTGRES_HOST=localhost
export POSTGRES_DATABASE=reviews-classifier
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=versatushpc
export POSTGRES_PORT=5432

python3 android_app_dataset_reader.py

# Já inserido
# python3 google_apps_reader.py
# python3 google_apps_extended_reader.py

unset postgres_host
unset postgres_db
unset postgres_user
unset postgres_password
unset postgres_port
