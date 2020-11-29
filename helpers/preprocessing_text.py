import unidecode



# Esta função foi retirada integralmente do site:
# https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
# Ela é útil pois muitos dos reviews contém emojis
import pandas as pd
import re

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# Formatar palavra/expresão para inserção no banco de dados. Assim facilitamos
# na hora de realizar comparações de strings. Ideal para PT-BR, mas pode ser
# aplicado em outros casos de línguas.
# Exemplo: 
# input: BOtão difÍcil de LoCaliZar
# output: botao dificil de localizar
def formatar_texto(palavra):
    return unidecode.unidecode(palavra).lower()

def format_string(string_raw):
    string_treatment_1 = remove_emoji(string_raw)
    print(string_treatment_1)
    string_treatment_2 = formatar_texto(string_treatment_1)
    print(string_treatment_2)
    return string_treatment_2


string = "Botão difícil de Localizar So much it's so nice to play \U0001F5FF \U0001F6FF \U0001F1FF \U000027B0 \U0001F251"

print(format_string(string))