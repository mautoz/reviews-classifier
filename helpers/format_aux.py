import unidecode

# Formatar palavra/expresão para inserção no banco de dados. Assim facilitamos
# na hora de realizar comparações de strings.
# Exemplo: 
# input: BOtão difÍcil de LoCaliZar
# output: botao dificil de localizar
def formatar_texto(palavra):
    return unidecode.unidecode(palavra).lower()


# print(formatar_texto("Botão difícil de Localizar"))
