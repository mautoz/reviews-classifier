import unidecode

# Formatar palavra/expresão para inserção no banco de dados. Assim facilitamos
# na hora de realizar comparações de strings.
# Exemplo: 
# input: BOtão difÍcil de LoCaliZar
# output: botao dificil de localizar
def formatar_palavra_chave(palavra):
    return unidecode.unidecode(palavra).lower()


print(formatar_palavra_chave("Botão difícil de Localizar"))
