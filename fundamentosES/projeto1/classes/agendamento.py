# Classe Agendamento
class Agendamento:

    # Método para inicialização dos atributos
    def __init__(self, atividade, membro, id = None):
        self.atividade = atividade
        self.membro = membro
        self.id = id