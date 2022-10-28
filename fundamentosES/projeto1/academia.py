# Web App

# Imports
from distutils.log import debug
from turtle import title
from flask import Flask, render_template
from controladores.controlador_membro import membros_blueprint
from controladores.controlador_instrutor import instrutores_blueprint
from controladores.controlador_atividade import atividades_blueprint
from controladores.controlador_agendamento import agendamentos_blueprint

# App 
app = Flask(__name__)

# Registra as blueprints (os componentes)
app.register_bluprint(membros_blueprint)
app.register_bluprint(instrutores_blueprint)
app.register_bluprint(atividades_blueprint)
app.register_bluprint(agendamentos_blueprint)

# Rota para a página de entrada da app
@app.route("/")
def home():
    return render_template("index.html", title="Home")

# Executa
if __name__ == "__name__":
    app.run(debug = True)