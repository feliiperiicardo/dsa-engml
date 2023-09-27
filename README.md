# dsa-engml
Repositório destinado à centralizar os conteúdos práticos (códigos) da Formação Engenheiro de Machine Learning ofertada pela Data Science Academy

## Módulo 1
### Requisitos

- Python 3.8.9
- Docker 4.8.2
- Postgres 14.4
- pgAdmin4 6.12
- Flask 2.2.1

### Ambiente
#### Docker

Crie o container dbapp:
```bash
$ docker run --name dbapp -e POSTGRES_PASSWORD=XXX -d -p 5432:5432 postgres
```

Inicialize o container dbapp:
```bash
$ docker start dbapp
```

#### pgAdmin

## Módulo 2

- [requirements.txt](https://github.com/feliiperiicardo/dsa-engml/blob/main/ML_py_cpp/requirements.txt)