import sqlite3
from datetime import datetime

BANCO = "cromo_memoria.db"


def conectar():
    return sqlite3.connect(BANCO)


def inicializar_memoria():

    with conectar() as conexao:

        conexao.execute("""
            CREATE TABLE IF NOT EXISTS conversas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                data_hora TEXT NOT NULL
            )
        """)


def salvar_mensagem(role, content):

    with conectar() as conexao:

        conexao.execute("""
            INSERT INTO conversas (role, content, data_hora)
            VALUES (?, ?, ?)
        """, (
            role,
            content,
            datetime.now().isoformat(timespec="seconds")
        ))


def carregar_historico(limite=12):

    with conectar() as conexao:

        cursor = conexao.execute("""
            SELECT role, content
            FROM conversas
            ORDER BY id DESC
            LIMIT ?
        """, (limite,))

        mensagens = cursor.fetchall()

    mensagens.reverse()

    return [
        {"role": role, "content": content}
        for role, content in mensagens
    ]


def limpar_memoria():

    with conectar() as conexao:

        conexao.execute("DELETE FROM conversas")


inicializar_memoria()
