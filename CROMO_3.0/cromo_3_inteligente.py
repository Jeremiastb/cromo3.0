# ============================================================
# CROMO 3.0 - ROBO EDUCACIONAL COM INTELIGENCIA ARTIFICIAL
# Machine Learning + Rede Neural MLP + Ollama
# Tkinter + pyttsx3 + SpeechRecognition + SQLite
# Visual: armadura verde, detalhes verdes e luzes/circuitos brancos
# ============================================================

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import math
import json
import urllib.request
import urllib.error
import socket
import time
import warnings

import speech_recognition as sr
import pyttsx3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from memoria import salvar_mensagem, carregar_historico

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1 PASSO: CONFIGURACOES
# ============================================================

NOME_ROBO = "CROMO"
MODELO_OLLAMA = "llama3.2"
URL_OLLAMA = "http://localhost:11434/api/chat"
LIMITE_HISTORICO = 12

CONFIANCA_MINIMA = 0.70
MARGEM_MINIMA = 0.15

# Paleta visual do CROMO
FUNDO = "#0B151B"
FUNDO_CANVAS = "#10242B"
VERDE = "#13B85A"
VERDE_CLARO = "#39E879"
VERDE_ESCURO = "#087A3B"
BRANCO = "#F4FFF9"
BRANCO_LUZ = "#DFFFF0"
PRETO_VISOR = "#071014"
AZUL_STATUS = "#B9FFE0"

fila_comandos = queue.Queue()
fila_interface = queue.Queue()
fila_voz = queue.Queue()

encerrando = threading.Event()
escuta_ativa = threading.Event()
reconhecedor = sr.Recognizer()

try:
    historico_conversa = carregar_historico(LIMITE_HISTORICO)
    if not isinstance(historico_conversa, list):
        historico_conversa = []
except Exception as erro:
    print("Erro ao carregar historico:", erro)
    historico_conversa = []

acao_atual = "parado"
inicio_acao = time.time()


# ============================================================
# 2 PASSO: BASE DE TREINAMENTO
# ============================================================

frases_treinamento = [
    # SAUDACAO
    "oi", "ola", "bom dia", "boa tarde", "boa noite",
    "e ai cromo", "ola robo", "tudo bem", "como vai", "saudacoes",

    # NOME
    "qual e seu nome", "como voce se chama", "diga seu nome",
    "quem e voce", "seu nome", "qual o nome do robo",
    "voce se chama cromo", "apresente seu nome",

    # IDENTIDADE
    "o que voce e", "voce e um robo", "explique quem voce e",
    "fale sobre voce", "qual sua funcao", "voce e inteligente",
    "qual seu objetivo", "o que voce consegue fazer",

    # ACENAR
    "acene", "acene com a mao", "levante a mao", "mande um oi",
    "cumprimente", "faca um cumprimento", "diga ola com a mao",
    "movimente seu braco",

    # DANCAR
    "dance", "vamos dancar", "faca uma danca", "dance para mim",
    "movimente seu corpo", "mostre seus movimentos", "danca", "rebola",

    # GIRAR
    "gire", "gire seu corpo", "faca um giro", "de uma volta",
    "gire para mim", "rotacione", "faca uma rotacao",

    # PULAR
    "pule", "salte", "faca um pulo", "pule para cima",
    "mostre um salto", "salte no ar",

    # ANDAR
    "ande", "caminhe", "faca uma caminhada",
    "movimente suas pernas", "ande para frente", "caminhe um pouco",

    # DESPEDIDA
    "tchau", "ate logo", "ate mais", "adeus", "vou embora",
    "encerrar conversa", "finalizar", "ate a proxima",

    # CONVERSA GERAL
    "explique inteligencia artificial", "o que e inteligencia artificial",
    "o que e uma rede neural", "explique machine learning",
    "o que e um neuronio", "como funciona um algoritmo",
    "fale sobre tecnologia", "explique aprendizado de maquina",
    "como funciona uma inteligencia artificial", "o que sao dados",
    "explique deep learning", "o que e programacao",
    "como funciona um computador", "explique python",
    "o que e um robo inteligente", "como uma rede neural aprende",
    "qual a diferenca entre ia e machine learning",
    "o que e um modelo computacional"
]

intencoes_treinamento = [
    # SAUDACAO - 10
    "saudacao", "saudacao", "saudacao", "saudacao", "saudacao",
    "saudacao", "saudacao", "saudacao", "saudacao", "saudacao",

    # NOME - 8
    "nome", "nome", "nome", "nome", "nome", "nome", "nome", "nome",

    # IDENTIDADE - 8
    "identidade", "identidade", "identidade", "identidade",
    "identidade", "identidade", "identidade", "identidade",

    # ACENAR - 8
    "acenar", "acenar", "acenar", "acenar",
    "acenar", "acenar", "acenar", "acenar",

    # DANCAR - 8
    "dancar", "dancar", "dancar", "dancar",
    "dancar", "dancar", "dancar", "dancar",

    # GIRAR - 7
    "girar", "girar", "girar", "girar", "girar", "girar", "girar",

    # PULAR - 6
    "pular", "pular", "pular", "pular", "pular", "pular",

    # ANDAR - 6
    "andar", "andar", "andar", "andar", "andar", "andar",

    # DESPEDIDA - 8
    "despedida", "despedida", "despedida", "despedida",
    "despedida", "despedida", "despedida", "despedida",

    # CONVERSA - 18
    "conversa", "conversa", "conversa", "conversa", "conversa",
    "conversa", "conversa", "conversa", "conversa", "conversa",
    "conversa", "conversa", "conversa", "conversa", "conversa",
    "conversa", "conversa", "conversa"
]

if len(frases_treinamento) != len(intencoes_treinamento):
    raise ValueError(
        f"Base de treinamento inconsistente: {len(frases_treinamento)} "
        f"frases e {len(intencoes_treinamento)} intencoes."
    )


# ============================================================
# 3 PASSO: TREINAMENTO DA REDE NEURAL
# ============================================================

print("\nTreinando a inteligencia do CROMO...")

modelo = Pipeline([
    (
        "vetorizador",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "rede_neural",
        MLPClassifier(
            hidden_layer_sizes=(40, 20),
            activation="relu",
            solver="adam",
            max_iter=1500,
            random_state=42,
            learning_rate_init=0.002,
            early_stopping=False
        )
    )
])

modelo.fit(frases_treinamento, intencoes_treinamento)

print("Treinamento concluido!")
print("Rede neural pronta para classificar intencoes.\n")


# ============================================================
# 4 PASSO: CLASSIFICACAO DE INTENCOES
# ============================================================

def identificar_intencao(texto):
    probabilidades = modelo.predict_proba([texto])[0]
    classes = modelo.classes_
    indices = probabilidades.argsort()[::-1]

    intencao = classes[indices[0]]
    confianca = float(probabilidades[indices[0]])
    segunda_confianca = float(probabilidades[indices[1]])
    margem = confianca - segunda_confianca

    vetor = modelo.named_steps["vetorizador"].transform([texto])
    palavras_reconhecidas = vetor.nnz

    return intencao, confianca, margem, palavras_reconhecidas


# ============================================================
# 5 PASSO: DETECCAO DE PERGUNTAS
# ============================================================

def eh_pergunta(texto):
    texto = texto.lower().strip()

    palavras_pergunta = [
        "o que", "como", "qual", "quais", "quem", "quando",
        "onde", "porque", "por que", "explique", "defina",
        "fale sobre", "me explique", "me ensine", "para que"
    ]

    if "?" in texto:
        return True

    return any(texto.startswith(palavra) for palavra in palavras_pergunta)


# ============================================================
# 6 PASSO: IA GENERATIVA - OLLAMA
# ============================================================

def conversar_com_ia(pergunta):
    global historico_conversa

    instrucao = (
        "Seu nome e CROMO. Voce e um robo educacional que conversa "
        "em portugues brasileiro. Responda de maneira clara, didatica "
        "e objetiva. Explique conceitos de inteligencia artificial, "
        "programacao, tecnologia, matematica e ciencias. Quando "
        "necessario, apresente exemplos simples. Nao invente "
        "informacoes. Use respostas adequadas para estudantes iniciantes."
    )

    mensagens = [{"role": "system", "content": instrucao}]
    mensagens.extend(historico_conversa)
    mensagens.append({"role": "user", "content": pergunta})

    dados = {
        "model": MODELO_OLLAMA,
        "messages": mensagens,
        "stream": False
    }

    requisicao = urllib.request.Request(
        URL_OLLAMA,
        data=json.dumps(dados).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=120) as resposta_http:
            resultado = json.loads(resposta_http.read().decode("utf-8"))

        texto_resposta = (
            resultado.get("message", {}).get("content", "").strip()
        )

        if not texto_resposta:
            return "O Ollama retornou uma resposta vazia."

        try:
            salvar_mensagem("user", pergunta)
            salvar_mensagem("assistant", texto_resposta)
            historico_novo = carregar_historico(LIMITE_HISTORICO)
            if isinstance(historico_novo, list):
                historico_conversa = historico_novo
        except Exception as erro_memoria:
            print("Erro ao salvar historico:", erro_memoria)

        return texto_resposta

    except urllib.error.HTTPError as erro:
        return (
            f"Erro HTTP do Ollama: {erro.code}. "
            "Verifique o modelo e o servico."
        )
    except (TimeoutError, socket.timeout):
        return "A resposta demorou muito. Tente fazer uma pergunta mais curta."
    except urllib.error.URLError:
        return (
            "Nao consegui acessar o Ollama. Verifique se o servico esta "
            "ativo e se o modelo llama3.2 esta instalado."
        )
    except json.JSONDecodeError:
        return "O Ollama retornou uma resposta invalida."
    except Exception as erro:
        return f"Ocorreu um erro na IA generativa: {erro}"


# ============================================================
# 7 PASSO: RESPOSTAS E ACOES DO ROBO
# ============================================================

def executar_intencao(intencao):
    respostas = {
        "saudacao": "Ola! Seja bem-vindo. Eu sou o CROMO, seu robo educacional.",
        "nome": "Meu nome e CROMO. Sou um robo educacional com inteligencia artificial.",
        "identidade": (
            "Sou o CROMO 3.0. Utilizo aprendizado de maquina, "
            "uma rede neural e inteligencia artificial generativa."
        ),
        "acenar": "Ola! Estou acenando para voce!",
        "dancar": "Vamos dancar! Observe meus movimentos!",
        "girar": "Preparado? Vou fazer um giro!",
        "pular": "Olha o meu salto!",
        "andar": "Vou caminhar um pouco!",
        "despedida": "Ate logo! Foi um prazer conversar com voce."
    }

    acoes = {
        "saudacao": "acenar",
        "nome": "parado",
        "identidade": "pensar",
        "acenar": "acenar",
        "dancar": "dancar",
        "girar": "girar",
        "pular": "pular",
        "andar": "andar",
        "despedida": "parado"
    }

    return respostas.get(intencao, ""), acoes.get(intencao, "parado")


# ============================================================
# 8 PASSO: THREAD DE PROCESSAMENTO
# ============================================================

def processar_comandos():
    while not encerrando.is_set():
        try:
            pergunta = fila_comandos.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if pergunta is None:
                break

            intencao, confianca, margem, palavras = identificar_intencao(pergunta)

            usar_ia_generativa = (
                intencao == "conversa"
                or eh_pergunta(pergunta)
                or palavras == 0
                or confianca < CONFIANCA_MINIMA
                or margem < MARGEM_MINIMA
            )

            if usar_ia_generativa:
                fila_interface.put((
                    "status",
                    "CROMO esta consultando a IA generativa..."
                ))
                resposta = conversar_com_ia(pergunta)
                acao = "pensar"
                origem = "IA GENERATIVA"
            else:
                resposta, acao = executar_intencao(intencao)
                origem = "MACHINE LEARNING"

            fila_interface.put(("diagnostico", intencao, confianca, origem))
            fila_interface.put(("resposta", resposta, acao))

        except Exception as erro:
            fila_interface.put((
                "resposta",
                f"Erro no processamento: {erro}",
                "parado"
            ))
        finally:
            fila_comandos.task_done()


# ============================================================
# 9 PASSO: THREAD DE VOZ - PYTTSX3
# ============================================================

def executar_voz():
    motor = None

    try:
        motor = pyttsx3.init()
        motor.setProperty("rate", 165)
        motor.setProperty("volume", 1.0)

        for voz in motor.getProperty("voices"):
            dados_voz = (str(voz.id) + " " + str(voz.name)).lower()
            if any(
                termo in dados_voz
                for termo in ("brazil", "portuguese", "brasil", "portugues")
            ):
                motor.setProperty("voice", voz.id)
                break

        print("Sistema de voz inicializado!")

    except Exception as erro:
        print("Erro ao inicializar a voz:", erro)
        motor = None

    while not encerrando.is_set():
        try:
            texto = fila_voz.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if texto is None:
                break
            if motor is not None:
                motor.say(texto)
                motor.runAndWait()
        except Exception as erro:
            print("Erro na fala:", erro)
        finally:
            fila_voz.task_done()

    if motor is not None:
        try:
            motor.stop()
        except Exception:
            pass


# ============================================================
# 9.1 PASSO: RECONHECIMENTO DE VOZ
# ============================================================

def executar_reconhecimento_voz():
    while not encerrando.is_set():
        if not escuta_ativa.wait(timeout=0.2):
            continue

        try:
            with sr.Microphone() as microfone:
                fila_interface.put(("status", "CROMO esta ajustando o microfone..."))
                reconhecedor.adjust_for_ambient_noise(microfone, duration=0.5)
                fila_interface.put(("status", "CROMO esta ouvindo voce..."))

                while escuta_ativa.is_set() and not encerrando.is_set():
                    try:
                        audio = reconhecedor.listen(
                            microfone,
                            timeout=1,
                            phrase_time_limit=7
                        )
                        fila_interface.put((
                            "status",
                            "CROMO esta reconhecendo sua fala..."
                        ))

                        texto = reconhecedor.recognize_google(
                            audio,
                            language="pt-BR"
                        )

                        if texto.strip():
                            fila_interface.put(("entrada_voz", texto))

                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        fila_interface.put((
                            "status",
                            "Nao consegui entender. Pode repetir?"
                        ))
                    except sr.RequestError as erro:
                        fila_interface.put((
                            "status",
                            f"Erro no reconhecimento de voz: {erro}"
                        ))
                        escuta_ativa.clear()
                        fila_interface.put(("microfone_estado", False))
                        break

        except Exception as erro:
            if not encerrando.is_set():
                fila_interface.put(("status", f"Erro no microfone: {erro}"))
                escuta_ativa.clear()
                fila_interface.put(("microfone_estado", False))


# ============================================================
# 10 PASSO: INTERFACE GRAFICA
# ============================================================

janela = tk.Tk()
janela.title("CROMO 3.0 - Robo Educacional Inteligente")
janela.geometry("920x900")
janela.minsize(760, 760)
janela.configure(bg=FUNDO)
janela.resizable(True, True)

titulo = tk.Label(
    janela,
    text="CROMO 3.0",
    font=("Arial", 25, "bold"),
    fg=VERDE_CLARO,
    bg=FUNDO
)
titulo.pack(pady=(10, 2))

subtitulo = tk.Label(
    janela,
    text="MACHINE LEARNING  |  REDE NEURAL  |  IA GENERATIVA",
    font=("Arial", 10, "bold"),
    fg=BRANCO,
    bg=FUNDO
)
subtitulo.pack()


# ============================================================
# 11 PASSO: AREA DE DESENHO DO ROBO
# ============================================================

canvas = tk.Canvas(
    janela,
    width=700,
    height=430,
    bg=FUNDO_CANVAS,
    highlightthickness=2,
    highlightbackground=VERDE
)
canvas.pack(pady=10)

# Fundo tecnológico e circuitos decorativos
canvas.create_oval(205, 350, 495, 410, fill="#0A3030", outline=VERDE_ESCURO, width=2)
canvas.create_oval(230, 360, 470, 400, outline=BRANCO_LUZ, width=2)
canvas.create_line(20, 80, 125, 80, fill=BRANCO, width=2)
canvas.create_line(20, 80, 20, 145, fill=BRANCO, width=2)
canvas.create_line(575, 100, 680, 100, fill=BRANCO, width=2)
canvas.create_line(680, 100, 680, 165, fill=BRANCO, width=2)
canvas.create_oval(12, 72, 28, 88, outline=BRANCO, width=2)
canvas.create_oval(672, 92, 688, 108, outline=BRANCO, width=2)

canvas.create_text(
    350, 25,
    text="CROMO  |  SISTEMA ONLINE",
    fill=BRANCO,
    font=("Arial", 11, "bold")
)


# ============================================================
# 12 PASSO: DESENHO DO ROBO - ARMADURA VERDE E LUZES BRANCAS
# ============================================================

# Antena
antena = canvas.create_line(
    350, 42, 350, 72,
    fill=BRANCO, width=5
)
antena_luz = canvas.create_oval(
    342, 30, 358, 46,
    fill=BRANCO, outline=VERDE_CLARO, width=2
)

# Laterais da cabeca / modulos auditivos
orelha_esquerda = canvas.create_oval(
    265, 88, 310, 145,
    fill=VERDE_ESCURO, outline=BRANCO, width=3
)
orelha_esquerda_luz = canvas.create_oval(
    276, 101, 299, 132,
    fill=FUNDO_CANVAS, outline=BRANCO, width=3
)
orelha_direita = canvas.create_oval(
    390, 88, 435, 145,
    fill=VERDE_ESCURO, outline=BRANCO, width=3
)
orelha_direita_luz = canvas.create_oval(
    401, 101, 424, 132,
    fill=FUNDO_CANVAS, outline=BRANCO, width=3
)

# Cabeca verde
cabeca = canvas.create_oval(
    285, 60, 415, 180,
    fill=VERDE, outline=BRANCO, width=4
)
# Faixa superior da cabeca
topo_cabeca = canvas.create_arc(
    300, 53, 400, 112,
    start=0, extent=180,
    style=tk.ARC, outline=VERDE_CLARO, width=5
)

# Visor preto
visor = canvas.create_rounded_rectangle if hasattr(canvas, "create_rounded_rectangle") else None
# Oval para compatibilidade com todas as versoes do Tkinter
visor = canvas.create_oval(
    296, 91, 404, 151,
    fill=PRETO_VISOR, outline=BRANCO, width=3
)

# Olhos brancos luminosos
olho_esquerdo = canvas.create_oval(
    315, 108, 335, 132,
    fill=BRANCO, outline=BRANCO_LUZ, width=2
)
olho_direito = canvas.create_oval(
    365, 108, 385, 132,
    fill=BRANCO, outline=BRANCO_LUZ, width=2
)
# Boca digital
boca = canvas.create_line(
    333, 143, 367, 143,
    fill=BRANCO, width=4, capstyle=tk.ROUND
)

# Pescoco
pescoco = canvas.create_rectangle(
    330, 178, 370, 205,
    fill=VERDE_ESCURO, outline=BRANCO, width=2
)

# Ombros
ombro_esquerdo = canvas.create_oval(
    250, 198, 300, 245,
    fill=VERDE, outline=BRANCO, width=3
)
ombro_direito = canvas.create_oval(
    400, 198, 450, 245,
    fill=VERDE, outline=BRANCO, width=3
)

# Tronco / armadura
tronco = canvas.create_polygon(
    290, 190,
    410, 190,
    435, 300,
    265, 300,
    fill=VERDE, outline=BRANCO, width=4
)

# Placas do peito
placa_peito = canvas.create_polygon(
    310, 210, 390, 210, 405, 280, 295, 280,
    fill=VERDE_ESCURO, outline=BRANCO, width=2
)

# Reator central branco
anel_peito = canvas.create_oval(
    324, 220, 376, 272,
    fill=VERDE, outline=BRANCO, width=4
)
luz_peito = canvas.create_oval(
    338, 234, 362, 258,
    fill=BRANCO, outline=BRANCO_LUZ, width=2
)

# Linhas/circuitos brancos do tronco
circuito_esquerdo = canvas.create_line(
    285, 235, 305, 235, 315, 250,
    fill=BRANCO, width=3, capstyle=tk.ROUND, joinstyle=tk.ROUND
)
circuito_direito = canvas.create_line(
    415, 235, 395, 235, 385, 250,
    fill=BRANCO, width=3, capstyle=tk.ROUND, joinstyle=tk.ROUND
)

# Bracos
braco_esquerdo = canvas.create_line(
    280, 220, 235, 290,
    fill=VERDE, width=23, capstyle=tk.ROUND
)
braco_direito = canvas.create_line(
    420, 220, 465, 290,
    fill=VERDE, width=23, capstyle=tk.ROUND
)

# Faixas luminosas dos bracos
luz_braco_esquerdo = canvas.create_line(
    266, 245, 246, 275,
    fill=BRANCO, width=4, capstyle=tk.ROUND
)
luz_braco_direito = canvas.create_line(
    434, 245, 454, 275,
    fill=BRANCO, width=4, capstyle=tk.ROUND
)

# Juntas dos cotovelos
junta_esquerda = canvas.create_oval(
    222, 278, 248, 304,
    fill=VERDE_ESCURO, outline=BRANCO, width=3
)
junta_direita = canvas.create_oval(
    452, 278, 478, 304,
    fill=VERDE_ESCURO, outline=BRANCO, width=3
)

# Maos
mao_esquerda = canvas.create_oval(
    215, 296, 253, 330,
    fill=VERDE, outline=BRANCO, width=3
)
mao_direita = canvas.create_oval(
    447, 296, 485, 330,
    fill=VERDE, outline=BRANCO, width=3
)

# Pernas
perna_esquerda = canvas.create_line(
    310, 294, 300, 355,
    fill=VERDE, width=25, capstyle=tk.ROUND
)
perna_direita = canvas.create_line(
    390, 294, 400, 355,
    fill=VERDE, width=25, capstyle=tk.ROUND
)

# Detalhes brancos nas pernas
luz_perna_esquerda = canvas.create_line(
    303, 320, 300, 340,
    fill=BRANCO, width=4
)
luz_perna_direita = canvas.create_line(
    397, 320, 400, 340,
    fill=BRANCO, width=4
)

# Pes
pe_esquerdo = canvas.create_oval(
    270, 345, 325, 375,
    fill=VERDE_ESCURO, outline=BRANCO, width=3
)
pe_direito = canvas.create_oval(
    375, 345, 430, 375,
    fill=VERDE_ESCURO, outline=BRANCO, width=3
)

# Brilho decorativo dos pes
luz_pe_esquerdo = canvas.create_line(
    282, 365, 313, 365, fill=BRANCO, width=3
)
luz_pe_direito = canvas.create_line(
    387, 365, 418, 365, fill=BRANCO, width=3
)


# ============================================================
# 13 PASSO: CONTROLE DA ANIMACAO
# ============================================================

def definir_acao(nova_acao):
    global acao_atual, inicio_acao
    acao_atual = nova_acao
    inicio_acao = time.time()


def atualizar_robo():
    global acao_atual

    if encerrando.is_set():
        return

    tempo = time.time() - inicio_acao
    deslocamento_y = 0

    braco_esq_x, braco_esq_y = 235, 310
    braco_dir_x, braco_dir_y = 465, 310
    mao_esq_x, mao_esq_y = 234, 313
    mao_dir_x, mao_dir_y = 466, 313

    perna_esq_x, perna_dir_x = 300, 400

    if acao_atual == "pular":
        deslocamento_y = -abs(math.sin(tempo * 5)) * 35
        if tempo > 1.5:
            definir_acao("parado")

    elif acao_atual == "dancar":
        deslocamento_y = math.sin(tempo * 8) * 7
        braco_esq_x = 235 + math.sin(tempo * 8) * 30
        braco_esq_y = 285 - abs(math.sin(tempo * 8)) * 35
        braco_dir_x = 465 - math.sin(tempo * 8) * 30
        braco_dir_y = 285 - abs(math.cos(tempo * 8)) * 35
        mao_esq_x, mao_esq_y = braco_esq_x, braco_esq_y + 3
        mao_dir_x, mao_dir_y = braco_dir_x, braco_dir_y + 3
        perna_esq_x = 300 + math.sin(tempo * 8) * 12
        perna_dir_x = 400 - math.sin(tempo * 8) * 12
        if tempo > 5:
            definir_acao("parado")

    elif acao_atual == "acenar":
        braco_dir_x = 465 + math.sin(tempo * 12) * 18
        braco_dir_y = 230 + math.sin(tempo * 12) * 24
        mao_dir_x, mao_dir_y = braco_dir_x, braco_dir_y + 3
        if tempo > 2.5:
            definir_acao("parado")

    elif acao_atual == "andar":
        braco_esq_x = 235 + math.sin(tempo * 8) * 12
        braco_dir_x = 465 - math.sin(tempo * 8) * 12
        mao_esq_x, mao_dir_x = braco_esq_x, braco_dir_x
        perna_esq_x = 300 + math.sin(tempo * 8) * 17
        perna_dir_x = 400 - math.sin(tempo * 8) * 17
        if tempo > 4:
            definir_acao("parado")

    elif acao_atual == "girar":
        escala = abs(math.cos(tempo * 4))
        largura = 130 * escala
        canvas.coords(
            cabeca,
            350 - largura / 2, 60,
            350 + largura / 2, 180
        )
        if tempo > 2:
            definir_acao("parado")

    else:
        deslocamento_y = math.sin(tempo * 2) * 3

    y = deslocamento_y

    # Cabeca e visor
    if acao_atual != "girar":
        canvas.coords(cabeca, 285, 60+y, 415, 180+y)
    canvas.coords(visor, 296, 91+y, 404, 151+y)
    canvas.coords(olho_esquerdo, 315, 108+y, 335, 132+y)
    canvas.coords(olho_direito, 365, 108+y, 385, 132+y)
    canvas.coords(boca, 333, 143+y, 367, 143+y)
    canvas.coords(antena, 350, 42+y, 350, 72+y)
    canvas.coords(antena_luz, 342, 30+y, 358, 46+y)

    # Modulos laterais
    canvas.move(orelha_esquerda, 0, 0)
    canvas.coords(orelha_esquerda, 265, 88+y, 310, 145+y)
    canvas.coords(orelha_esquerda_luz, 276, 101+y, 299, 132+y)
    canvas.coords(orelha_direita, 390, 88+y, 435, 145+y)
    canvas.coords(orelha_direita_luz, 401, 101+y, 424, 132+y)

    # Pescoco, ombros e tronco
    canvas.coords(pescoco, 330, 178+y, 370, 205+y)
    canvas.coords(ombro_esquerdo, 250, 198+y, 300, 245+y)
    canvas.coords(ombro_direito, 400, 198+y, 450, 245+y)
    canvas.coords(
        tronco,
        290, 190+y, 410, 190+y, 435, 300+y, 265, 300+y
    )
    canvas.coords(
        placa_peito,
        310, 210+y, 390, 210+y, 405, 280+y, 295, 280+y
    )
    canvas.coords(anel_peito, 324, 220+y, 376, 272+y)
    canvas.coords(luz_peito, 338, 234+y, 362, 258+y)
    canvas.coords(
        circuito_esquerdo,
        285, 235+y, 305, 235+y, 315, 250+y
    )
    canvas.coords(
        circuito_direito,
        415, 235+y, 395, 235+y, 385, 250+y
    )

    # Bracos, juntas e maos
    canvas.coords(braco_esquerdo, 280, 220+y, braco_esq_x, braco_esq_y+y)
    canvas.coords(braco_direito, 420, 220+y, braco_dir_x, braco_dir_y+y)
    canvas.coords(
        luz_braco_esquerdo,
        266, 245+y, braco_esq_x+10, braco_esq_y-15+y
    )
    canvas.coords(
        luz_braco_direito,
        434, 245+y, braco_dir_x-10, braco_dir_y-15+y
    )
    canvas.coords(
        junta_esquerda,
        braco_esq_x-13, braco_esq_y-13+y,
        braco_esq_x+13, braco_esq_y+13+y
    )
    canvas.coords(
        junta_direita,
        braco_dir_x-13, braco_dir_y-13+y,
        braco_dir_x+13, braco_dir_y+13+y
    )
    canvas.coords(
        mao_esquerda,
        mao_esq_x-19, mao_esq_y-17+y,
        mao_esq_x+19, mao_esq_y+17+y
    )
    canvas.coords(
        mao_direita,
        mao_dir_x-19, mao_dir_y-17+y,
        mao_dir_x+19, mao_dir_y+17+y
    )

    # Pernas e pes
    canvas.coords(perna_esquerda, 310, 294+y, perna_esq_x, 355+y)
    canvas.coords(perna_direita, 390, 294+y, perna_dir_x, 355+y)
    canvas.coords(
        luz_perna_esquerda,
        perna_esq_x+3, 320+y, perna_esq_x, 340+y
    )
    canvas.coords(
        luz_perna_direita,
        perna_dir_x-3, 320+y, perna_dir_x, 340+y
    )
    canvas.coords(
        pe_esquerdo,
        perna_esq_x-30, 345+y, perna_esq_x+25, 375+y
    )
    canvas.coords(
        pe_direito,
        perna_dir_x-25, 345+y, perna_dir_x+30, 375+y
    )
    canvas.coords(
        luz_pe_esquerdo,
        perna_esq_x-18, 365+y, perna_esq_x+13, 365+y
    )
    canvas.coords(
        luz_pe_direito,
        perna_dir_x-13, 365+y, perna_dir_x+18, 365+y
    )

    # Pulso do nucleo branco
    brilho = 2 + abs(math.sin(time.time() * 3)) * 2
    canvas.itemconfig(anel_peito, width=brilho + 2)
    canvas.itemconfig(luz_peito, outline=BRANCO_LUZ, width=brilho)

    janela.after(50, atualizar_robo)


# ============================================================
# 14 PASSO: AREA DE CONVERSA
# ============================================================

painel_conversa = scrolledtext.ScrolledText(
    janela,
    width=95,
    height=8,
    font=("Consolas", 10),
    bg="#071116",
    fg=BRANCO,
    insertbackground=BRANCO,
    wrap=tk.WORD,
    relief=tk.FLAT,
    borderwidth=2
)
painel_conversa.pack(padx=15, pady=8, fill="both", expand=True)
painel_conversa.config(state=tk.DISABLED)


def escrever_mensagem(remetente, mensagem):
    painel_conversa.config(state=tk.NORMAL)
    painel_conversa.insert(tk.END, f"{remetente}: {mensagem}\n\n")
    painel_conversa.see(tk.END)
    painel_conversa.config(state=tk.DISABLED)


# ============================================================
# 15 PASSO: ENTRADA DE COMANDOS
# ============================================================

entrada = tk.Entry(
    janela,
    font=("Arial", 13),
    bg="#E9FFF2",
    fg="#102018",
    insertbackground="#102018",
    relief=tk.FLAT
)
entrada.pack(padx=20, pady=8, fill="x")


def enviar_comando():
    if encerrando.is_set():
        return

    pergunta = entrada.get().strip()
    if not pergunta:
        return

    if pergunta.lower() in ["sair", "fechar", "encerrar programa"]:
        fechar_programa()
        return

    escrever_mensagem("VOCE", pergunta)
    fila_interface.put(("status", "CROMO esta processando sua mensagem..."))
    fila_comandos.put(pergunta)
    entrada.delete(0, tk.END)


# ============================================================
# 16 PASSO: STATUS DO ROBO
# ============================================================

status = tk.Label(
    janela,
    text="CROMO pronto para conversar.",
    font=("Arial", 10, "bold"),
    fg=AZUL_STATUS,
    bg=FUNDO
)
status.pack(pady=5)


# ============================================================
# 16.1 PASSO: CONTROLE DO MICROFONE
# ============================================================

def atualizar_botao_microfone(ativo):
    if ativo:
        botao_voz.config(text="PARAR MICROFONE", bg="#9B2C2C")
    else:
        botao_voz.config(text="ATIVAR MICROFONE", bg=VERDE_ESCURO)


def controlar_microfone():
    if encerrando.is_set():
        return

    if escuta_ativa.is_set():
        escuta_ativa.clear()
        atualizar_botao_microfone(False)
        status.config(text="Microfone desativado.")
    else:
        escuta_ativa.set()
        atualizar_botao_microfone(True)
        status.config(text="Preparando microfone...")


# ============================================================
# 16.2 PASSO: BOTOES
# ============================================================

frame_botoes = tk.Frame(janela, bg=FUNDO)
frame_botoes.pack(pady=(3, 12))

botao_enviar = tk.Button(
    frame_botoes,
    text="ENVIAR COMANDO",
    command=enviar_comando,
    font=("Arial", 10, "bold"),
    bg=VERDE_ESCURO,
    fg="white",
    activebackground=VERDE,
    activeforeground="white",
    width=20,
    cursor="hand2",
    relief=tk.FLAT
)
botao_enviar.pack(side=tk.LEFT, padx=8)

botao_voz = tk.Button(
    frame_botoes,
    text="ATIVAR MICROFONE",
    command=controlar_microfone,
    font=("Arial", 10, "bold"),
    bg=VERDE_ESCURO,
    fg="white",
    activebackground=VERDE,
    activeforeground="white",
    width=20,
    cursor="hand2",
    relief=tk.FLAT
)
botao_voz.pack(side=tk.LEFT, padx=8)

entrada.bind("<Return>", lambda evento: enviar_comando())


# ============================================================
# 17 PASSO: RECEBIMENTO DAS RESPOSTAS NA INTERFACE
# ============================================================

def verificar_fila_interface():
    if encerrando.is_set():
        return

    while True:
        try:
            evento = fila_interface.get_nowait()
        except queue.Empty:
            break

        try:
            tipo = evento[0]

            if tipo == "entrada_voz":
                texto = evento[1]
                escrever_mensagem("VOCE (VOZ)", texto)
                status.config(text="CROMO esta processando sua fala...")
                fila_comandos.put(texto)

            elif tipo == "status":
                status.config(text=evento[1])

            elif tipo == "diagnostico":
                intencao, confianca, origem = evento[1], evento[2], evento[3]
                status.config(
                    text=(
                        f"Origem: {origem} | Intencao: {intencao} | "
                        f"Probabilidade: {confianca:.2f}"
                    )
                )

            elif tipo == "resposta":
                resposta, acao = evento[1], evento[2]
                escrever_mensagem("CROMO", resposta)
                definir_acao(acao)
                fila_voz.put(resposta)

            elif tipo == "microfone_estado":
                atualizar_botao_microfone(evento[1])

        except Exception as erro:
            print("Erro ao atualizar interface:", erro)
        finally:
            fila_interface.task_done()

    if not encerrando.is_set():
        janela.after(100, verificar_fila_interface)


# ============================================================
# 18 PASSO: ENCERRAMENTO DO PROGRAMA
# ============================================================

def fechar_programa():
    if encerrando.is_set():
        return

    print("Encerrando CROMO...")
    encerrando.set()
    escuta_ativa.clear()

    fila_comandos.put(None)
    fila_voz.put(None)

    try:
        janela.destroy()
    except tk.TclError:
        pass


janela.protocol("WM_DELETE_WINDOW", fechar_programa)


# ============================================================
# 19 PASSO: INICIALIZACAO DAS THREADS
# ============================================================

thread_processamento = threading.Thread(
    target=processar_comandos,
    daemon=True,
    name="Thread-Processamento"
)
thread_voz = threading.Thread(
    target=executar_voz,
    daemon=True,
    name="Thread-Voz"
)
thread_reconhecimento = threading.Thread(
    target=executar_reconhecimento_voz,
    daemon=True,
    name="Thread-Reconhecimento"
)

thread_reconhecimento.start()
thread_processamento.start()
thread_voz.start()


# ============================================================
# 20 PASSO: INICIO DO SISTEMA
# ============================================================

escrever_mensagem(
    "CROMO",
    "Oie! Eu sou o CROMO 3.0. Minha inteligencia artificial está pronta!"
)
fila_voz.put(
    "Ola! Eu sou o Cromo três ponto zero. Minha inteligencia artificial esta pronta!"
)

atualizar_robo()
verificar_fila_interface()

print("\n======================================")
print("CROMO 3.0 INICIADO COM SUCESSO!")
print("Machine Learning: ATIVO")
print("Rede Neural MLP: ATIVA")
print("Interface Tkinter: ATIVA")
print("Sistema de voz: INICIALIZADO")
print("Reconhecimento de voz: DISPONIVEL")
print("Ollama: CONFIGURADO")
print("Memoria SQLite: CONFIGURADA")
print("Visual: ARMADURA VERDE E LUZES BRANCAS")
print("======================================\n")

janela.mainloop()
