import speech_recognition as sr

reconhecedor = sr.Recognizer()

print("Inicializando microfone...")

try:
    with sr.Microphone() as microfone:

        print("Ajustando o ruido ambiente...")
        reconhecedor.adjust_for_ambient_noise(
            microfone,
            duration=1
        )

        print("Pode falar!")

        audio = reconhecedor.listen(
            microfone,
            timeout=10,
            phrase_time_limit=8
        )

    print("Reconhecendo sua fala...")

    texto = reconhecedor.recognize_google(
        audio,
        language="pt-BR"
    )

    print("Você disse:", texto)

except sr.WaitTimeoutError:
    print("Nenhuma fala foi detectada.")

except sr.UnknownValueError:
    print("Não consegui entender o que você falou.")

except sr.RequestError as erro:
    print("Erro no serviço de reconhecimento:", erro)

except Exception as erro:
    print("Erro:", erro)