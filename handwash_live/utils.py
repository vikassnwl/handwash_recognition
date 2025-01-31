import pyttsx3
print("Initializing speech engine...")
engine = pyttsx3.init()
print("Speech engine initialized successfully!")

def speak(sentence, speech_rate=200):
    engine.setProperty("rate", speech_rate)
    engine.say(sentence)
    engine.runAndWait()