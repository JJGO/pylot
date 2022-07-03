from IPython.display import Audio, display

BEEP_URL = "https://github.com/JJGO/pylot-assets/raw/main/audio/arcade-beep.wav"


def beep():
    display(Audio(url=BEEP_URL, autoplay=True))
