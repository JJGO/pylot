def beep():
    BEEP_URL = "https://github.com/JJGO/pylot-assets/raw/main/audio/arcade-beep.wav"
    from IPython.display import Audio, display
    display(Audio(url=BEEP_URL, autoplay=True))
