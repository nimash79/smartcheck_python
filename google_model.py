import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable
with sr.AudioFile("nima.wav") as source:
    audio_text = r.record(source)
    # audio_text = r.listen(source)
    # recoginze_() method will throw a request
    # error if the API is unreachable,
    # hence using exception handling
    
    try:
        # using google speech recognition
        text = r.recognize_google(audio_text, language="fa-IR")
        print("Text: "+ text)
        if "جنگ" in text:
            print("ALARM: Someone is talking about war!")
    except:
         print("Sorry, I did not get that")