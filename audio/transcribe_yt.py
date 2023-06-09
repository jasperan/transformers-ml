'''
@author jasperan

Requires:
- ffmpeg?
- pytube
- transformers
- ujson
- rich
'''

import pytube
import pytube.exceptions
import os
from transformers import pipeline
from rich import print
import ujson as json 

while True:
    try:
        input_link = input("YouTube Video URL:")
        yt = pytube.YouTube(input_link) 

        wav_filename = 'output.wav'

        audio = yt.streams.filter(only_audio=True).first() # get only the audio
        audio_path = audio.download(filename=wav_filename)
        print(audio_path)

        # write to audio to .wav file with ffmpeg
        os.system(f'ffmpeg -i "{audio_path}" "{wav_filename}"')
        break

    except pytube.exceptions.RegexMatchError:  # Handles all the input exceptions
        print("🛑 Invalid URL! 🛑")


# use the HF transformer for wav2vec2
cls = pipeline("automatic-speech-recognition")
result = cls(wav_filename)

print('Summary: {}'.format(result))
