from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

"""
T5-Small is the checkpoint with 60 million parameters.

Developed by: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. See associated paper and GitHub repo
Model type: Language model
Language(s) (NLP): English, French, Romanian, German

You can either use t5-small or t5-base; or stick to what's there, language-specific machine translation encoders developed by Helsinki-NLP which work awesome
"""

'''
sentences = [
    "Hugging Face is a community-based open-source platform for machine learning.",
    "I love to eat pizza.",
    "The quick brown fox jumps over the lazy dog.",
    "The capital of France is Paris.",
    "I am learning how to code.",
    "The sun rises in the east and sets in the west.",
    "My favorite color is blue.",
    "The Earth is round.",
    "I enjoy listening to music.",
    "Water freezes at 0 degrees Celsius."
]
'''

sentences = ["I would like do discover the nuances between English and other foreign languages through the power of transformers"]

new_translations = list()

# all language variations I want to create for forward-backward translation
language_variations = ['ar', 'ru', 'de', 'fr', 'zh', 'es',
                       'it', 'ROMANCE', 'nl', 'sv', 'fi'] # need to find suitable checkpoints for korean 'ko'


for x in language_variations:
    fwd_translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-{}-{}".format('en', x))
    bck_translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-{}-{}".format(x, 'en'))

    for i in sentences:
        translated_forward = fwd_translator(i)
        # Now, we put it back and append it into a list of translated sentences
        t5_decoded_input = translated_forward[0]['translation_text']
        print('After translation: {}'.format(t5_decoded_input))

        #print(t5_decoded_input)
        translated_backward = bck_translator(t5_decoded_input)
        #print(translated_backward[0]['translation_text'])  
        new_translations.append(translated_backward[0]['translation_text'])
        #print('Backwards-Translated into {}'.format(translated_backward))


print('New translations: {}'.format(new_translations))

print('[STAT] TT: {} UT: {}'.format(len(new_translations), len(list(set(new_translations)))))

print('Unique translations: {}'.format(list(set(new_translations))))

'''
# german pipelines
translator = pipeline(task="translation",
                        model="Helsinki-NLP/opus-mt-en-de") # t5-small
backwards_translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-de-en") # t5-small


for i in sentences:


    translated_forward = translator(i)
    # Now, we put it back and append it into a list of translated sentences
    t5_decoded_input = translated_forward[0]['translation_text']
    print('After translation: {}'.format(t5_decoded_input))

    #print(t5_decoded_input)
    translated_backward = backwards_translator(t5_decoded_input)
    #print(translated_backward[0]['translation_text'])  
    new_translations.append(translated_backward)
    #print('Backwards-Translated into {}'.format(translated_backward))

print('New translations: {}'.format(new_translations))

'''