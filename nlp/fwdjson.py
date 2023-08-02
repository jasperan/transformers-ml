from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
import ujson as json

def load_json_list(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        print(len(data))
    return list(data)

json_file_path = "./example.json"
data_list = load_json_list(json_file_path)



sentences = ["I would like do discover the nuances between English and other foreign languages through the power of transformers"]

new_translations = list()

# all language variations I want to create for forward-backward translation
#language_variations = ['ar', 'ru', 'de', 'fr', 'zh', 'es',
#                       'it', 'ROMANCE', 'nl', 'sv', 'fi'] # need to find suitable checkpoints for korean 'ko'
language_variations = ['es', 'de']

for x in language_variations:
    fwd_translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-{}-{}".format('en', x), max_length=500)
    for i in data_list:
        print(i)
        assert 'input' in i
        assert 'output' in i
        assert 'instruction' in i
        
        #i['input']
        #i['output']
        #i['instruction']


        try:
            translated_input = fwd_translator(i['input'])
            translated_output = fwd_translator(i['output'])
            translated_input = translated_input[0]['translation_text']
            translated_output = translated_output[0]['translation_text']
        except IndexError:
            print('[ERR] {} {}'.format(translated_input, translated_output))
            continue


        print('[INPUT] {}'.format(translated_input))
        print('[OUTPUT] {}'.format(translated_output))
    
        new_json = {
            'instruction': 'Eres un experto médico y responderás preguntas relacionadas con consultas médicas.',
            'input': translated_input,
            'output': translated_output,
        }

        new_translations.append(new_json)



print('New translations: {}'.format(new_translations))

print('[STAT] TT: {} UT: {}'.format(len(new_translations), len(list(set(new_translations)))))

print('Unique translations: {}'.format(list(set(new_translations))))

import json

output_file_path = "./new_translations.json"

with open(output_file_path, 'w') as file:
    json.dump(new_translations, file)


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