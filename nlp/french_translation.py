from transformers import pipeline

text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
translator = pipeline(task="translation", model="t5-small")
print(translator(text))


en_fr_translator = pipeline("translation_en_to_fr")
print(en_fr_translator("How old are you?"))