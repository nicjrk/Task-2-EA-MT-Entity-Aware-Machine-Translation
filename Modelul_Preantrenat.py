from transformers import MarianMTModel, MarianTokenizer

# Încarcă modelul pre-antrenat pentru traducere din engleză în germană
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    # Ttokenizam datele de intrare
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # generam traducerea cu modelul prea
    translated = model.generate(**inputs)

    # decodificare
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


print("Model ready. Type a sentence in English:")
while True:
    input_text = input("You: ")
    if input_text.lower() in ['exit', 'quit']:
        print("Exiting translation.")
        break
    print("Translation:", translate_text(input_text))
