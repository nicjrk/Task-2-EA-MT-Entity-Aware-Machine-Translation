import json

def create_jsonl_dataset(en_file, de_file, output_file, max_lines=15000):
    with open(en_file, 'r', encoding='utf-8') as f_en, \
         open(de_file, 'r', encoding='utf-8') as f_de, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for i, (en, de) in enumerate(zip(f_en, f_de)):
            if i >= max_lines:
                break
                
            en = en.strip()
            de = de.strip()
            
            data = {
                "id": f"opensubs_{i}",
                "source_locale": "en",
                "target_locale": "de",
                "source": en,
                "target": de,
                "entities": [],
                "from": "opensubs"
            }
            
            f_out.write(json.dumps(data) + '\n')

# Căile către fișiere
input_en = "./training data/train/de/OpenSubtitles.de-en.en"
input_de = "./training data/train/de/OpenSubtitles.de-en.de"
output_jsonl = "./training data/train/de/train_tiny2.jsonl"

create_jsonl_dataset(input_en, input_de, output_jsonl)