import re
import requests
import sys

def clean_latex(text):
    """Uklanja LaTeX komande kako bi LLM dobio čisti tekst."""
    text = re.sub(r'\\dan\{.*?\}', '\n--- ', text)  # Pretvara dan u čitljiv separator
    text = re.sub(r'\\begin\{itemize\}|\\end\{itemize\}', '', text)
    text = re.sub(r'\\item', '•', text)
    text = re.sub(r'\\href\{.*?\}\{(.*?)\}', r'\1', text) # Ostavi samo tekst linka
    text = re.sub(r'\\[a-z]+(\{.*?\})?', '', text) # Briše ostale generičke komande
    return text.strip()

def extract_latest_week(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Hvata sadržaj od prvog \tjedan do \newpage ili sljedećeg \tjedan
        # Ovo osigurava da uzimamo samo zadnji tjedan koji si upisao na vrh
        pattern = r'\\tjedan\{.*?\}.*?((?:(?!\\tjedan|\\newpage).)*)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            raw_text = match.group(1).strip()
            return clean_latex(raw_text)
        return None
    except FileNotFoundError:
        print(f"Pogreška: Datoteka {file_path} nije pronađena.")
        sys.exit(1)

def get_summary(text):
    url = "http://localhost:11434/api/generate"
    prompt = (
        f"Djeluj kao mentor za diplomski rad na FER-u. "
        f"Sažmi sljedeće bilješke o radu u nekoliko konciznih tehničkih bullet pointa i kad opisuješ što je napravljeno koristi svršene glagole tj kao da je sve opisano i napravljeno. "
        f"Piši na hrvatskom jeziku.\n\nBILJEŠKE:\n{text}"
    )
    
    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 1000, # Ograničavamo dužinu odgovora radi brzine
            "temperature": 0.4  # Manja temperatura za precizniji, manje 'maštovit' sažetak
        }
    }

    print("Generiram sažetak (Llama 3.2 3B)...")
    response = requests.post(url, json=payload)
    return response.json().get('response', "Greška u generiranju.")

if __name__ == "__main__":
    file_name = "biljeske.tex" # Promijeni u pravi naziv svog fajla
    
    extracted_text = extract_latest_week(file_name)
    
    if extracted_text:
        print(f"--- Ekstrahirani tekst ({len(extracted_text)} znakova) ---")
        print(extracted_text)
        print("Generiranje sažetka...")
        summary = get_summary(extracted_text)
        print("\nSAŽETAK TJEDNA:")
        print(summary)
    else:
        print("Nije pronađen sadržaj pod oznakom \\tjedan.")