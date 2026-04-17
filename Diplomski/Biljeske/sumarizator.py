import re
import sys

def extract_latest_week(content):
    pattern = r'\\tjedan\{.*?\}((?:(?!\\tjedan).)*)'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    if not matches:
        return None
    return matches[0].group(1).strip()

def clean_item(text):
    text = re.sub(r'\\lstinline\|([^|]*)\|', r'`\1`', text)
    text = re.sub(r'\\href\{.*?\}\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\textit\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\textbf\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', '', text)
    text = re.sub(r'\\\\\s*', ' ', text)
    text = re.sub(r'\\#', '#', text)
    return text.strip()

def extract_items_from_block(block):
    items = re.findall(r'\\item\s+(.*?)(?=\\item|$)', block, re.DOTALL)
    return [clean_item(item.strip()) for item in items if item.strip()]

def parse_days(week_text):
    day_pattern = r'\\dan\{([^}]+)\}((?:(?!\\dan).)*)'
    days = []
    for match in re.finditer(day_pattern, week_text, re.DOTALL):
        day_name = match.group(1).strip()
        day_body = match.group(2)

        # Find all itemize blocks
        itemize_blocks = re.findall(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', day_body, re.DOTALL)
        items = []
        for block in itemize_blocks:
            items.extend(extract_items_from_block(block))

        if items:
            days.append((day_name, items))
    return days

def extract_week_title(content):
    match = re.search(r'\\tjedan\{.*?\}\{([^}]+)\}', content)
    return match.group(1).strip() if match else "Nepoznati tjedan"

if __name__ == "__main__":
    file_path = "biljeske.tex"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Pogreška: Datoteka {file_path} nije pronađena.")
        sys.exit(1)

    week_title = extract_week_title(content)
    week_text = extract_latest_week(content)

    if not week_text:
        print("Nema pronađenog tjedna.")
        sys.exit(1)

    days = parse_days(week_text)

    if not days:
        print("Nema bullet pointova za prikaz.")
        sys.exit(1)

    print(f"\n=== {week_title} ===\n")
    for day_name, items in days:
        print(f"[ {day_name} ]")
        for item in items:
            print(f"  • {item}")
        print()