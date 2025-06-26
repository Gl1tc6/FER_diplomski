import re
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Question:
    topic: str
    date: str
    question: str
    answer: str
    subject: str  # MI ili ZI

class FlashcardQuiz:
    def __init__(self):
        self.questions: List[Question] = []
    
    def parse_file(self, content: str, subject: str) -> None:
        """Parsira sadržaj datoteke i ekstraktira pitanja i odgovore"""
        # Podijeli sadržaj po temama
        tema_sections = re.split(r'Tema:', content)[1:]  # Preskoči prvi prazan dio
        
        for section in tema_sections:
            lines = section.strip().split('\n')
            if len(lines) < 2:
                continue
                
            # Izvuci naziv teme i datum
            header = lines[0].strip()
            date_line = lines[1].strip() if len(lines) > 1 else ""
            
            topic = header
            date = ""
            if date_line.startswith("Datum:"):
                date = date_line.replace("Datum:", "").strip()
            
            # Pronađi sva Q: i A: parova
            content_lines = lines[2:] if date else lines[1:]
            content_text = '\n'.join(content_lines)
            
            # Regex za pronalaženje Q: i A: parova
            qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
            matches = re.findall(qa_pattern, content_text, re.DOTALL)
            
            for question_text, answer_text in matches:
                question_text = question_text.strip()
                answer_text = answer_text.strip()
                
                if question_text and answer_text:
                    self.questions.append(Question(
                        topic=topic,
                        date=date,
                        question=question_text,
                        answer=answer_text,
                        subject=subject
                    ))
    
    def load_data(self, zi_content: str, mi_content: str) -> None:
        """Učitava podatke iz obje datoteke"""
        self.parse_file(zi_content, "ZI")
        self.parse_file(mi_content, "MI")
    
    def get_questions_by_subject(self, subject_filter: str) -> List[Question]:
        """Filtrira pitanja prema predmetu"""
        if subject_filter == "sve":
            return self.questions
        else:
            return [q for q in self.questions if q.subject == subject_filter.upper()]
    
    def display_menu(self) -> str:
        """Prikazuje glavni meni i vraća izbor korisnika"""
        print("\n" + "="*50)
        print("🎓 FLASHCARD KVIZ")
        print("="*50)
        print("Izaberite što želite kvizirati:")
        print("1. Samo MI")
        print("2. Samo ZI")
        print("3. Sve (MI + ZI)")
        print("4. Izlaz")
        print("="*50)
        
        while True:
            choice = input("Vaš izbor (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            print("Molimo unesite valjani izbor (1-4)")
    
    def run_quiz(self, questions: List[Question]) -> None:
        """Pokreće kviz s danim pitanjima"""
        if not questions:
            print("Nema dostupnih pitanja za odabrani filter!")
            return
        
        print(f"\n🎯 Kviz pokrenut! Dostupno je {len(questions)} pitanja.")
        print("Pritisnite ENTER za prikaz odgovora, 'q' za izlaz iz kviza.\n")
        
        # Pomiješaj pitanja
        random.shuffle(questions)
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Pitanje {i}/{len(questions)}")
            print(f"📚 Predmet: {question.subject}")
            print(f"📋 Tema: {question.topic}")
            if question.date:
                print(f"📅 Datum: {question.date}")
            print("-" * 50)
            print(f"❓ {question.question}")
            print("-" * 50)
            
            user_input = input("Pritisnite ENTER za odgovor (ili 'q' za izlaz): ").strip().lower()
            
            if user_input == 'q':
                print("Kviz prekinut. Hvala na sudjelovanju! 👋")
                break
            
            print(f"\n✅ ODGOVOR:")
            print(f"{question.answer}")
            print("\n" + "="*70)
            
            if i < len(questions):
                next_input = input("Pritisnite ENTER za sljedeće pitanje (ili 'q' za izlaz): ").strip().lower()
                if next_input == 'q':
                    print("Kviz završen. Hvala na sudjelovanju! 👋")
                    break
        else:
            print(f"\n🎉 Čestitamo! Završili ste sve pitanja ({len(questions)})!")
            print("Hvala na sudjelovanju! 👋")
    
    def show_statistics(self) -> None:
        """Prikazuje statistike o pitanjima"""
        mi_count = len([q for q in self.questions if q.subject == "MI"])
        zi_count = len([q for q in self.questions if q.subject == "ZI"])
        total = len(self.questions)
        
        print(f"\n📊 STATISTIKE:")
        print(f"MI pitanja: {mi_count}")
        print(f"ZI pitanja: {zi_count}")
        print(f"Ukupno: {total}")
        
        # Prikaži teme po predmetima
        mi_topics = set(q.topic for q in self.questions if q.subject == "MI")
        zi_topics = set(q.topic for q in self.questions if q.subject == "ZI")
        
        print(f"\n📚 MI teme ({len(mi_topics)}):")
        for topic in sorted(mi_topics):
            count = len([q for q in self.questions if q.subject == "MI" and q.topic == topic])
            print(f"  • {topic} ({count} pitanja)")
        
        print(f"\n📚 ZI teme ({len(zi_topics)}):")
        for topic in sorted(zi_topics):
            count = len([q for q in self.questions if q.subject == "ZI" and q.topic == topic])
            print(f"  • {topic} ({count} pitanja)")
    
    def run(self) -> None:
        """Glavna petlja aplikacije"""
        while True:
            choice = self.display_menu()
            
            if choice == '4':
                print("Hvala što ste koristili Flashcard kviz! 👋")
                break
            elif choice == '1':
                questions = self.get_questions_by_subject("MI")
                self.run_quiz(questions)
            elif choice == '2':
                questions = self.get_questions_by_subject("ZI")
                self.run_quiz(questions)
            elif choice == '3':
                questions = self.get_questions_by_subject("sve")
                self.run_quiz(questions)

def main():
    quiz = FlashcardQuiz()
    
    # OVDJE ZAMIJENITE S ČITANJEM STVARNIH DATOTEKA:
    with open('ZI_q-a.txt', 'r', encoding='utf-8') as f:
        zi_content = f.read()
    with open('MI_q-a.txt', 'r', encoding='utf-8') as f:
        mi_content = f.read()
    
    quiz.load_data(zi_content, mi_content)
    
    # Prikaži statistike
    quiz.show_statistics()
    
    # Pokreni kviz
    quiz.run()

if __name__ == "__main__":
    main()
