import random

class QuizParser:
    def __init__(self, filename):
        self.multiple_choice = []
        self.short_answer = []
        self.parse_file(filename)

    def parse_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read().split('\n')
            
            i = 0
            while i < len(content):
                if content[i].startswith('#'):  # Multiple choice question
                    question = content[i][1:].strip()
                    i += 2
                    answers = []
                    correct_answer = None
                    
                    while i < len(content) and content[i].strip() != '':
                        if content[i].startswith('.'):
                            correct_answer = content[i][1:].strip()
                            answers.append(correct_answer)
                        else:
                            answers.append(content[i].strip())
                        i += 1
                    
                    self.multiple_choice.append({
                        'question': question,
                        'answers': answers,
                        'correct': correct_answer
                    })
                
                elif content[i].startswith('$'):  # Short answer question
                    question = content[i][1:].strip()
                    i += 1
                    while i < len(content) and content[i].strip() == '':
                        i += 1
                    answer = content[i].strip()
                    self.short_answer.append({
                        'question': question,
                        'answer': answer
                    })
                i += 1

    def run_quiz(self, quiz_type='mixed'):
        questions = []
        if quiz_type == 'multiple':
            questions = self.multiple_choice
        elif quiz_type == 'short':
            questions = self.short_answer
        else:  # mixed
            questions = self.multiple_choice + self.short_answer
        
        random.shuffle(questions)
        score = 0
        total = len(questions)
        
        print("Welcome to the Quiz!")
        print("-------------------")
        
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}/{total}:")
            print(q['question'])
            
            if 'answers' in q:  # Multiple choice
                for j, answer in enumerate(q['answers'], 1):
                    print(f"{j}. {answer}")
                
                while True:
                    try:
                        choice = int(input("\nEnter your answer (number): "))
                        if 1 <= choice <= len(q['answers']):
                            break
                        print("Please enter a valid number!")
                    except ValueError:
                        print("Please enter a valid number!")
                
                user_answer = q['answers'][choice-1]
                if user_answer == q['correct']:
                    print("Correct!")
                    score += 1
                else:
                    print(f"Wrong! The correct answer is: {q['correct']}")
            
            else:  # Short answer
                user_answer = input("\nYour answer: ").strip()
                if user_answer.lower() == q['answer'].lower():
                    print("Correct!")
                    score += 1
                else:
                    print(f"Wrong! The correct answer is: {q['answer']}")
        
        print(f"\nQuiz completed! Your score: {score}/{total} ({(score/total)*100:.1f}%)")
        return score, total

def main():
    filename = "/home/ante/Documents/GitHub/FER_diplomski/1semestar/ARHRAC2/2015_16_teorija/ARH2_Teorija4wins.txt"
    quiz = QuizParser(filename)
    
    print("\nSelect quiz type:")
    print("1. Multiple choice questions only")
    print("2. Short answer questions only")
    print("3. Mixed (both types)")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if 1 <= choice <= 3:
                break
            print("Please enter a valid number!")
        except ValueError:
            print("Please enter a valid number!")
    
    quiz_types = {
        1: 'multiple',
        2: 'short',
        3: 'mixed'
    }
    
    quiz.run_quiz(quiz_types[choice])

if __name__ == "__main__":
    main()