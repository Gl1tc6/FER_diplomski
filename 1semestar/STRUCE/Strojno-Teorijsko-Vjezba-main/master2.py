import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random

class QuizApp:
    def __init__(self, root, quiz_dir):
        self.root = root
        self.root.title("FER Quiz")
        
        if not os.path.exists(quiz_dir):
            messagebox.showerror("Error", f"Directory not found: {quiz_dir}")
            return
            
        self.quiz_dir = quiz_dir
        self.score = 0
        self.current_question = 0
        
        self.questions = self.load_questions()
        
        if not self.questions:
            messagebox.showerror("Error", "No questions found in the specified directory.")
            return
            
        random.shuffle(self.questions)
        self.setup_gui()
        self.load_next_question()
    
    def load_questions(self):
        questions = []
        print(f"Looking for questions in: {self.quiz_dir}")
        
        try:
            for folder in sorted(os.listdir(self.quiz_dir)):
                folder_path = os.path.join(self.quiz_dir, folder)
                print(f"\nChecking folder: {folder}")
                
                if os.path.isdir(folder_path):
                    # List all files in the directory
                    all_files = os.listdir(folder_path)
                    print(f"Files in directory: {all_files}")
                    
                    # Get all image files (case insensitive)
                    image_files = [f for f in all_files 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"Found image files: {image_files}")
                    
                    # Sort image files by number
                    try:
                        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
                        print(f"Sorted image files: {image_files}")
                    except Exception as e:
                        print(f"Error sorting image files: {e}")
                    
                    # Look for answers file
                    answer_file = os.path.join(folder_path, 'Answers.txt')
                    if not os.path.exists(answer_file):
                        answer_file = os.path.join(folder_path, 'answers.txt')
                    
                    if os.path.exists(answer_file):
                        print(f"Found answers file: {answer_file}")
                        with open(answer_file, 'r') as f:
                            answers = [line.strip() for line in f.readlines()]
                        print(f"Answers read: {answers}")
                        
                        # Pair images with answers
                        for i, (img, ans) in enumerate(zip(image_files, answers)):
                            print(f"Pairing image {img} with answer {ans}")
                            questions.append({
                                'folder': folder_path,
                                'image': img,
                                'correct_answer': ans,
                                'topic': folder
                            })
                    else:
                        print(f"No answers.txt found in {folder_path}")
                        
        except Exception as e:
            print(f"Error loading questions: {str(e)}")
            messagebox.showerror("Error", f"Error loading questions: {str(e)}")
            return []
            
        print(f"\nTotal questions loaded: {len(questions)}")
        return questions
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, expand=True, fill='both')
        
        # Topic label
        self.topic_label = ttk.Label(main_frame, text="", font=('Arial', 12))
        self.topic_label.pack(pady=(0, 10))
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(pady=10)
        
        # Options frame
        self.options_frame = ttk.Frame(main_frame)
        self.options_frame.pack(pady=10)
        
        # Option buttons in a grid
        self.option_buttons = []
        options = ['A', 'B', 'C', 'D']
        for i, option in enumerate(options):
            btn = ttk.Button(self.options_frame, 
                           text=option,
                           width=20,
                           command=lambda x=option: self.check_answer(x))
            btn.grid(row=i//2, column=i%2, padx=5, pady=5)
            self.option_buttons.append(btn)
        
        # Feedback label
        self.feedback_label = ttk.Label(main_frame, text="", font=('Arial', 11))
        self.feedback_label.pack(pady=10)
        
        # Next button (initially hidden)
        self.next_button = ttk.Button(main_frame, text="Next Question", 
                                    command=self.next_question)
        
        # Score and progress labels
        self.score_label = ttk.Label(main_frame, text="Score: 0/0", font=('Arial', 11))
        self.score_label.pack(pady=10)
        
        self.progress_label = ttk.Label(main_frame, text="Question 0/0", font=('Arial', 11))
        self.progress_label.pack(pady=(0, 10))
    
    def load_next_question(self):
        # Reset feedback and hide next button
        self.feedback_label.configure(text="")
        self.next_button.pack_forget()
        
        # Enable all option buttons
        for btn in self.option_buttons:
            btn.configure(state='normal')
        
        if self.current_question < len(self.questions):
            question = self.questions[self.current_question]
            
            # Update topic
            topic_name = question['topic'].replace('_', ' ')
            self.topic_label.configure(text=topic_name)
            
            # Load and display image
            image_path = os.path.join(question['folder'], question['image'])
            try:
                img = Image.open(image_path)
                
                # Calculate resize dimensions while maintaining aspect ratio
                display_width = 800
                display_height = 600
                img_ratio = img.size[0] / img.size[1]
                
                if img_ratio > display_width/display_height:
                    new_width = display_width
                    new_height = int(display_width / img_ratio)
                else:
                    new_height = display_height
                    new_width = int(display_height * img_ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
            
            # Update labels
            self.score_label.configure(
                text=f"Score: {self.score}/{self.current_question}")
            self.progress_label.configure(
                text=f"Question {self.current_question + 1}/{len(self.questions)}")
        else:
            # Quiz finished
            self.show_final_score()
    
    def check_answer(self, selected_answer):
        question = self.questions[self.current_question]
        correct_answer = question['correct_answer']
        
        # Disable all option buttons
        for btn in self.option_buttons:
            btn.configure(state='disabled')
        
        if selected_answer == correct_answer:
            self.score += 1
            self.feedback_label.configure(
                text=f"Correct! The answer is {correct_answer}",
                foreground='green'
            )
        else:
            self.feedback_label.configure(
                text=f"Incorrect. The correct answer is {correct_answer}",
                foreground='red'
            )
        
        # Show next button
        self.next_button.pack(pady=10)
    
    def next_question(self):
        self.current_question += 1
        self.load_next_question()
    
    def show_final_score(self):
        # Clear screen and show final score
        for widget in self.root.winfo_children():
            widget.destroy()
        
        final_frame = ttk.Frame(self.root)
        final_frame.pack(padx=20, pady=20, expand=True)
        
        final_score = ttk.Label(
            final_frame, 
            text=f"Quiz Complete!\nFinal Score: {self.score}/{len(self.questions)}",
            font=('Arial', 14)
        )
        final_score.pack(pady=20)
        
        percentage = (self.score / len(self.questions)) * 100 if self.questions else 0
        percentage_label = ttk.Label(
            final_frame,
            text=f"Percentage: {percentage:.1f}%",
            font=('Arial', 12)
        )
        percentage_label.pack(pady=10)
        
        # Restart button
        restart_btn = ttk.Button(
            final_frame,
            text="Restart Quiz",
            command=lambda: self.__init__(self.root, self.quiz_dir)
        )
        restart_btn.pack(pady=20)

def main():
    quiz_dir = "/home/ante/Documents/GitHub/FER_diplomski/1semestar/STRUCE/Strojno-Teorijsko-Vjezba-main/Data/Questions"
    
    if not os.path.exists(quiz_dir):
        print(f"Error: Directory not found: {quiz_dir}")
        return
        
    root = tk.Tk()
    app = QuizApp(root, quiz_dir)
    root.mainloop()

if __name__ == "__main__":
    main()