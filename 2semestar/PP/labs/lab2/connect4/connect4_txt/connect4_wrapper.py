#!/usr/bin/env python3
import subprocess
import os
import sys

class Connect4Game:
    def __init__(self, width=7, height=6, depth=8):
        self.width = width
        self.height = height
        self.depth = depth
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.board_file = "igra1.txt"
        self.player_turn = True  # True = igrač (2), False = računalo (1)
        
    def create_board_file(self):
        """Stvara datoteku s pločom u potrebnom formatu"""
        with open(self.board_file, 'w') as f:
            f.write(f"{self.height} {self.width}\n")
            for row in self.board:
                f.write(" " + "  ".join(map(str, row)) + " \n")
    
    def load_board_from_file(self):
        """Učitava ploču iz datoteke"""
        try:
            with open(self.board_file, 'r') as f:
                lines = f.readlines()
                # Preskačemo prvu liniju s dimenzijama
                for i in range(1, len(lines)):
                    row_data = lines[i].strip().split()
                    if row_data:  # Ako red nije prazan
                        for j, val in enumerate(row_data):
                            if j < self.width:
                                self.board[i-1][j] = int(val)
        except FileNotFoundError:
            print("Greška: Datoteka ploca.txt nije pronađena!")
            sys.exit(1)
    
    def display_board(self):
        """Prikazuje trenutno stanje ploče"""
        print("\nTrenutno stanje ploče:")
        print("+" + "---+" * self.width)
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("   |", end="")
                elif cell == 1:
                    print(" O |", end="")  # Računalo
                else:
                    print(" X |", end="")  # Igrač
            print()
            print("+" + "---+" * self.width)
        
        # Prikaz brojeva stupaca
        print(" ", end="")
        for i in range(1, self.width + 1):
            print(f" {i} ", end=" ")
        print("\n")
    
    def make_player_move(self):
        """Obrađuje potez igrača"""
        while True:
            try:
                col = int(input(f"Unesite stupac (1-{self.width}): ")) - 1
                if 0 <= col < self.width:
                    # Provjeri je li stupac pun
                    if self.board[0][col] == 0:
                        # Pronađi najniži prazan red u stupcu
                        for row in range(self.height - 1, -1, -1):
                            if self.board[row][col] == 0:
                                self.board[row][col] = 2  # Igrač je 2
                                break
                        break
                    else:
                        print("Stupac je pun! Odaberite drugi stupac.")
                else:
                    print(f"Molim unesite broj između 1 i {self.width}!")
            except ValueError:
                print("Molim unesite valjani broj!")
    
    def run_computer_move(self):
        """Pokreće računalni potez pomoću MPI programa"""
        try:
            # Stvara datoteku s trenutnim stanjem
            self.create_board_file()
            
            # Pokreće MPI program
            cmd = f"mpirun -np 4 ./connect4_mpi {self.board_file} {self.depth}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Greška pri pokretanju programa: {result.stderr}")
                return False
            
            output = result.stdout.strip()
            print(f"Izlaz programa: {output}")
            
            # Provjeri je li igra završena
            if "Igra zavrsena" in output:
                if "pobjeda racunala" in output.lower():
                    print("Računalo je pobijedilo!")
                else:
                    print("Igra je završena!")
                return True
            
            # Učitaj novu ploču nakon računalnog poteza
            self.load_board_from_file()
            return False
            
        except subprocess.TimeoutExpired:
            print("Program je prekasno odgovorio (timeout)!")
            return False
        except Exception as e:
            print(f"Greška: {e}")
            return False
    
    def check_game_end(self):
        """Provjeri je li igra završena (jednostavna provjera za punu ploču)"""
        # Provjeri je li gornji red pun
        return all(cell != 0 for cell in self.board[0])
    
    def play(self):
        """Glavna petlja igre"""
        print("🎮 Connect4 igra pokrenuta!")
        print("Vi ste X (igrač 2), računalo je O (igrač 1)")
        print("Cilj: spojite 4 svoje figure u niz (vodoravno, okomito ili dijagonalno)")
        
        # Stvori početnu ploču
        self.create_board_file()
        self.display_board()
        
        while True:
            if self.player_turn:
                print("🎯 Vaš red!")
                self.make_player_move()
                self.create_board_file()  # Ažuriraj datoteku
                self.display_board()
                
                # Provjeri je li igra završena
                if self.check_game_end():
                    print("⚖️ Ploča je puna! Neriješeno!")
                    break
                    
                self.player_turn = False
                
            else:
                print("🤖 Red računala...")
                game_ended = self.run_computer_move()
                if game_ended:
                    self.display_board()
                    break
                    
                self.display_board()
                
                # Provjeri je li igra završena
                if self.check_game_end():
                    print("⚖️ Ploča je puna! Neriješeno!")
                    break
                    
                self.player_turn = True

def main():
    # Provjeri postoje li potrebni argumenti
    if len(sys.argv) > 1:
        try:
            depth = int(sys.argv[1])
        except ValueError:
            depth = 8
    else:
        depth = 8
    
    # Provjeri postoji li izvršna datoteka
    if not os.path.exists("./connect4_mpi"):
        print("Greška: Datoteka './connect4_mpi' nije pronađena!")
        print("Molim kompajlirajte vašu C++ aplikaciju i stavite ju u isti direktorij.")
        sys.exit(1)
    
    print(f"Connect4 Python Wrapper")
    print(f"Dubina pretrage: {depth}")
    
    # Stvori i pokreni igru
    game = Connect4Game(depth=depth)
    
    try:
        game.play()
    except KeyboardInterrupt:
        print("\n\n👋 Igra prekinuta. Hvala na igranju!")
    except Exception as e:
        print(f"\n❌ Neočekivana greška: {e}")
    finally:
        # Očisti privremene datoteke ako je potrebno
        if os.path.exists("ploca.txt"):
            print(f"Datoteka 'igra1.txt' ostaje sačuvana s konačnim stanjem igre.")

if __name__ == "__main__":
    main()