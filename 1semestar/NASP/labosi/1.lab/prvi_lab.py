import locale
import numpy as np
locale.setlocale(locale.LC_ALL, '')

""" Kod za prvu laboratorijsku vježbu
Podijeljeni su zadatci u podklase te pozvani u main sa testnim primjerima


Code for first lab assigment
Individual assigments are divided in classes and called in main with test data """

class zad1:
    def __init__(self) -> None:
        names = ["Ana", "Petar", "Ana", "Lucija", "Vanja", "Pavao", "Lucija"]
        names_desc = self.reverse_sort_homemade(names)
        selected_names = names_desc[:-1]
        unique_selected_names = set(selected_names)
        pass_names = []
        for el in unique_selected_names:
            pass_names.append(f"{el}- pass")
        
        print("Names: ", names)
        print("Sorted: ", names_desc)
        print("Selected: ", selected_names)
        print("Unique: ", unique_selected_names)
        print("PASS: ", pass_names)
        
    def reverse_sort(self, names :list) -> list:
        sortArr = sorted(names, reverse=True)
        return sortArr
    
    def reverse_sort_homemade(self, names :list) -> list:
        temp = names.copy()
        res = []
        for i in range(len(names)):
            m = max(temp)
            temp.remove(m)
            res.append(m)
        return res

class zad2:
    def __init__(self) -> None:
        person_data = {
            "Ana": 1995,
            "Zoran": 1978,
            "Lucija": 2001,
            "Anja": 1997
        }
        
        for k,v in person_data.items():
            person_data[k] = v-1
        
        print(person_data)
        
        year_age = []
        
        for year in person_data.values():
            year_age.append((year, 2022-year))
        
        print(year_age)
            
        
class zad3:
    def __init__(self) -> None:
        vector_a = np.array([[1], [3], [5]])
        vector_b = np.array([[2], [4], [6]])
        
        mat_mul = np.outer(vector_a, vector_b)
        vect_dot = vector_a @ vector_b.T
        mat_exp = [x**2 for x in mat_mul]
        sub_mat = [mat_exp[-2][-2:], mat_exp[-1][-2:]]
        
        print(f"Vector A: {vector_a}\nVector B: {vector_b}")
        print(f"Matrix mult.:\n{mat_mul}\n")
        print(f"Vector . prod:\n{vect_dot}\n")
        print(f"Matrix expontent.:\n{mat_exp}\n")
        print(f"SubMatrix.:\n{sub_mat}\n")
        
    
class zad4:
    def __init__(self) -> None:
        first_person = self.Person("Marko", 39)
        second_person = self.Person("Ivan", 17)
        print(second_person)
        
        second_person.increase_age()
        print(second_person)
        
        first_person_detail = self.PersonDetail("Ana", 25, "Unska 3")
        first_person_detail.increase_age()
        
        print(first_person_detail)
        
    
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def increase_age(self) -> None:
            self.age += 1
        
        def __str__(self) -> str:
            return f"{self.name}, {self.age}"
    
    class PersonDetail(Person):
        def __init__(self, name, age, address):
            super().__init__(name, age)
            self.address = address

print("Prvi zadatak: \n######")
zad1()
print("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\n")
print("Drugi zadatak: \n######")
zad2()
print("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\n")
print("Treći zadatak: \n######")
zad3()
print("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\n")
print("Četvrti zadatak: \n######")
zad4()