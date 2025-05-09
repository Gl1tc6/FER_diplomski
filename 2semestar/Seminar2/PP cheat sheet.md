# Cheat Sheet – Paralelno programiranje

---

## 1. Osnovni modeli (Flynnova taksonomija)  
- **SISD**: jedno procesorsko jezgro, jedan niz podataka 
- **SIMD**: više procesora izvodi istu instrukciju nad različitim podacima (lock-step)
- **MIMD**: svaki procesor neovisno izvodi vlastiti kod i pristupa podacima
---

## 2. Ključna svojstva paralelnih algoritama  
- **Istodobnost (concurrency)**: više radnji može se izvršiti istovremeno 
- **Skalabilnost (scalability)**: algoritam mora učinkovito rasti s brojem procesora 
- **Lokalnost (locality)**: minimalizirati udaljene (globalne) pristupe memoriji 
- **Modularnost (modularity)**: dijelovi algoritma primjenjivi u različitim kontekstima 

---

## 3. Zakonitosti paralelizacije  

### 3.1 Amdahlov zakon  
- **Formula**:  
  $$
    S(N_p) = \frac{1}{(1 - p) + \tfrac{p}{N_p}}
  $$
  gdje je \(p\) paralelizabilni udio, \(N_p\) broj procesora 
- **Maksimum** ( \(N_p\to\infty\) ):  
  \(\displaystyle S_\infty = \frac{1}{1-p}\)  
- **Kompleksnost**: O(1) 

### 3.2 Gustafsonov zakon  
- **Pretpostavke**: problem raste s \(N_p\), sekvencijalni dio ostaje konstantan  
- **Formula**:  
$$
    S(N_p) = s + p\,N_p,\quad s + p = 1
$$
  gdje je \(s\) slijedni udio, \(p=1-s\) paralelni udio 
- **Kompleksnost**: O(1)

---

## 4. PRAM model (sinhroni)  

### 4.1 Varijante PRAM-a  
- **EREW**: isključivo čitanje/pisanje  
- **CREW**: konkurentno čitanje, ekskluzivno pisanje  
- **CRCW**: konkurentno pisanje (rješava se slučajnim izborom ili zbrajanjem)
### 4.2 Algoritam reduciranja (reduce)  
- **Pseudokod** (bin. stablo):  
```
for d = 0 to ⌈log₂n⌉-1 parallel:
	for i = 0 to n-1 step 2^(d+1):
	    A[i+2^(d+1)-1] += A[i+2^d-1]
```
- **Vrijeme**: O(log n) uz Θ(n) procesora

### 4.3 Prefiks-suma (scan)

1. **Up-sweep** (isto kao reduce)
2. **Down-sweep**
    - postavi korijen na neutralni element
    - rekurzivno raspodijeli parcijalne sume

- **Vrijeme**: O(log n) uz Θ(n) procesora
### 4.4 Brentovo pravilo

- Svaki PRAM algoritam s vremenom O(log n) može se efektivno izvesti na p procesora u $O\!\bigl(\tfrac{n}{p} + \log p\bigr)$ vremenu

---

## 5. Asinkroni PRAM (aPRAM)

### 5.1 Parametri modela

- **d**: trošak jednog globalnog pristupa memoriji
- **B(p)**: trošak sinkronizacije svih p procesora
- **Lokalna operacija**: 1 jedinica
- **Globalni pristup**: d jedinica
- **Sinkronizacija**: B(p) jedinica

### 5.2 Prilagodba PRAM algoritama

- Svaka PRAM instrukcija → čitanje + računanje + pisanje + ograda
- Ukupna složenost: $O(B + \tfrac{n}{p})\,t_{\rm PRAM}$

---

## 6. MPI – osnove i kolektive

### 6.1 Inicijalizacija i završetak

````c
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
/* … */
MPI_Finalize(); 
````


### 6.2 Osnovne funkcije razmjene poruka  
- **Blokirajuće**:  
 ```C
MPI_Send(buf,count,datatype,dest,tag,comm);
MPI_Recv(buf,count,datatype,source,tag,comm,&status);
```

- **Neblokirajuće**:  
```c
MPI_Isend(..., &request);
MPI_Irecv(..., &request);
MPI_Wait(&request,&status);
```

### 6.3 Probe za asinkroni prijem

````c
MPI_Iprobe(source,tag,comm,&flag,&status);
MPI_Probe(source,tag,comm,&status);
MPI_Get_count(&status,datatype,&count);

### 6.4 Kolektivne operacije (idealne složenosti, drveno stablo)  
| Operacija   | Kratki opis                           | Vrijeme    |
|-------------|---------------------------------------|------------|
| `MPI_Bcast`   | root → svi                             | O(log P)   |
| `MPI_Reduce`  | svi → root (reduce operator)           | O(log P)   |
| `MPI_Scatter` | root distrib. podnizova              | O(log P)   |
| `MPI_Gather`  | svi šalju podnizove → root            | O(log P)   |
| `MPI_Barrier` | globalna sinkronizacija               | O(log P)   | 

