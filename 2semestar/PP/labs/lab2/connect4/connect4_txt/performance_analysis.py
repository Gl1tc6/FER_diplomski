import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Postavke za hrvatska slova
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Učitavanje podataka
def load_data():
    # Podaci iz CSV datoteka
    data_7_tasks = {
        'processors': [1, 2, 3, 4, 5, 6],
        'time_seconds': [3.63, 10.79, 2.71, 2.03, 2.06, 2.04]
    }
    
    data_49_tasks = {
        'processors': [1, 2, 3, 4, 5, 6],
        'time_seconds': [3.60, 10.93, 2.07, 1.79, 1.50, 1.37]
    }
    
    data_343_tasks = {
        'processors': [1, 2, 3, 4, 5, 6],
        'time_seconds': [3.72, 12.35, 2.20, 1.79, 1.59, 1.68]
    }
    
    return data_7_tasks, data_49_tasks, data_343_tasks

# Funkcija za izračun ubrzanja i učinkovitosti
def calculate_metrics(data, baseline_processors=1):
    processors = np.array(data['processors'])
    times = np.array(data['time_seconds'])
    
    # Uzmi sekvencijalno vrijeme (1 procesor) kao baseline
    sequential_time = times[processors == baseline_processors][0]
    
    # Izračunaj ubrzanje (Speedup = T_sequential / T_parallel)
    speedup = sequential_time / times
    
    # Izračunaj učinkovitost (Efficiency = Speedup / P)
    efficiency = speedup / processors
    
    return speedup, efficiency

# Kreiranje tablica rezultata
def create_results_table(data_7, data_49, data_343):
    processors = data_7['processors']
    
    # Izračunaj metrike za sve scenarije
    speedup_7, efficiency_7 = calculate_metrics(data_7)
    speedup_49, efficiency_49 = calculate_metrics(data_49) 
    speedup_343, efficiency_343 = calculate_metrics(data_343)
    
    # Kreiraj DataFrame za tablicu
    results_df = pd.DataFrame({
        'Broj procesora': processors,
        '7 zadataka - Vrijeme (s)': data_7['time_seconds'],
        '7 zadataka - Ubrzanje': np.round(speedup_7, 2),
        '7 zadataka - Učinkovitost': np.round(efficiency_7, 2),
        '49 zadataka - Vrijeme (s)': data_49['time_seconds'],
        '49 zadataka - Ubrzanje': np.round(speedup_49, 2),
        '49 zadataka - Učinkovitost': np.round(efficiency_49, 2),
        '343 zadatka - Vrijeme (s)': data_343['time_seconds'],
        '343 zadatka - Ubrzanje': np.round(speedup_343, 2),
        '343 zadatka - Učinkovitost': np.round(efficiency_343, 2)
    })
    
    return results_df

# Kreiranje grafova
def create_plots(data_7, data_49, data_343):
    # Izračunaj metrike
    speedup_7, efficiency_7 = calculate_metrics(data_7)
    speedup_49, efficiency_49 = calculate_metrics(data_49)
    speedup_343, efficiency_343 = calculate_metrics(data_343)
    
    processors = np.array(data_7['processors'])
    
    # Kreiraj figure s 2 subplota
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graf ubrzanja
    ax1.plot(processors, speedup_7, 'o-', label='7 zadataka', linewidth=2, markersize=8)
    ax1.plot(processors, speedup_49, 's-', label='49 zadataka', linewidth=2, markersize=8)
    ax1.plot(processors, speedup_343, '^-', label='343 zadatka', linewidth=2, markersize=8)
    ax1.plot(processors, processors, '--', color='gray', alpha=0.7, label='Idealno ubrzanje')
    
    ax1.set_xlabel('Broj procesora')
    ax1.set_ylabel('Ubrzanje')
    ax1.set_title('Ubrzanje paralelnog algoritma')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(processors)
    
    # Graf učinkovitosti
    ax2.plot(processors, efficiency_7, 'o-', label='7 zadataka', linewidth=2, markersize=8)
    ax2.plot(processors, efficiency_49, 's-', label='49 zadataka', linewidth=2, markersize=8)
    ax2.plot(processors, efficiency_343, '^-', label='343 zadatka', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Idealna učinkovitost')
    
    ax2.set_xlabel('Broj procesora')
    ax2.set_ylabel('Učinkovitost')
    ax2.set_title('Učinkovitost paralelnog algoritma')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(processors)
    ax2.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('ubrzanje_i_ucinkovitost.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return speedup_7, speedup_49, speedup_343, efficiency_7, efficiency_49, efficiency_343

# Dodatni graf - usporedba vremena izvršavanja
def create_time_comparison_plot(data_7, data_49, data_343):
    processors = np.array(data_7['processors'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(processors, data_7['time_seconds'], 'o-', label='7 zadataka', linewidth=2, markersize=8)
    plt.plot(processors, data_49['time_seconds'], 's-', label='49 zadataka', linewidth=2, markersize=8)
    plt.plot(processors, data_343['time_seconds'], '^-', label='343 zadatka', linewidth=2, markersize=8)
    
    plt.xlabel('Broj procesora')
    plt.ylabel('Vrijeme izvršavanja (s)')
    plt.title('Usporedba vremena izvršavanja za različit broj zadataka')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(processors)
    plt.yscale('log')  # Logaritamska skala zbog velikih razlika
    
    plt.tight_layout()
    plt.savefig('usporedba_vremena.png', dpi=300, bbox_inches='tight')
    plt.show()

# Analiza anomalije s 2 procesora
def analyze_two_processor_anomaly(data_7, data_49, data_343):
    print("=== ANALIZA ANOMALIJE S 2 PROCESORA ===")
    
    datasets = [
        ("7 zadataka", data_7),
        ("49 zadataka", data_49), 
        ("343 zadatka", data_343)
    ]
    
    for name, data in datasets:
        times = data['time_seconds']
        t1 = times[0]  # 1 procesor
        t2 = times[1]  # 2 procesora
        t3 = times[2]  # 3 procesora
        
        print(f"\n{name}:")
        print(f"  1 procesor: {t1:.2f}s")
        print(f"  2 procesora: {t2:.2f}s (povećanje: {((t2/t1-1)*100):.1f}%)")
        print(f"  3 procesora: {t3:.2f}s")
        print(f"  Omjer T2/T1: {t2/t1:.2f}")

# Glavna funkcija
def main():
    # Učitaj podatke
    data_7, data_49, data_343 = load_data()
    
    # Kreiraj tablicu rezultata
    results_table = create_results_table(data_7, data_49, data_343)
    print("=== TABLICA REZULTATA MJERENJA ===")
    print(results_table.to_string(index=False))
    
    # Spremi tablicu u CSV
    results_table.to_csv('rezultati_mjerenja.csv', index=False)
    print("\nTablica spremljena u 'rezultati_mjerenja.csv'")
    
    # Kreiraj glavne grafove
    print("\n=== KREIRANJE GRAFOVA ===")
    speedup_7, speedup_49, speedup_343, eff_7, eff_49, eff_343 = create_plots(data_7, data_49, data_343)
    
    # Kreiraj dodatni graf vremena
    create_time_comparison_plot(data_7, data_49, data_343)
    
    # Analiziraj anomaliju
    analyze_two_processor_anomaly(data_7, data_49, data_343)
    
    # Ispis ključnih metrika
    print("\n=== KLJUČNE METRIKE ===")
    print(f"Maksimalno ubrzanje:")
    print(f"  7 zadataka: {max(speedup_7):.2f}x (s {np.argmax(speedup_7)+1} procesora)")
    print(f"  49 zadataka: {max(speedup_49):.2f}x (s {np.argmax(speedup_49)+1} procesora)")  
    print(f"  343 zadatka: {max(speedup_343):.2f}x (s {np.argmax(speedup_343)+1} procesora)")
    
    print(f"\nNajbolja učinkovitost (osim 1 procesora):")
    print(f"  7 zadataka: {max(eff_7[1:]):.2f} (s {np.argmax(eff_7[1:])+2} procesora)")
    print(f"  49 zadataka: {max(eff_49[1:]):.2f} (s {np.argmax(eff_49[1:])+2} procesora)")
    print(f"  343 zadatka: {max(eff_343[1:]):.2f} (s {np.argmax(eff_343[1:])+2} procesora)")

if __name__ == "__main__":
    main()

# KOMENTAR O UTJECAJU BROJA ZADATAKA NA UBRZANJE I UČINKOVITOST
print("""
=== KOMENTAR O REZULTATIMA ===

Analiza rezultata pokazuje nekoliko ključnih uvida o utjecaju broja zadataka na paralelizaciju:

1. **ANOMALIJA S 2 PROCESORA**
   Sva tri scenarija pokazuju značajno pogoršanje performansi s 2 procesora u odnosu na 1 procesor.
   Ova anomalija može biti uzrokovana:
   - Problemima s load balancing-om (neravnomjerna distribucija posla)
   - Komunikacijskim overhead-om koji nadmašuje prednosti paralelizacije  
   - Sinkronizacijskim problemima između procesa
   - Contention za dijeljene resurse

2. **UTJECAJ ZRNATOSTI ZADATAKA**
   - **7 zadataka (gruba zrnatost)**: Najgore skaliranje zbog nedovoljne paralelizacije
     Maksimalno ubrzanje ~1.8x, što ukazuje na ograničenu mogućnost paralelizacije
   
   - **49 zadataka (srednja zrnatost)**: Najbolje skaliranje i učinkovitost
     Maksimalno ubrzanje ~2.6x s dobrom učinkovitošću na višim brojevima procesora
   
   - **343 zadatka (fina zrnatost)**: Umjereno skaliranje zbog komunikacijskog overhead-a
     Maksimalno ubrzanje ~2.3x, ali s padom učinkovitosti na 6 procesora

3. **KOMUNIKACIJSKI OVERHEAD**
   S povećanjem broja zadataka raste i komunikacijski overhead, što objašnjava:
   - Smanjenje relativnog ubrzanja kod 343 zadatka u odnosu na 49
   - Pad učinkovitosti na većem broju procesora

4. **OPTIMALNA KONFIGURACIJA**
   - Za sve scenarije, optimalan broj procesora je 3-5
   - 49 zadataka pokazuje najbolji balans između paralelizacije i overhead-a
   - 6 procesora često pokazuje pad učinkovitosti zbog overhead-a

5. **AMDAHLOV ZAKON**
   Rezultati su u skladu s Amdahlovim zakonom - postoji serijski dio programa koji 
   ograničava paralelizaciju, osobito izraženo kod 7 zadataka gdje je sekvencijski
   dio proporcionalno veći.

PREPORUKE:
- Izbjeći korištenje 2 procesora zbog anomalije
- Za ovaj tip algoritma, optimalno je koristiti 3-5 procesora
- 49 zadataka predstavlja dobru granularnost za paralelizaciju
- Potrebna je dodatna optimizacija load balancing-a i komunikacije
""")