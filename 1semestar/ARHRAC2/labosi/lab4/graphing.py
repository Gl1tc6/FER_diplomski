import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj CSV podatke
df = pd.read_csv('benchmark_results.csv')

# Graf ukupnog vremena CPU vs GPU
plt.figure(figsize=(10, 6))
plt.plot(df['Dimenzija'], df['CPU vrijeme (ms)'], 'o-', label='CPU')
plt.plot(df['Dimenzija'], df['GPU ukupno (ms)'], 'o-', label='GPU')
plt.xlabel('Dimenzija matrice')
plt.ylabel('Vrijeme (ms)')
plt.title('Usporedba vremena izvođenja CPU vs GPU')
plt.legend()
plt.grid(True)
plt.savefig('cpu_vs_gpu.png')
plt.close()

# Graf raspodjele vremena za GPU implementaciju
plt.figure(figsize=(10, 6))
plt.stackplot(df['Dimenzija'], 
             [df['GPU→RAM (ms)'], df['Računanje (ms)'], df['RAM→GPU (ms)']],
             labels=['Prijenos GPU→RAM', 'Računanje', 'Prijenos RAM→GPU'])
plt.xlabel('Dimenzija matrice')
plt.ylabel('Vrijeme (ms)')
plt.title('Raspodjela vremena za GPU implementaciju')
plt.legend()
plt.grid(True)
plt.savefig('gpu_breakdown.png')
plt.close()