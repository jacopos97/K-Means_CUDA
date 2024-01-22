# Implementazione dell'algoritmo k-means in CUDA

Questo repository contiene un'implementazione dell'algoritmo k-means in CUDA.

## Parametri di configurazione
I parametri che si possono variare per ottenere diverse configurazioni sono:

- **POINTS_NUMBER:** Numero di punti del dataset. Le scelte possibili sono 4000, 40000, 400000;
   
- **CLUSTER_NUMBER:** Indica il valore K dell'algoritmo K-means. I valori possibili sono 2, 4, 8, 16;
   
- **ITERATION_NUMBER:** Numero di iterazioni desiderate per un'esecuzione di k-means.

- **THREADS_PER_BLOCK:** Numero di threads per blocco.
