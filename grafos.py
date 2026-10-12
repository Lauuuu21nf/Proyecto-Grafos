import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Matriz Ponderada
df = pd.read_csv("string_interactions_123_nodes.tsv", sep='\t')
#print(df.columns)
df['peso_normalizado'] = df['combined_score'] / 1000.0
G_completo = nx.from_pandas_edgelist(df, source='#node1', target='node2', edge_attr='peso_normalizado')

componentes = sorted(nx.connected_components(G_completo), key=len, reverse=True)
G = G_completo.subgraph(componentes[0]).copy()
print(f"Topología inicial Componente más grande: {G.number_of_nodes()} proteínas y {G.number_of_edges()} aristas.")

nodos_lista = list(G.nodes())
Ap = nx.to_numpy_array(G, nodelist=nodos_lista, weight='peso_normalizado')
#print(nodos_lista)
#print(Ap)

L_sym, D = csgraph.laplacian(Ap, normed=True, return_diag=True)
'''Usamos solo normed el cual computa el Laplaciano simetrico normalizado, el return_diag es para
un arreglo con los grados de cada vértice para la matriz diagonal de grados de los nodos del 
grafo D. IMPORTANTE: Scipy hace la conversión de la ecuación 1 automáticamente'''
valores_propios, vectores_propios = eigh(L_sym)
#print("Vectores propios", vectores_propios)
#print("Valores propios", valores_propios)
brechas = np.diff(valores_propios)
'''ecuación 5'''
k_optimo = np.argmax(brechas[1:30]) + 2 
'''La posición i que genere el salto más grande es el óptimo de clústeres. Le sumamos 2 porque
borramos el primer valor propio (0) pero necesitamos que estén en sus posiciones correspondientes'''
print(f"Número óptimo de clústeres (k): {k_optimo}")

'''solo me está saliendo el segundo eigenvalor (que es obvio porque es el máximo, pero no el que necesitamos)
entonces matplot al fallo y ver los picos en una gráfica y ver candidatos. ¿Por qué no es el que necesitamos?
porque nos entregaría entonces dos complejos, dos de 60 proteínas o uno de 100 y otro de 20'''

rango_k = range(2, 31)
valores_brechas = brechas[1:30]

plt.figure(figsize=(10, 5))
plt.plot(rango_k, valores_brechas, marker='o', linestyle='-', color='#2ca02c')


plt.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='Máximo Global (Macro-corte)')

plt.title('Análisis de la Brecha Espectral (Eigengap)', fontsize=14)
plt.xlabel('Número de Clústeres (k)', fontsize=12)
plt.ylabel('Magnitud de la Brecha ($\lambda_{i+1} - \lambda_i$)', fontsize=12)
plt.xticks(np.arange(2, 31, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

'''Ahora procede ver en qué candidatos hay mas genes y cuales porque los 23 no van a estar físicamente en el
mismo clúster'''
candidatos_k = [4, 6, 8]


genes_enfermedad = [
    'FANCA', 'FANCB', 'FANCC', 'BRCA2', 'FANCD2', 'FANCE', 'FANCF', 'FANCG', 
    'FANCI', 'BRIP1', 'FANCL', 'FANCM', 'PALB2', 'RAD51C', 'SLX4', 'ERCC4', 
    'RAD51', 'BRCA1', 'UBE2T', 'XRCC2', 'MAD2L2', 'RFWD3', 'FAAP100'
]

for k in candidatos_k:
    print(f"\n Distribución de los 23 genes con k = {k}")
    
    U = vectores_propios[:, :k]
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    '''fijamos el random_state para obtener siempre los mismos clusteres, 42 porque 
    eso dijo stack overflow'''
    etiquetas = kmeans.fit_predict(U)
    '''esto fue idea de gemini y es que primero calcula las coordenadas donde deben quedar los
    k centroides y luego mide la distancia de cada proteina a ese centroide para definir a
    cual pertenece'''
    
    
    diccionario_clusters = {i: [] for i in range(k)}
    for nodo, etiqueta in zip(nodos_lista, etiquetas):
        diccionario_clusters[etiqueta].append(nodo)
    
    # Mostrar en qué clústeres cayeron los genes que buscamos
    for cluster_id, proteinas in diccionario_clusters.items():
        genes_recuperados = [p for p in proteinas if p in genes_enfermedad]
        
        # Solo imprimimos los clústeres que capturaron algún gen patológico
        if len(genes_recuperados) > 0:
            ruido = len(proteinas) - len(genes_recuperados)
            print(f"Clúster {cluster_id + 1} (Total: {len(proteinas)} proteínas | Ruido: {ruido}):")
            print(f"Contiene {len(genes_recuperados)} genes FA: {genes_recuperados}\n")

'''Moraleja: argmax no sirvió para nada, solo para decirnos que habia que graficar'''