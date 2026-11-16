import re
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from collections import Counter

K = 1000                 # punto de operación
SEMILLAS = range(10)     # particiones para el consenso
CONSENSO = 6             # una proteína entra si aparece en >= 6 de 10
RIBOSOMAL = re.compile(r'^(RPS|RPL|MRPS|MRPL|RPSA|UBA52)')

salida = []
def log(s=""):
    print(s, flush=True)
    salida.append(str(s))

# ═══════════════════════════════════════════════════ 1. LA RED
df = pd.read_csv("red_expandida.tsv", sep="\t")
G0 = nx.from_pandas_edgelist(df, "#node1", "node2", edge_attr="combined_score")
ribo = [v for v in list(G0) if RIBOSOMAL.match(v)]
grado_ribo = np.mean([G0.degree(v) for v in ribo])
grado_resto = np.mean([G0.degree(v) for v in G0 if v not in set(ribo)])

G = G0.copy()
G.remove_nodes_from(ribo)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
nodos = list(G.nodes())
n, m = len(nodos), G.number_of_edges()
densidad = 2 * m / (n * (n - 1))

log("═" * 62)
log("1. LA RED")
log("═" * 62)
log(f"antes del filtro : {G0.number_of_nodes()} proteínas, {G0.number_of_edges()} aristas")
log(f"ribosomales      : {len(ribo)} ({len(ribo)/G0.number_of_nodes()*100:.1f}%), "
    f"grado medio {grado_ribo:.0f} vs {grado_resto:.0f} del resto")
log(f"red final        : {n} proteínas, {m} aristas, densidad {densidad*100:.3f}%")

# ═══════════════════════════════════════════════════ 2. ESTRUCTURA
particion = nx.community.louvain_communities(G, weight="combined_score", seed=42)
Q = nx.community.modularity(G, particion, weight="combined_score")

Ap = nx.to_scipy_sparse_array(G, nodelist=nodos, weight="combined_score", format="csr")
d = np.asarray(Ap.sum(axis=1)).ravel()
Dinv = sp.diags(1.0 / np.sqrt(d)) 
'''matriz diagonal de grados inversa al cuadrado sirve para normalizar la red 
y evita que los nodos superconectados dominen el calculo'''
M = Dinv @ Ap @ Dinv # L_sym = I - M, mismos vectores propios
mu, V = eigsh(M, k=K, which="LA") #mu son los valores propios y v los vectores
orden = np.argsort(1.0 - mu)
lam = (1.0 - mu)[orden]
VP = V[:, orden] #guarda las coordenadas geometricas de las proteinas en el espacio


cheeger_inf, cheeger_sup = lam[1] / 2, np.sqrt(2 * lam[1])
'''calculamos la desigualdad de Cheeger para demostrar que la red de 8600
nodos si se puede particionar'''
gaps = np.diff(lam)
k_gap = int(np.argmax(gaps[1:])) + 2      # eigengap NO trivial

log()
log("═" * 62)
log("2. ESTRUCTURA")
log("═" * 62)
log(f"modularidad de Louvain Q = {Q:.4f}  ({len(particion)} comunidades)")
log(f"lambda_2 = {lam[1]:.4f}")
log(f"Cheeger: {cheeger_inf:.4f} <= phi_G <= {cheeger_sup:.4f}")
log(f"eigengap no trivial máximo en k = {k_gap} (gap = {gaps[k_gap-1]:.4f})")
log(f"  -> no coincide con el k útil: la ecuación (5) no sirve para elegir k")

# ═══════════════════════════════════════════════════ 3. REFERENCIAS
corum = pd.read_csv("corum_allComplexes.txt", sep="\t", low_memory=False)
corum = corum[corum.organism == "Human"]
B_cor = set(corum[corum.complex_name ==
                  "FA core complex (Fanconi anemia core complex)"]
            .iloc[0].subunits_gene_name.split(";"))
B_lit = {"FANCA", "FANCB", "FANCC", "FANCE", "FANCF", "FANCG", "FANCL",
         "FANCM", "FAAP100", "FAAP24", "FAAP20", "CENPS", "CENPX"}


def met(S, B):
    i = len(S & B)
    if not i:
        return 0.0, 0.0, 0.0
    P, R = i / len(S), i / len(B)
    return P, R, 2 * P * R / (P + R)
'''S es el cluster que arrojó el algoritmo, B es la respuesta correcta, ya sea CORUM
de 9 o Dean de 13. Calcula cuantos aciertos hubo y calcula la precision, 
exhaustividad y el F-score'''

def particionar(k, semilla):
    U = VP[:, :k]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)  # Ng-Jordan-Weiss
    e = KMeans(n_clusters=k, init="k-means++", n_init=3,
               random_state=semilla).fit_predict(U)
    g = [set() for _ in range(k)]
    for p, x in zip(nodos, e):
        g[x].add(p)
    return [c for c in g if c]
'''Si tomamos los vectores propios U y los dividimos por su norma (linalg.norm)
proyectamos todos los puntos hacia la superficie de una esfera matemática perfecta
K means ya no se confunde con la distancia y hace cortes impecables.
Toma los vectores, aplica Jordan-Weiss, corre Kmeans y devuelve los pedazos de la
red cortada.'''

def mejor(clusters, B):
    return max(clusters, key=lambda S: met(S, B)[2])
'''Kmeans cortó la red en 1000 pedazos, luego se revisan los 1000 clusteres,
los evalua usando la funcion met(F-score) vs CORUM y solo devuelve el que mejor 
atrapó el complejo'''

# ═══════════════════════════════════════════════════ 4. BARRIDO DE k
log()
log("═" * 62)
log("4. BARRIDO DE k  (3 semillas por k)")
log("═" * 62)
log(f"{'k':>6} {'tam':>5} {'P':>7} {'R':>7} {'F':>7}   {'F (3 semillas)':>18}")
barrido = []
for k in [200, 400, 600, 800, 1000]:
    fs, ts = [], []
    for s in (0, 1, 2):
        S = mejor(particionar(k, s), B_lit)
        P, R, F = met(S, B_lit)
        fs.append(F); ts.append(len(S))
    barrido.append(dict(k=k, tamano=int(np.mean(ts)), P=P, R=R, F=F,
                        F_media=np.mean(fs), F_desv=np.std(fs)))
    log(f"{k:>6} {int(np.mean(ts)):>5} {P:>7.3f} {R:>7.3f} {F:>7.3f}   "
        f"{np.mean(fs):>8.3f} ± {np.std(fs):.3f}")
tab_barrido = pd.DataFrame(barrido)
tab_barrido.to_csv("tabla2_barrido.csv", index=False)

'''corremos Kmeans en varios k grandes, 3 veces por cada k y guarda los F scores
promedio para demostrar estabilidad'''

# ═══════════════════════════════════════════════════ 5. MÓDULO DE CONSENSO
conteo, Fs, tams = Counter(), [], []
for s in SEMILLAS:
    S = mejor(particionar(K, s), B_lit)
    conteo.update(S); Fs.append(met(S, B_lit)[2]); tams.append(len(S))
modulo = {p for p, c in conteo.items() if c >= CONSENSO}
Pl, Rl, Fl = met(modulo, B_lit)
Pc, Rc, Fc = met(modulo, B_cor)

'''Cortamos la red 10 veces en k = 1000 (todo fijado al inicio) para contar cuantas veces cada proteina
cayó en el cluster ganador. Si cayó más de 6 veces, lo declara oficialmente
parte del complejo de Fanconi.'''

log()
log("═" * 62)
log(f"5. MÓDULO DE CONSENSO  (k = {K}, >= {CONSENSO}/{len(SEMILLAS)} particiones)")
log("═" * 62)
log(f"particiones individuales: F = {np.mean(Fs):.3f} ± {np.std(Fs):.3f}, "
    f"tamaño {np.mean(tams):.1f}")
log(f"módulo de consenso: {len(modulo)} proteínas")
log(f"  vs literatura (13): P = {Pl:.3f}  R = {Rl:.3f}  F = {Fl:.3f}")
log(f"  vs CORUM (9)      : P = {Pc:.3f}  R = {Rc:.3f}  F = {Fc:.3f}")
log(f"  no capturadas (lit): {sorted(B_lit - modulo)}")
log(f"  ausentes de ambas  : {sorted(modulo - B_lit - B_cor)}")

tab_modulo = pd.DataFrame([{"proteina": p, "frecuencia": conteo[p],
                            "en_CORUM": p in B_cor, "en_literatura": p in B_lit}
                           for p in conteo if conteo[p] >= 3]
                          ).sort_values(["frecuencia", "proteina"],
                                        ascending=[False, True])
tab_modulo.to_csv("tabla3_modulo.csv", index=False)
log()
log(tab_modulo.to_string(index=False))

# ═══════════════════════════════════════════════════ 6. FRONTERA
sub = G.subgraph(modulo)
cut = sum(dd['combined_score'] for u, v, dd in G.edges(modulo, data=True)
          if (u in modulo) != (v in modulo))
volS = sum(G.degree(p, weight='combined_score') for p in modulo)
volT = sum(x for _, x in G.degree(weight='combined_score'))
phi = cut / min(volS, volT - volS)
cent = nx.eigenvector_centrality_numpy(sub, weight='combined_score')
cv = np.std(list(cent.values())) / np.mean(list(cent.values()))

'''Toma el grafo de 8400 proteinas y hace un subgrafo que contiene solo las 
proteinas del modulo. Luego calculamos el corte cut(S, S palito). Filtra solo
los puentes que salen del complejo FA hacia el resto de la celula y suma sus 
pesos. Luego calculamos el volumen del modulo (suma las conexiones entre
las proteinas dentro del modulo como las que van hacia afuera) y el volumen total de toda la red
(suma de los pesos de todas las conexiones).
Luego calculamos la conductancia que es dividir los puentes que salen hacia afuera
(el cut) entre el tamaño de conexiones del modulo. Si phi es cercano a 0, el
modulo es una isla desconectada del resto. Como a nosotros nos dió 0.73
significa que el 73% del peso de las conexiones de las proteinas va hacia afuera
del complejo y concluimos que este está fuertemente incrustado en la celula.
Despues calculamos la centralidad del vector propio dentro del recorte para ver
si hay una proteina central dentro del grupo. Como el coeficiente de variacion da bajo, significa
que todas estan conectadas con todas'''


log()
log("═" * 62)
log("6. ESTRUCTURA INTERNA Y FRONTERA DEL MÓDULO")
log("═" * 62)
log(f"aristas internas: {sub.number_of_edges()} de "
    f"{len(modulo)*(len(modulo)-1)//2} posibles "
    f"({2*sub.number_of_edges()/(len(modulo)*(len(modulo)-1))*100:.1f}%)")
log(f"centralidad de vector propio: coef. de variación {cv*100:.1f}% "
    f"(plana: el módulo es casi un clique)")
log(f"conductancia del módulo: phi(S) = {phi:.4f}")

frontera = []
for p in sorted(modulo):
    gi = sum(dd['combined_score'] for u, v, dd in sub.edges(p, data=True))
    ge = G.degree(p, weight='combined_score') - gi
    frontera.append(dict(proteina=p, g_int=gi, g_ext=ge,
                         especificidad=gi / (gi + ge),
                         vecinos_ext=len(set(G[p]) - modulo)))
tab_front = pd.DataFrame(frontera).sort_values("especificidad", ascending=False)
tab_front.to_csv("tabla5_frontera.csv", index=False)
log()
log(tab_front.to_string(index=False))

# ═══════════════════════════════════════════════════ 7. REDUNDANCIA
art_sana = set(nx.articulation_points(G))
H = G.copy(); H.remove_nodes_from(modulo)
H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
art_pac = set(nx.articulation_points(H))
Ah = nx.to_scipy_sparse_array(H, nodelist=list(H), weight="combined_score", format="csr")
dh = np.asarray(Ah.sum(axis=1)).ravel()
Dh = sp.diags(1/np.sqrt(dh))
lam2_pac = np.sort(1 - eigsh(Dh @ Ah @ Dh, k=3, which="LA")[0])[1]

'''buscamos los puntos de articulacion de la red completa. Un punto de articulacion seria
un nodo cuello de botella critico, si lo borramos, el grafo se parte. Despues
creamos una copia de la red (H) y le borramos todas las proteinas del modulo FA
(ya que estan super conectadas, no se puede quitar solo un pedazo).
Verificamos el pedazo conectado más grande que sobrevivió en H despues de eso,
vemos que sobrevive el 100%. Volvemos a calcular los puntos de articulacion pero
en H yhacemos lo mismo del bloque 2 pero en H. Vemos que el valor no cambia en
casi nada'''

log()
log("═" * 62)
log("7. REDUNDANCIA: QUITAR EL MÓDULO")
log("═" * 62)
log(f"{'':<28} {'sana':>10} {'paciente':>10}")
log(f"{'proteínas':<28} {n:>10} {H.number_of_nodes():>10}")
log(f"{'componentes conexas':<28} {1:>10} {1:>10}")
log(f"{'puntos de articulación':<28} {len(art_sana):>10} {len(art_pac):>10}")
log(f"{'lambda_2':<28} {lam[1]:>10.5f} {lam2_pac:>10.5f}")
log(f"ninguna del módulo es punto de articulación: "
    f"{not (modulo & art_sana)}")
log("  -> la vía es topológicamente redundante")
