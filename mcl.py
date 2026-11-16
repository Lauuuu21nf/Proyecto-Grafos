import re
import numpy as np
import pandas as pd
import networkx as nx

RED = "db/red_expandida.tsv"
SEMILLA_LOUVAIN = 42

# Red
# Filtramos las proteinas ribosomales para borrarlas,
# luego revisamos si la componente sigue conexa para evitar lios despues
RIBOSOMAL = re.compile(r'^(RPS|RPL|MRPS|MRPL|RPSA|UBA52)')

df = pd.read_csv(RED, sep='\t')
G = nx.from_pandas_edgelist(df, '#node1', 'node2', edge_attr='combined_score')
n_antes = G.number_of_nodes()
G.remove_nodes_from([v for v in list(G) if RIBOSOMAL.match(v)])
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print(f"Red: {n_antes} -> {G.number_of_nodes()} proteínas "
      f"(se quitaron {n_antes - G.number_of_nodes()} ribosomales)")

# Para no analizar una matriz 8427x8427 Louvain parte esas proteinas en comunidades buscando
# al menos una de las siete proteinas FA,
# al final devuelve una comunidad de unicamente 1034 proteinas
comunidades = nx.community.louvain_communities(G, weight='combined_score',
                                               seed=SEMILLA_LOUVAIN)
FA = {'FANCA', 'FANCB', 'FANCC', 'FANCE', 'FANCF', 'FANCG', 'FANCL'}
C = G.subgraph(next(c for c in comunidades if FA & c)).copy()
nodos = list(C.nodes())
print(f"Comunidad funcional: {len(nodos)} proteínas, {C.number_of_edges()} aristas")

# las referencias
corum = pd.read_csv("db/corum_allComplexes.txt", sep='\t', low_memory=False)
corum = corum[corum.organism == "Human"]
B_corum = set(corum[corum.complex_name ==
                    "FA core complex (Fanconi anemia core complex)"]
              .iloc[0].subunits_gene_name.split(';'))
B_lit = {"FANCA", "FANCB", "FANCC", "FANCE", "FANCF", "FANCG", "FANCL",
         "FANCM", "FAAP100", "FAAP24", "FAAP20", "CENPS", "CENPX"}

# S es lo encontrado segun la referencia B
def metricas(S, B):
    #P = qué parte del clúster es del complejo.
    #R = qué parte del complejo capturamos.
    #F = media armónica, castiga los extremos.
    i = len(S & B) # i es cuantas comparten
    if not i:
        return 0.0, 0.0, 0.0 # Si no comparten ninguna se evita dividir por cero
    P, R = i / len(S), i / len(B)
    return P, R, 2 * P * R / (P + R)


# MCL
def mcl(A, inflacion, expansion=2, iteraciones=100, poda=1e-5):
    # Ponemos un 1 en la diagonal, sin él la caminata no puede "quedarse quieta" y MCL no
    # distingue el centro de un clúster de su periferia. Tiene que valer lo
    # mismo que una arista fuerte: si vale mucho más, la caminata nunca se
    # mueve y todo colapsa en un solo clúster.
    A = A.copy()
    np.fill_diagonal(A, 1.0)

    # Matriz estocástica por columnas: la columna j son las probabilidades
    # de saltar desde j hacia cada vecino.
    M = A / A.sum(axis=0)

    for i in range(iteraciones):
        previo = M

        M = np.linalg.matrix_power(M, expansion)   # expansión: mezcla (ver los vecinos de los vecinos)
        M = M ** inflacion                          # inflación: separa
        M = M / M.sum(axis=0)                       # renormalizar

        # Poda: la inflación deja muchísimas entradas en valores como 1e-300.
        # No son cero pero no significan nada, y si no se podan la extracción
        # de clústeres los cuenta como conexiones reales.
        M[M < poda] = 0.0
        M = M / M.sum(axis=0)

        if np.allclose(M, previo, atol=poda):
            return M, i + 1
    return M, iteraciones


def extraer_clusters(M, nodos, poda=1e-5):
    # Recorre las filas buscando entradas significativas y sus columnas no nulas serian los miembros
    clusters, ya_asignados = [], set()
    for i in range(len(nodos)):
        miembros = [j for j in np.nonzero(M[i, :] > poda)[0]
                    if j not in ya_asignados]
        if miembros:
            clusters.append({nodos[j] for j in miembros})
            ya_asignados.update(miembros)
    return clusters


# barrido de la inflación
# r controla la granularidad: r bajo da pocos clústeres grandes, r alto da
# muchos clústeres pequeños. Es el parámetro análogo a la k del espectral
A = nx.to_numpy_array(C, nodelist=nodos, weight='combined_score')

print(f"\n{'r':>5} {'iter':>5} {'clústeres':>10} {'tam.FA':>7} "
      f"{'P':>7} {'R':>7} {'F':>7}")
filas = []
for r in [1.4, 1.8, 2.2, 2.6, 3.0, 4.0, 5.0, 6.0]:
    M, it = mcl(A, inflacion=r)
    clusters = extraer_clusters(M, nodos)
    S = max(clusters, key=lambda c: metricas(c, B_lit)[2]) #importante nos qudamos con el que mejor
                                                           # recupera el complejo
    P, R, F = metricas(S, B_lit)
    Fc = metricas(S, B_corum)[2]
    filas.append(dict(r=r, iteraciones=it, clusteres=len(clusters),
                      tamano=len(S), P=P, R=R, F=F, F_corum=Fc))
    print(f"{r:>5.1f} {it:>5} {len(clusters):>10} {len(S):>7} "
          f"{P:>7.3f} {R:>7.3f} {F:>7.3f}")

tabla = pd.DataFrame(filas)
tabla.to_csv("mcl_barrido.csv", index=False)


# Luego de saber cual es el mejor r se vuelve a correr MCL con ese r

mejor = tabla.loc[tabla.F.idxmax()]
M, _ = mcl(A, inflacion=float(mejor.r))
S = max(extraer_clusters(M, nodos), key=lambda c: metricas(c, B_lit)[2])
print(f"\n{'=' * 58}\nMEJOR: r = {mejor.r}  ->  {len(S)} proteínas\n{'=' * 58}")
print(f"módulo: {sorted(S)}")
print(f"del complejo : {sorted(S & B_lit)}")
print(f"otras        : {sorted(S - B_lit)}")
print(f"no capturadas: {sorted(B_lit - S)}")
pd.DataFrame({"proteina": sorted(S),
              "en_CORUM": [p in B_corum for p in sorted(S)],
              "en_literatura": [p in B_lit for p in sorted(S)]}
             ).to_csv("mcl_modulo.csv", index=False)
print("\n-> mcl_barrido.csv, mcl_modulo.csv")
