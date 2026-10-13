import networkx as nx
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.cluster import KMeans

# la red
df = pd.read_csv("string_interactions_123_nodes.tsv", sep="\t")
G = nx.from_pandas_edgelist(df, "#node1", "node2", edge_attr="combined_score")
nodos = list(G.nodes())
Ap = nx.to_numpy_array(G, nodelist=nodos, weight="combined_score")
valores_propios, vectores_propios = eigh(csgraph.laplacian(Ap, normed=True))


# complejos humanos de corum recortados a las proteinas que si están en la red.
corum = pd.read_csv("corum_allComplexes.txt", sep="\t", low_memory=False)
corum = corum[corum.organism == "Human"]

referencia = []
for _, fila in corum.iterrows():
    subunidades = set(fila.subunits_gene_name.split(";")) & set(nodos)
    if len(subunidades) >= 3:
        # pedimos 3 unidades porque la mitad de corum son pares y un par lo captura
        referencia.append((fila.complex_name, subunidades))

print(f"Red: {len(nodos)} proteínas | Complejos evaluables: {len(referencia)}")


def metricas(S, B):
    # P es que parte del cluster es del complejo y R es que parte del complejo capturamos
    i = len(S & B)
    P, R = i / len(S), i / len(B)
    F = 2 * P * R / (P + R) if i else 0.0
    return P, R, F, i * i / (len(S) * len(B))


def evaluar(clusters):
    # a cada complejo de referencia le busca el clúster que mejor lo recupera.
    filas = []
    for nombre, B in referencia:
        # omega: overlap score, >= 0.25 cuenta como complejo recuperado.
        P, R, F, omega = max((metricas(S, B) for S in clusters), key=lambda m: m[2])
        filas.append(
            {
                "complejo": nombre,
                "subunidades": len(B),
                "P": round(P, 3),
                "R": round(R, 3),
                "F": round(F, 3),
                "omega": round(omega, 3),
                "recuperado": omega >= 0.25,
            }
        )
    return pd.DataFrame(filas).sort_values("F", ascending=False)


def particionar(k):
    etiquetas = KMeans(
        n_clusters=k, init="k-means++", n_init=10, random_state=42
    ).fit_predict(vectores_propios[:, :k])
    return [{p for p, e in zip(nodos, etiquetas) if e == i} for i in range(k)]


print(f"\n{'k':>3} {'F_prom':>7} {'recuperados':>12}")
resultados = {}
for k in [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30]:
    tabla = evaluar(particionar(k))
    resultados[k] = tabla
    print(f"{k:>3} {tabla.F.mean():>7.3f} {tabla.recuperado.sum():>7}/{len(tabla)}")

mejor_k = max(resultados, key=lambda k: resultados[k].F.mean())
tabla = resultados[mejor_k]
tabla.to_csv(f"resultados_corum_k{mejor_k}.csv", index=False)

print(f"\nMejor k = {mejor_k}\n")
print(tabla.head(10).to_string(index=False))
print("\n--- Complejos de Fanconi ---")
print(
    tabla[tabla.complejo.str.contains("Fanconi|FA co", case=False)].to_string(
        index=False
    )
)
