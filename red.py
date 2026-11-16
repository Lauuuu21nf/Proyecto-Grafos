import pandas as pd

LINKS = "9606.protein.links.v12.5.txt.gz"   # red completa humana
INFO = "9606.protein.info.v12.5.txt.gz"     # diccionario ENSP -> símbolo
UMBRAL = 700        # escala 0-1000 del archivo de STRING (700 = confianza alta)
NIVELES = 2         # vecindades de grado 1 y 2
SALIDA = "red_expandida.tsv"

GENES_FA = ['FANCA', 'FANCB', 'FANCC', 'BRCA2', 'FANCD2', 'FANCE', 'FANCF',
            'FANCG', 'FANCI', 'BRIP1', 'FANCL', 'FANCM', 'PALB2', 'RAD51C',
            'SLX4', 'ERCC4', 'RAD51', 'BRCA1', 'UBE2T', 'XRCC2', 'MAD2L2',
            'RFWD3', 'FAAP100']


info = pd.read_csv(INFO, sep='\t')
a_simbolo = dict(zip(info['#string_protein_id'], info['preferred_name']))
print(f"Diccionario: {len(a_simbolo)} proteínas")

# 2. aristas sobre el umbral. El archivo de links va separado por ESPACIOS,
#    no por tabuladores, y se lee por trozos para no agotar la memoria.
trozos = []
for trozo in pd.read_csv(LINKS, sep=' ', chunksize=2_000_000):
    trozos.append(trozo[trozo.combined_score >= UMBRAL])
links = pd.concat(trozos, ignore_index=True)
links['n1'] = links.protein1.map(a_simbolo)
links['n2'] = links.protein2.map(a_simbolo)
links = links.dropna(subset=['n1', 'n2'])
print(f"Aristas con score >= {UMBRAL}: {len(links)}")

# 3. expansión por niveles
semillas = set(GENES_FA) & (set(links.n1) | set(links.n2))
if len(semillas) < len(GENES_FA):
    print(f"  aviso: {sorted(set(GENES_FA) - semillas)} no tienen aristas sobre el umbral")

'''une todos los nodos de la columna n1 y la n2 para saber qué proteínas 
sobrevivieron al filtro de score >= 700 y los intersecta con los 23 genes de
FA. Esto para saber si el filtro fue tan duro que borró uno de los genes
principales.'''

red, frontera = set(semillas), set(semillas)
print(f"nivel 0 (semillas): {len(red)} proteínas")
for nivel in range(1, NIVELES + 1):
    vecinos = (set(links.loc[links.n1.isin(frontera), 'n2']) |
               set(links.loc[links.n2.isin(frontera), 'n1']))
    '''Busca si de alguna semilla hay alguna otra proteina en frontera que
    está en la columna 1 para que le dé su socio de la columna 2. De aquí salen
    todos los vecinos.'''
    frontera = vecinos - red
    red |= frontera
    '''a los vecinos encontrados se les resta los nodos que ya estan en la red
    y se añaden de frontera en red'''
    print(f"nivel {nivel}: +{len(frontera)} -> {len(red)} proteínas")

# 4. subgrafo inducido, en el formato del export web
sub = links[links.n1.isin(red) & links.n2.isin(red)].copy()
sub['combined_score'] = sub.combined_score / 1000.0
sub = sub.rename(columns={'n1': '#node1', 'n2': 'node2'})
sub[['#node1', 'node2', 'combined_score']].to_csv(SALIDA, sep='\t', index=False)

'''vemos en la base donde n1 y n2 pertenecen a red para extraer sus aristas
y los pesos de estas, se normalizan y se exporta'''

n = len(set(sub['#node1']) | set(sub.node2))
m = len({frozenset(e) for e in zip(sub['#node1'], sub.node2)})
print(f"\n{SALIDA}: {n} proteínas, {m} aristas únicas, "
      f"densidad {100 * m / (n * (n - 1) / 2):.2f}%")

'''fórmula de densidad de un grafo no dirigido 2m/(n(n-1))'''
