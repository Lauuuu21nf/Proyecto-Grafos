# Identificación de Complejos Proteicos en la Anemia de Fanconi mediante el Análisis Espectral de Grafos

Teoría de Grafos 2026-II — Universidad del Rosario.

## Requisitos

```bash
pip install -r requirements.txt
```

## Bases de datos

Descargar de: https://drive.google.com/drive/folders/1xRsZVoTgN_ht2dcrx5sGo0eC-tGvlti4?usp=sharing

Mantener los archivos en la raíz del proyecto, junto a los scripts.

## Orden de ejecución

```bash
python red.py
```
Construye la red expandida a partir de los archivos de STRING y la guarda en `red_expandida.tsv`.

```bash
python consolidar.py
```
Corre el agrupamiento espectral sobre la red y produce el módulo de consenso con sus tablas de resultados.

```bash
python mcl.py
```
Corre Markov Clustering sobre la misma red para contrastar los resultados del agrupamiento espectral.
