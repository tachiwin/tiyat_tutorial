![Tiyat](assets/tiyat-logo.png)

# Tiyat

**Tiyat** es un modelo de lenguaje autoregresivo desarrollado por [Tachiwin](https://tachiwin.org). Su nombre significa *suelo* o *tierra*: una base desde la cual aprender, experimentar y construir.

Tachiwin es una iniciativa indígena, mexicana, comunitaria y sin fines de lucro que desarrolla Inteligencia Artificial para las lenguas originarias de México. El proyecto comenzó desde el Tutunakú como punto de partida y tiene como horizonte las 68 lenguas originarias nacionales y sus aproximadamente 360 variantes.

Este repositorio contiene un recorrido reproducible para construir, entrenar, guardar, publicar y utilizar Tiyat desde Google Colab o un entorno Jupyter.

## Notebooks

Los notebooks están disponibles en formato Jupyter, listos para abrir directamente desde GitHub o descargar y ejecutar:

1. **[Entrenar Tiyat desde cero](01_entrenamiento_tiyat.ipynb)**  
   Presenta el proyecto y recorre la preparación del entorno, el tokenizador, los datos, la arquitectura, el entrenamiento, los checkpoints, la reanudación y la publicación del modelo en Hugging Face.

2. **[Generar texto con Tiyat](02_inferencia_tiyat.ipynb)**  
   Carga los pesos publicados, reconstruye la arquitectura y muestra cómo generar continuaciones en distintas lenguas.

También conservamos las fuentes Jupytext para facilitar la revisión de cambios:

- [`01_entrenamiento_tiyat.py`](01_entrenamiento_tiyat.py)
- [`02_inferencia_tiyat.py`](02_inferencia_tiyat.py)

## Arquitectura

Tiyat arch1 utiliza una arquitectura **Transformer decoder-only** implementada desde cero con JAX y Flax/NNX.

| Característica | Valor |
| --- | --- |
| Contexto máximo | 918 tokens |
| Tokenizador | [`tachiwin/tokenizer_64k`](https://huggingface.co/tachiwin/tokenizer_64k) |
| Dimensión de embeddings | 768 |
| Bloques Transformer | 12 |
| Cabezas de atención | 24 |
| Dimensión feed-forward | 3072 |
| Dropout | 0.1 |
| Embeddings de entrada y salida | No compartidos |
| Framework | JAX, Flax/NNX y Optax |
| Formato de exportación | NPZ |

La arquitectura incluye embeddings de tokens y posiciones, atención causal multi-head, conexiones residuales, normalización por capas y una proyección de salida independiente.

## Datos y recursos

La configuración predeterminada utiliza:

- **Corpus:** [`tachiwin/tiyat-ground-pretrain-m1024`](https://huggingface.co/datasets/tachiwin/tiyat-ground-pretrain-m1024)
- **Tokenizador:** [`tachiwin/tokenizer_64k`](https://huggingface.co/tachiwin/tokenizer_64k)
- **Checkpoints y publicación:** [Hugging Face Hub](https://huggingface.co/)

Los notebooks permiten cambiar los repositorios y parámetros desde una sección de configuración inicial.

## Uso rápido

1. Abre [`01_entrenamiento_tiyat.ipynb`](01_entrenamiento_tiyat.ipynb) en GitHub, Jupyter o Google Colab.
2. Configura tu usuario de Hugging Face.
3. Ejecuta las celdas en orden para entrenar y publicar tu modelo.
4. Abre [`02_inferencia_tiyat.ipynb`](02_inferencia_tiyat.ipynb).
5. Indica el repositorio publicado y genera tus primeras continuaciones.

El entrenamiento guarda checkpoints privados en Hugging Face para poder continuar después de una interrupción. Al finalizar, el notebook crea un repositorio de modelo con los pesos, la configuración, el tokenizador y la información del entrenamiento.

## Estructura del repositorio

```text
.
├── 01_entrenamiento_tiyat.ipynb   # Notebook de entrenamiento
├── 01_entrenamiento_tiyat.py      # Fuente Jupytext
├── 02_inferencia_tiyat.ipynb      # Notebook de inferencia
├── 02_inferencia_tiyat.py         # Fuente Jupytext
├── assets/
│   └── tiyat-logo.png             # Identidad gráfica de Tiyat
└── README.md
```

## Licencia

Este proyecto se distribuye bajo la licencia **Apache License 2.0**. El archivo `LICENSE` será añadido al repositorio.

Los conjuntos de datos, tokenizadores y modelos enlazados pueden tener condiciones propias; consulta sus respectivos repositorios.

## Apoya a Tachiwin

Tachiwin es un proyecto comunitario y sin fines de lucro. Tu apoyo ayuda a continuar el desarrollo abierto de tecnologías para las lenguas originarias de México.

- [Haz una donación a Tachiwin](https://tachiwin.org/donate)
- [Conoce más sobre Tachiwin](https://tachiwin.org)

[![Tachiwin](https://tachiwin.org/tachiwin-badge-xsm.png)](https://tachiwin.org)
