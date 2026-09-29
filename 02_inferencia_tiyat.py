# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Genera texto con Tiyat: de un prompt a una continuación
#
# [<img src="https://tachiwin.org/tachiwin-badge-sm.png" alt="Tachiwin" height="42">](https://tachiwin.org)
#
# ¡Bienvenido de vuelta! En el cuaderno anterior construiste y entrenaste **Tiyat**,
# el modelo de lenguaje de Tachiwin cuyo nombre significa *suelo* o *tierra*. Ahora
# abriremos el resultado, le daremos el comienzo de un texto y observaremos cómo elige
# cada nuevo token.
#
# [Tachiwin](https://tachiwin.org) es una iniciativa indígena, mexicana,
# comunitaria y sin fines de lucro que desarrolla Inteligencia Artificial en lenguas
# originarias de México. Comenzó desde el Tutunakú y trabaja con la mirada puesta en
# las 68 lenguas originarias nacionales y sus aproximadamente 360 variantes, desde
# principios de soberanía tecnológica, colaboración y conocimiento abierto.
#
# Tiyat no busca una respuesta almacenada ni comprende una pregunta como una persona.
# Recibe un **prompt** —el texto con el que tú comienzas—, calcula qué tokens podrían
# aparecer después y elige uno. La continuación vuelve a entrar al modelo y el ciclo
# se repite. Esa operación sencilla, realizada muchas veces, produce un texto nuevo.
#
# ## El recorrido completo
#
# En este cuaderno vas a:
#
# 1. localizar los pesos y la configuración exportados;
# 2. reconstruir la misma red usada durante el entrenamiento;
# 3. recuperar dentro de ella los valores aprendidos;
# 4. entender cómo se elige el siguiente token;
# 5. probar prompts en distintas lenguas y comparar estrategias.
#
# No necesitas volver a descargar el corpus ni entrenar otra vez. Eso hace que este
# cuaderno sea más rápido y fácil de compartir.
#
# > Tiyat es experimental. Una continuación puede parecer convincente y aun así ser
# > incorrecta, mezclar lenguas o reproducir problemas presentes en los datos. No es
# > una traducción validada ni la voz de una comunidad. Revisa resultados lingüísticos
# > con hablantes, especialmente antes de publicarlos o utilizarlos en una aplicación.

# %% [markdown]
# ## 0. Prepara el espacio de trabajo
#
# Solo necesitamos las bibliotecas para reconstruir la red, leer el tokenizador y
# hacer predicciones. Puedes ejecutar inferencia en CPU, aunque una GPU suele ser más
# rápida. Descomenta la instalación si estás en una sesión nueva de Colab.
#
# Tiyat tiene una arquitectura propia implementada con JAX y Flax. Por eso la
# reconstruimos aquí en vez de pedir a `AutoModel` que suponga cómo está organizada.

# %%
# !pip install -U "jax" "flax" "transformers" "huggingface_hub"

# %% [markdown]
# ## 1. Dile al cuaderno dónde están los archivos
#
# El entrenamiento produjo una pequeña carpeta con todo lo necesario para usar el
# modelo. Puedes cargarla de dos maneras:
#
# 1. **Desde este mismo entorno:** sube o copia la carpeta
#    `tiyat_arch1_export`. Esta opción es privada y no requiere publicar nada.
# 2. **Desde Hugging Face:** cambia `USE_LOCAL_EXPORT` a `False` e indica un
#    repositorio que contenga exactamente los archivos exportados.
#
# Los checkpoints del primer cuaderno no son este paquete: guardan información extra
# para reanudar el entrenamiento. Aquí buscamos la exportación final en NPZ. Si usas
# un repositorio privado, ejecuta `huggingface_hub.login()` de manera interactiva;
# nunca pegues credenciales dentro del código.

# %%
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from flax.nnx import traversals
from transformers import AutoTokenizer

USE_LOCAL_EXPORT = True
LOCAL_EXPORT_DIR = Path("tiyat_arch1_export")
MODEL_REPO = "tachiwin/tiyat_alpha_arch1"  # Solo se usa cuando USE_LOCAL_EXPORT = False.
SEED = 42

# %% [markdown]
# ## 2. Reconstruye la red y devuelve lo aprendido
#
# Imagina la arquitectura como un instrumento y los pesos como la afinación que
# adquirió con la práctica. `config.json` nos dice cómo reconstruir el instrumento:
# cuántas capas, cabezas de atención y dimensiones debe tener. El archivo NPZ contiene
# los millones de valores aprendidos que colocaremos en los lugares correspondientes.
#
# La definición siguiente debe coincidir exactamente con Tiyat arch1. No estamos
# entrenando otra vez: los valores aleatorios con los que nace temporalmente el modelo
# se sustituyen enseguida por los pesos guardados.

# %%
class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        positions = jnp.arange(token_ids.shape[1])[None, :]
        return self.token_emb(token_ids) + self.pos_emb(positions)


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=embed_dim, qkv_features=embed_dim,
            out_features=embed_dim, decode=False, rngs=rngs,
        )
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(embed_dim, ff_dim, rngs=rngs), nnx.gelu,
            nnx.Linear(ff_dim, embed_dim, rngs=rngs),
        )

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = x + self.dropout1(self.attention(self.ln1(x), mask=mask), deterministic=not train)
        return x + self.dropout2(self.ffn(self.ln2(x)), deterministic=not train)


class MiniGPT(nnx.Module):
    def __init__(self, config: Dict[str, Any], *, rngs: nnx.Rngs):
        self.maxlen = config["maxlen"]
        self.embedding = TokenAndPositionEmbedding(
            config["maxlen"], config["vocab_size"], config["embed_dim"], rngs=rngs
        )
        self.transformer_blocks = nnx.List([
            TransformerBlock(config["embed_dim"], config["num_heads"], config["feed_forward_dim"],
                             config["dropout_rate"], rngs=rngs)
            for _ in range(config["num_transformer_blocks"])
        ])
        self.ln_final = nnx.LayerNorm(config["embed_dim"], rngs=rngs)
        # arch1 siempre usa una proyección de salida independiente.
        self.output_layer = nnx.Linear(config["embed_dim"], config["vocab_size"], use_bias=False, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        sequence_length = token_ids.shape[1]
        causal_mask = jnp.tril(jnp.ones((sequence_length, sequence_length)))
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, causal_mask, train=train)
        return self.output_layer(self.ln_final(x))


# %%
def load_local_export(export_dir: Path):
    required = ["config.json", "flax_model.weights.npz"]
    missing = [name for name in required if not (export_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan {missing} en {export_dir}. Ejecuta primero el cuaderno de entrenamiento "
            "o indica un repositorio con esos archivos."
        )
    with open(export_dir / "config.json", encoding="utf-8") as file:
        config = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    weights_file = export_dir / "flax_model.weights.npz"
    return config, tokenizer, weights_file


def load_hub_export(repo_id: str):
    from huggingface_hub import hf_hub_download

    config_file = hf_hub_download(repo_id, "config.json")
    weights_file = hf_hub_download(repo_id, "flax_model.weights.npz")
    with open(config_file, encoding="utf-8") as file:
        config = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return config, tokenizer, Path(weights_file)


if USE_LOCAL_EXPORT:
    model_config, tokenizer, weights_path = load_local_export(LOCAL_EXPORT_DIR)
else:
    model_config, tokenizer, weights_path = load_hub_export(MODEL_REPO)

if model_config.get("tie_embeddings", False):
    raise ValueError("Este cuaderno está diseñado para Tiyat arch1, que no comparte embeddings.")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = MiniGPT(model_config, rngs=nnx.Rngs(SEED))

# NPZ guarda rutas como texto (por ejemplo, "embedding/token_emb/embedding").
# Reconstruimos el árbol anidado que Flax espera y sustituimos el estado inicial.
with np.load(weights_path) as npz:
    flat_weights = {tuple(name.split("/")): jnp.asarray(value) for name, value in npz.items()}
pure_weights = traversals.unflatten_mapping(flat_weights)
state = nnx.state(model)
nnx.replace_by_pure_dict(state, pure_weights)
nnx.update(model, state)

parameter_count = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
print(f"Modelo cargado: {parameter_count / 1e6:.1f} M parámetros")
print(f"Longitud máxima de contexto: {model.maxlen} tokens")

# %% [markdown]
# ## 3. Elige el siguiente token
#
# Para cada posición, Tiyat entrega una puntuación a todos los tokens de su
# vocabulario. Falta decidir cómo convertir esas puntuaciones en una elección:
#
# - con `temperature=0` elegimos siempre el token con mayor puntuación. El resultado
#   es repetible, pero puede atascarse en continuaciones monótonas;
# - una temperatura como `0.7` reparte oportunidades entre varias opciones. Aumentar
#   demasiado el valor vuelve la salida más impredecible;
# - `top_k=40` descarta todas salvo las 40 opciones mejor puntuadas antes de sortear.
#
# La semilla controla el sorteo y permite repetir un experimento. Si mantienes prompt,
# parámetros y semilla, obtendrás la misma elección. Esto es útil para comparar un
# cambio a la vez.

# %%
def sample_next_token(key, logits: jnp.ndarray, temperature: float, top_k: Optional[int]) -> int:
    """Elige un token; temperature=0 significa generación codiciosa (greedy)."""
    if temperature <= 0:
        return int(jnp.argmax(logits))

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        _, top_indices = jax.lax.top_k(logits, k)
        keep = jnp.zeros_like(logits, dtype=bool).at[top_indices].set(True)
        logits = jnp.where(keep, logits, -jnp.inf)

    return int(jax.random.categorical(key, logits))


def generate(
    model: MiniGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    seed: int = SEED,
) -> str:
    """Continúa un prompt sin exceder la ventana de contexto del modelo."""
    generated_ids: List[int] = tokenizer.encode(prompt)
    if not generated_ids:
        raise ValueError("Escriba un prompt que el tokenizador pueda convertir en tokens.")

    key = jax.random.PRNGKey(seed)
    for index in range(max_new_tokens):
        context = generated_ids[-model.maxlen:]
        logits = model(jnp.asarray([context]), train=False)[0, -1, :]
        key, step_key = jax.random.split(key)
        next_token = sample_next_token(step_key, logits, temperature, top_k)
        generated_ids.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# %% [markdown]
# ## 4. Dale a Tiyat un punto de partida
#
# Un prompt no tiene que ser una pregunta: también puede ser el inicio de una oración,
# un título o unas palabras que marquen el contexto. Probaremos el mismo mecanismo con
# textos en distintas lenguas. No esperamos el mismo desempeño en todas ellas: la
# calidad depende, entre otras cosas, de cuánto texto adecuado encontró el modelo
# durante el entrenamiento.
#
# Estas muestras demuestran el flujo técnico; no constituyen una evaluación. Para
# evaluar una lengua hacen falta ejemplos diseñados con cuidado, criterios explícitos
# y revisión de hablantes. Puedes reemplazar la lista por textos cuyo uso sea adecuado.

# %%
PROMPTS = [
    "La tecnología puede apoyar a las comunidades cuando",
    "The future of language technology begins with",
    "kintachiwinkán",
    "Ti' jump'éel áak'ab tin wenele',",
]

for index, prompt in enumerate(PROMPTS):
    print("=" * 72)
    print("Prompt:", prompt)
    print("Generación:")
    print(generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, top_k=40, seed=SEED + index))

# %% [markdown]
# ## 5. Continúa explorando
#
# Prueba primero `temperature=0` y después valores como `0.3` o `0.7`. Conserva el
# prompt y la semilla y cambia una sola variable: así podrás explicar por qué cambió
# el resultado en vez de atribuirlo al azar.
#
# Mantén moderado `max_new_tokens`. Esta versión vuelve a leer el contexto completo
# después de cada token para que el proceso sea visible y fácil de estudiar; no usa
# una caché KV, una optimización común que evita repetir cálculos. Esa sencillez cuesta
# velocidad, pero deja una base clara para tu siguiente experimento.
#
# Ya recorriste el ciclo completo: texto → tokens → puntuaciones → elección → texto.
# Si una salida te sorprende, no preguntes solo “¿suena bien?”. Pregunta también qué
# datos pudieron influir, qué lengua está usando, quién puede evaluarla y qué límites
# debes comunicar. Esa mirada crítica es parte esencial de construir tecnología
# abierta y al servicio de las comunidades.

# %%
# prompt = "Estoy nuevamente aquí sentado a la orilla del tiempo"
# print(generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0, top_k=None))
