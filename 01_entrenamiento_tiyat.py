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
# # Construye Tiyat: un modelo de lenguaje desde cero
#
# [<img src="https://tachiwin.org/tachiwin-badge-sm.png" alt="Tachiwin" height="42">](https://tachiwin.org)
#
# ¡Hola! En este recorrido vas a construir y entrenar **Tiyat**, un modelo de
# lenguaje de Tachiwin. Su nombre significa *suelo* o *tierra*: aquello que sirve de
# base para comenzar a construir.
#
# [Tachiwin](https://tachiwin.org) es una iniciativa indígena, mexicana,
# comunitaria y sin fines de lucro que desarrolla Inteligencia Artificial para las
# lenguas originarias de México. Nació desde el Tutunakú como punto de partida y su
# horizonte comprende las 68 lenguas originarias nacionales y sus aproximadamente
# 360 variantes. El proyecto trabaja desde la soberanía tecnológica, la colaboración
# y el conocimiento abierto: queremos que estas herramientas puedan estudiarse,
# cuestionarse y recrearse, no que funcionen como cajas negras.
#
# Tiyat es un **modelo de lenguaje autoregresivo**. Dicho de manera sencilla: recibe
# un fragmento de texto e intenta adivinar qué token viene después. Luego incorpora
# ese token y vuelve a intentarlo. Al repetir este proceso puede continuar una frase
# en las lenguas y formas de escritura presentes en sus datos de entrenamiento. No
# consulta una enciclopedia ni recupera una respuesta correcta: aprende patrones del
# texto y genera continuaciones estadísticamente plausibles.
#
# ## El recorrido completo
#
# A lo largo del cuaderno vas a seguir el mismo camino que siguen los datos:
#
# 1. **Preparar el entorno** donde se hará el cálculo.
# 2. **Abrir el corpus y el tokenizador** para convertir texto en números.
# 3. **Construir la red neuronal** que aprenderá relaciones entre esos números.
# 4. **Practicar la predicción del siguiente token** miles de veces.
# 5. **Guardar avances** para poder continuar si Colab se desconecta.
# 6. **Exportar el resultado** y usarlo después en el cuaderno de inferencia.
#
# No necesitas conocer JAX ni haber entrenado un LLM. Antes de cada bloque de código
# veremos qué problema resuelve, qué debes esperar al ejecutarlo y cómo se conecta con
# el resto del proceso. Sí conviene reservar tiempo: entrenar desde cero es una tarea
# larga y requiere un acelerador. Una ejecución de pocos pasos sirve para comprobar
# el flujo; obtener un modelo útil requiere muchos datos, cómputo y evaluación junto
# con hablantes.
#
# > Tiyat es experimental. Puede producir texto incorrecto, mezclado o dañino y su
# > desempeño no será igual en todas las lenguas. Una salida fluida no es una prueba
# > de verdad ni una traducción validada.

# %% [markdown]
# ## 0. Prepara el espacio de trabajo
#
# Google Colab nos permite usar una GPU o TPU sin configurar una computadora desde
# cero. Antes de continuar, abre **Entorno de ejecución → Cambiar tipo de entorno de
# ejecución** y elige un acelerador disponible. La primera celda instala las
# bibliotecas que usaremos:
#
# - **JAX y Flax** hacen los cálculos de la red neuronal;
# - **Optax** decide cómo actualizar lo aprendido;
# - **Datasets y Transformers** descargan el corpus y el tokenizador;
# - **huggingface_hub** conserva checkpoints fuera de Colab.
#
# Descomenta y ejecuta la línea siguiente. Si Colab solicita reiniciar la sesión,
# hazlo una vez y vuelve aquí.

# %%
# !pip install -U "jax" "flax" "optax" "datasets" "transformers" "huggingface_hub"

# %% [markdown]
# ## 1. Reúne las decisiones en un solo lugar
#
# Una sesión de entrenamiento tiene dos tipos de decisiones. La **arquitectura**
# describe la forma y capacidad de la red: cuántas capas tiene, cuánto contexto puede
# leer y cuántas representaciones compara a la vez. Los **hiperparámetros** describen
# cómo practicará: cuántos ejemplos verá juntos, cuántas vueltas dará al corpus y qué
# tan grandes serán sus ajustes.
#
# Aquí usamos los valores canónicos de **Tiyat arch1**. Mantenerlos fijos hace que un
# checkpoint creado hoy tenga exactamente la misma forma cuando lo abras mañana. El
# tamaño del vocabulario se leerá del tokenizador, porque cada token necesita su
# propia representación.
#
# Para no perder horas de trabajo, guardaremos checkpoints en tu cuenta de Hugging
# Face. Cambia `HF_USERNAME` por tu usuario. Tu token se solicitará de forma segura
# más adelante: **no lo escribas en el código ni compartas una copia autenticada del
# cuaderno**.

# %%
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx
from flax.nnx import traversals
from datasets import load_dataset
from transformers import AutoTokenizer

# Recursos públicos de Tachiwin.
TOKENIZER_REPO = "tachiwin/tokenizer_64k"
DATASET_REPO = "tachiwin/tiyat-ground-pretrain-m1024"
TEXT_COLUMN = "text"

# Tu espacio privado para checkpoints duraderos.
HF_USERNAME = "TU_USUARIO"
CHECKPOINT_REPO = f"{HF_USERNAME}/tiyat-checkpoints-curso"
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_INTERVAL = 1000

# Valores canónicos de Tiyat arch1 (no modificar si se quieren pesos compatibles).
MAX_LENGTH = 918
EMBED_DIM = 768
NUM_HEADS = 24
FEED_FORWARD_DIM = 3072
NUM_TRANSFORMER_BLOCKS = 12
DROPOUT_RATE = 0.1
TIE_EMBEDDINGS = False  # arch1 usa una proyección de salida independiente.

# Hiperparámetros de entrenamiento. Baja BATCH_SIZE si falta memoria.
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
WARMUP_RATIO = 0.1
SEED = 42

# Para una prueba rápida usa, por ejemplo, 20. None recorre todas las épocas.
MAX_TRAIN_STEPS = None
EXPORT_DIR = Path("tiyat_arch1_export")

# %% [markdown]
# ## 2. Abre una caja fuerte para los avances
#
# Colab puede desconectarse y borrar sus archivos temporales. Un **checkpoint** es una
# fotografía del entrenamiento: contiene lo que Tiyat ha aprendido, el estado del
# optimizador y el punto exacto del recorrido donde debe continuar. Lo guardaremos en
# un repositorio privado de Hugging Face. Si vuelves a ejecutar el cuaderno, buscaremos
# automáticamente la fotografía más reciente.
#
# Ejecuta esta celda y pega un token de Hugging Face con permiso de escritura cuando
# aparezca el cuadro seguro. El token no se incorpora al cuaderno.

# %%
from huggingface_hub import HfApi, create_repo, hf_hub_download, login
from huggingface_hub.errors import EntryNotFoundError

if HF_USERNAME == "TU_USUARIO":
    raise ValueError("Cambia HF_USERNAME por tu usuario de Hugging Face en la configuración.")

login()  # Abre un cuadro seguro en Colab/Jupyter.
create_repo(CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True)
hf_api = HfApi()
print(f"Tus avances se guardarán en https://huggingface.co/{CHECKPOINT_REPO}")

# %% [markdown]
# ## 3. Del texto a los números: tokenizador y corpus
#
# Una red neuronal no recibe letras directamente; recibe números. El **tokenizador**
# divide el texto en piezas llamadas *tokens* y asigna un identificador a cada una.
# Un token puede ser una palabra, una parte de palabra, un signo o incluso un espacio.
# Esto es especialmente importante en un proyecto multilingüe: no debemos asumir que
# todas las lenguas separan o construyen sus palabras de la misma manera.
#
# El **corpus** es la colección de textos con la que Tiyat practicará. Cada ejemplo se
# tokeniza antes de entrar al modelo. Primero descargamos ambos recursos y mostramos
# una muestra para comprobar que la columna esperada existe. Ver una muestra no basta
# para auditar un corpus, pero sí ayuda a detectar una configuración equivocada antes
# de iniciar una ejecución costosa.

# %%
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
if tokenizer.pad_token is None:
    # El relleno solo completa secuencias cortas dentro de un lote; su pérdida se ignora.
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizador: {TOKENIZER_REPO}")
print(f"Tamaño del vocabulario: {tokenizer.vocab_size:,}")
print(f"Token de relleno: {tokenizer.pad_token_id}")

dataset = load_dataset(DATASET_REPO, split="train")
if TEXT_COLUMN not in dataset.column_names:
    raise ValueError(f"No existe la columna {TEXT_COLUMN!r}. Columnas: {dataset.column_names}")

print(f"Filas de entrenamiento: {len(dataset):,}")
print("Ejemplo:", dataset[0][TEXT_COLUMN][:200].replace("\n", " "))

# %% [markdown]
# ## 4. Construye el modelo
#
# Ahora construiremos la red neuronal de Tiyat. No hace falta memorizar cada clase;
# sigue primero el viaje de una frase:
#
# 1. **Embeddings:** cada identificador de token se convierte en una lista de números
#    que el modelo puede ajustar. También añadimos información de posición, porque
#    `casa azul` y `azul casa` contienen tokens parecidos pero no significan lo mismo.
# 2. **Atención:** cada token compara la información disponible en el texto anterior
#    para decidir qué partes son relevantes. Varias cabezas de atención permiten
#    buscar distintos tipos de relación al mismo tiempo.
# 3. **Capas internas:** doce bloques repiten este proceso y refinan la representación.
#    Las conexiones residuales y la normalización ayudan a que la señal y los
#    gradientes recorran una red profunda de forma estable.
# 4. **Predicción:** la capa de salida asigna una puntuación a cada token posible. El
#    token con mayor puntuación no siempre será el elegido durante la generación,
#    pero esas puntuaciones son la materia prima de la predicción.
#
# Tiyat pertenece a la familia de modelos **Transformer de tipo decoder-only**, la
# misma idea general usada por muchos modelos GPT. Su máscara causal funciona como
# una hoja que tapa la continuación correcta: mientras practica una posición, solo
# puede mirar lo que apareció antes. Así aprende la tarea que luego realizará al
# generar texto.

# %%
class TokenAndPositionEmbedding(nnx.Module):
    """Suma la representación del token y la de su posición en la secuencia."""

    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        positions = jnp.arange(token_ids.shape[1])[None, :]
        return self.token_emb(token_ids) + self.pos_emb(positions)


class TransformerBlock(nnx.Module):
    """Atención causal, conexiones residuales y red feed-forward."""

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

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = x + self.dropout1(self.attention(self.ln1(x), mask=mask), deterministic=not train)
        return x + self.dropout2(self.ffn(self.ln2(x)), deterministic=not train)


class MiniGPT(nnx.Module):
    """La implementación mínima de Tiyat arch1."""

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
        # En arch1 no se comparten los pesos de entrada y salida.
        self.output_layer = nnx.Linear(
            config["embed_dim"], config["vocab_size"], use_bias=False, rngs=rngs
        )

    def __call__(self, token_ids: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        sequence_length = token_ids.shape[1]
        causal_mask = jnp.tril(jnp.ones((sequence_length, sequence_length)))
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, causal_mask, train=train)
        return self.output_layer(self.ln_final(x))


MODEL_CONFIG = {
    "maxlen": MAX_LENGTH,
    "vocab_size": tokenizer.vocab_size,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "feed_forward_dim": FEED_FORWARD_DIM,
    "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
    "dropout_rate": DROPOUT_RATE,
    "tie_embeddings": TIE_EMBEDDINGS,
}

# %% [markdown]
# ## 5. Prepara el proceso de aprendizaje
#
# Al crear el modelo, sus millones de parámetros comienzan con valores aleatorios; en
# este momento Tiyat todavía no sabe continuar una frase. Entrenar significa ajustar
# esos valores poco a poco.
#
# Después de cada intento calcularemos el **gradiente**, una señal que indica en qué
# dirección debería cambiar cada parámetro para cometer un poco menos de error. AdamW
# organiza esos cambios. La tasa de aprendizaje controla el tamaño de cada paso: al
# inicio crece lentamente (*warmup*) para evitar cambios bruscos y después disminuye
# para afinar lo aprendido.
#
# La primera ejecución compilada puede tardar varios minutos. No es un bloqueo: JAX
# está preparando una versión optimizada de los cálculos para tu acelerador.

# %%
model = MiniGPT(MODEL_CONFIG, rngs=nnx.Rngs(SEED))
parameter_count = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
print(f"Dispositivos JAX: {jax.devices()}")
print(f"Parámetros entrenables: {parameter_count:,} ({parameter_count / 1e6:.1f} M)")

batches_per_epoch = len(dataset) // BATCH_SIZE
if batches_per_epoch == 0:
    raise ValueError("El conjunto tiene menos filas que BATCH_SIZE.")
planned_steps = batches_per_epoch * NUM_EPOCHS
if MAX_TRAIN_STEPS is not None:
    planned_steps = min(planned_steps, MAX_TRAIN_STEPS)
warmup_steps = max(1, int(planned_steps * WARMUP_RATIO))

learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=warmup_steps,
    decay_steps=max(warmup_steps + 1, planned_steps),
    end_value=1e-5,
)
optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP),
        optax.adamw(learning_rate=learning_rate_schedule, weight_decay=WEIGHT_DECAY),
    ),
    wrt=nnx.Param,
)

# %% [markdown]
# ## 6. Convierte el corpus en ejercicios
#
# Tiyat aprende con un ejercicio muy repetitivo: **predecir el siguiente token**. Si
# la secuencia fuera `la tierra florece`, le mostramos `la` para que prediga `tierra`,
# y `la tierra` para que prediga `florece`. En código podemos evaluar todas esas
# posiciones a la vez desplazando los objetivos una posición:
#
# ```text
# entrada:   la       tierra    florece
# objetivo:  tierra   florece   <fin>
# ```
#
# Agrupamos varios textos en un **lote** para aprovechar el acelerador. Como tienen
# longitudes distintas, agregamos tokens de relleno hasta formar un rectángulo. La
# máscara de pérdida ignora ese relleno: no queremos premiar al modelo por predecir
# espacios artificiales.
#
# La **pérdida** resume el error de las predicciones en un número. Queremos que baje
# con el tiempo, aunque no representa por sí sola la calidad lingüística ni dice si
# todas las lenguas están aprendiendo de manera equilibrada.

# %%
def tokenized_batches(data, batch_size: int, seed: int, start_batch: int = 0):
    """Recrea el mismo orden con la misma semilla y comienza en el lote indicado."""
    shuffled = data.shuffle(seed=seed)
    first_row = start_batch * batch_size
    for start in range(first_row, len(shuffled) - batch_size + 1, batch_size):
        texts = shuffled[start:start + batch_size][TEXT_COLUMN]
        encoded = tokenizer(
            texts, truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="np",
        )
        yield np.asarray(encoded["input_ids"], dtype=np.int32)


def loss_fn(model: MiniGPT, inputs: jnp.ndarray, targets: jnp.ndarray):
    logits = model(inputs, train=True)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    valid_target = (targets != tokenizer.pad_token_id).astype(per_token_loss.dtype)
    return (per_token_loss * valid_target).sum() / jnp.maximum(valid_target.sum(), 1.0)


@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, inputs: jnp.ndarray, targets: jnp.ndarray):
    loss, gradients = nnx.value_and_grad(loss_fn)(model, inputs, targets)
    optimizer.update(model, gradients)
    return loss

# %% [markdown]
# ## 7. Guarda y recupera checkpoints
#
# Un checkpoint debe recordar más que los pesos del modelo. También conserva el
# estado de AdamW —su memoria de actualizaciones anteriores— y la próxima época y
# lote que corresponde leer. De ese modo no repetimos desde el principio ni perdemos
# la dinámica del optimizador.
#
# Cada época usa una semilla conocida para ordenar el corpus. Al restaurar repetimos
# ese mismo orden y saltamos directamente al siguiente lote. El archivo
# `latest_checkpoint.pkl` funciona como un pequeño letrero que apunta a la fotografía
# más reciente.

# %%
def save_checkpoint(model, optimizer, step: int, next_epoch: int, next_batch: int, losses):
    payload = {
        "model_state": nnx.to_pure_dict(nnx.state(model)),
        "optimizer_state": nnx.to_pure_dict(nnx.state(optimizer)),
        "model_config": MODEL_CONFIG,
        "step": step,
        "next_epoch": next_epoch,
        "next_batch": next_batch,
        "losses": losses,
    }
    filename = f"checkpoint_step_{step}.pkl"
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / filename
        pointer_path = Path(directory) / "latest_checkpoint.json"
        with open(checkpoint_path, "wb") as file:
            pickle.dump(payload, file)
        with open(pointer_path, "w", encoding="utf-8") as file:
            json.dump({"filename": filename, "step": step}, file)
        hf_api.upload_file(checkpoint_path, f"checkpoints/{filename}", CHECKPOINT_REPO)
        hf_api.upload_file(pointer_path, "latest_checkpoint.json", CHECKPOINT_REPO)
    print(f"💾 Checkpoint guardado en el paso {step:,}")


def load_latest_checkpoint():
    try:
        pointer_path = hf_hub_download(CHECKPOINT_REPO, "latest_checkpoint.json", force_download=True)
    except EntryNotFoundError:
        print("Todavía no hay checkpoints; comenzaremos desde cero.")
        return None

    # Otros errores (conexión, permisos o archivo dañado) se muestran en vez de
    # comenzar silenciosamente y arriesgar una ejecución anterior.
    with open(pointer_path, encoding="utf-8") as file:
        pointer = json.load(file)
    checkpoint_path = hf_hub_download(
        CHECKPOINT_REPO, f"checkpoints/{pointer['filename']}", force_download=True
    )
    with open(checkpoint_path, "rb") as file:
        return pickle.load(file)


start_epoch = 0
start_batch = 0
step = 0
losses = []

if RESUME_FROM_CHECKPOINT:
    checkpoint = load_latest_checkpoint()
    if checkpoint is not None:
        if checkpoint["model_config"] != MODEL_CONFIG:
            raise ValueError("El checkpoint usa una arquitectura distinta a esta configuración.")
        model_state = nnx.state(model)
        nnx.replace_by_pure_dict(model_state, checkpoint["model_state"])
        nnx.update(model, model_state)
        optimizer_state = nnx.state(optimizer)
        nnx.replace_by_pure_dict(optimizer_state, checkpoint["optimizer_state"])
        nnx.update(optimizer, optimizer_state)
        step = checkpoint["step"]
        start_epoch = checkpoint["next_epoch"]
        start_batch = checkpoint["next_batch"]
        losses = checkpoint.get("losses", [])
        print(f"✅ Continuaremos desde el paso {step:,}, época {start_epoch + 1}, lote {start_batch:,}.")

# %% [markdown]
# ## 8. Entrena Tiyat
#
# Ya está todo conectado. En cada vuelta, Tiyat intenta predecir, mide su error,
# calcula gradientes y actualiza sus parámetros. Verás una línea de progreso cada diez
# pasos y un checkpoint cada `CHECKPOINT_INTERVAL` pasos.
#
# Es normal que la pérdida fluctúe: cada lote contiene textos diferentes. Observa la
# tendencia a lo largo de muchos pasos, no un valor aislado. Una pérdida descendente
# solo confirma que el modelo reconoce mejor los patrones del corpus; no garantiza
# exactitud, seguridad ni una cobertura justa entre lenguas. Esas preguntas requieren
# evaluaciones específicas y participación de hablantes.
#
# Si Colab se desconecta, vuelve a ejecutar las celdas en orden. Esta sección cargará
# el checkpoint más reciente y retomará el siguiente lote. Como el guardado ocurre a
# intervalos, se puede perder únicamente el trabajo posterior al último checkpoint.

# %%
started_at = time.time()
last_saved_step = step
stop_training = MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS
next_epoch, next_batch = start_epoch, start_batch

for epoch in range(start_epoch, NUM_EPOCHS):
    if stop_training:
        break
    first_batch = start_batch if epoch == start_epoch else 0
    batches = tokenized_batches(dataset, BATCH_SIZE, seed=SEED + epoch, start_batch=first_batch)

    for batch_index, batch in enumerate(batches, start=first_batch):
        inputs = jnp.asarray(batch)
        targets = jnp.concatenate(
            [inputs[:, 1:], jnp.full((inputs.shape[0], 1), tokenizer.pad_token_id, dtype=jnp.int32)], axis=1
        )
        loss = float(train_step(model, optimizer, inputs, targets))
        losses.append(loss)
        step += 1

        next_epoch = epoch
        next_batch = batch_index + 1
        if next_batch >= batches_per_epoch:
            next_epoch, next_batch = epoch + 1, 0

        if step == 1 or step % 10 == 0:
            print(f"Época {epoch + 1}/{NUM_EPOCHS} | paso {step}/{planned_steps} | pérdida {loss:.4f}")
        if step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, step, next_epoch, next_batch, losses)
            last_saved_step = step
        if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
            stop_training = True
            break

    start_batch = 0
    if stop_training:
        break

if step > last_saved_step:
    save_checkpoint(model, optimizer, step, next_epoch, next_batch, losses)

print(f"Entrenamiento terminado en {(time.time() - started_at) / 60:.1f} minutos.")
print(f"Última pérdida: {losses[-1]:.4f}" if losses else "No se procesaron lotes nuevos.")

# %% [markdown]
# ## 9. Empaca el modelo para usarlo
#
# Los checkpoints están pensados para **continuar entrenando** y por eso incluyen el
# optimizador. Para **generar texto** solo necesitamos una versión más ligera:
#
# - `flax_model.weights.npz`: lo que aprendió la red;
# - `config.json`: la forma exacta de la arquitectura;
# - los archivos del tokenizador: la correspondencia entre texto y tokens;
# - `training_info.json`: la procedencia básica de esta ejecución.
#
# NPZ es un contenedor sencillo de arreglos NumPy. El estado de Flax está organizado
# como un árbol, así que primero lo aplanamos y damos un nombre completo a cada
# arreglo. Este detalle evita el error de guardar solo objetos contenedores en lugar
# de los pesos reales. El cuaderno `02_inferencia_tiyat.py` reconstruye el árbol.

# %%
def export_npz(model: MiniGPT, tokenizer, config: Dict[str, Any], export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    pure_state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    flat_state = traversals.flatten_mapping(pure_state)
    weights = {"/".join(str(part) for part in path): np.asarray(value) for path, value in flat_state.items()}

    np.savez(export_dir / "flax_model.weights.npz", **weights)
    with open(export_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump({**config, "architecture": "MiniGPT (custom nnx, arch1)"}, file, indent=2)
    with open(export_dir / "training_info.json", "w", encoding="utf-8") as file:
        json.dump({
            "dataset": DATASET_REPO,
            "tokenizer": TOKENIZER_REPO,
            "training_steps": step,
            "last_loss": losses[-1] if losses else None,
        }, file, indent=2)
    tokenizer.save_pretrained(export_dir)

    total_arrays = len(weights)
    total_parameters = sum(array.size for array in weights.values())
    print(f"Exportadas {total_arrays} matrices/arreglos y {total_parameters / 1e6:.1f} M parámetros en {export_dir}/")


export_npz(model, tokenizer, MODEL_CONFIG, EXPORT_DIR)

# %% [markdown]
# ## 10. Guarda y comparte tu creación
#
# ¡Terminaste el recorrido de entrenamiento! El disco de Colab es temporal, así que
# ahora subiremos la exportación a un repositorio de Hugging Face. Allí quedarán juntos
# los pesos, la configuración y el tokenizador; podrás abrirlos desde otro entorno con
# `02_inferencia_tiyat.py` sin depender de esta sesión.
#
# Elige un nombre para `FINAL_MODEL_REPO`. Puedes crear un repositorio privado mientras
# exploras y volverlo público cuando estés listo para compartirlo. Si lo publicas,
# acompáñalo con información sobre el corpus, la licencia, la configuración y los
# límites conocidos. Al usar o mostrar sus resultados, evita asumir que son correctos:
# trata con cuidado y respeto tanto los textos de origen como las continuaciones que
# produce el modelo.

# %%
# Cambia el nombre si quieres guardar otra ejecución por separado.
FINAL_MODEL_REPO = f"{HF_USERNAME}/tiyat-arch1"

create_repo(FINAL_MODEL_REPO, repo_type="model", private=False, exist_ok=True)

readme = f"""---
language:
- multilingual
license: apache-2.0
library_name: flax
---

# Tiyat arch1

Modelo de lenguaje experimental entrenado con el cuaderno público de Tachiwin.

## Archivos

- **Pesos:** `flax_model.weights.npz`
- **Configuración:** `config.json`
- **Tokenizador:** archivos guardados junto al modelo
- **Procedencia:** `training_info.json`

## Uso

Abre `02_inferencia_tiyat.py` y configura `USE_LOCAL_EXPORT = False` junto con el
identificador de este repositorio.

## Nota de uso responsable

Las salidas son continuaciones generadas estadísticamente. Revísalas antes de asumir
que son correctas y trata con cuidado y respeto los textos de origen y sus resultados.
"""
(EXPORT_DIR / "README.md").write_text(readme, encoding="utf-8")

hf_api.upload_folder(
    folder_path=str(EXPORT_DIR),
    repo_id=FINAL_MODEL_REPO,
    repo_type="model",
    commit_message=f"Publicar exportación de Tiyat arch1 (paso {step})",
)
print(f"🎉 Tu modelo está disponible en https://huggingface.co/{FINAL_MODEL_REPO}")
print("En el cuaderno de inferencia, usa USE_LOCAL_EXPORT = False y este identificador.")
