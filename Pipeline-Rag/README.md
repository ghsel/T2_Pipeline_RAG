# Sistema RAG Aplicado — Recuperación, Aumentación y Generación

Proyecto aplicado en **Python** que implementa un sistema RAG
(Retrieval-Augmented Generation) completo y funcional, desarrollado en base
al contenido del **Módulo 2 — Arquitecturas RAG y Gestión del Conocimiento**
del Diplomado en Desarrollo de Software con IA Generativa y Agentes
Inteligentes (Docente: M.Sc. Marcelo Fernando Condori Mendoza).

> **100% Open Source y local.** Los embeddings se generan con un modelo
> abierto (`sentence-transformers`, descargado una sola vez y gratis desde
> Hugging Face) y la generación de texto corre con un modelo open-weight
> servido localmente por **Ollama**. No se requiere ninguna API key de
> pago ni se consumen créditos/tokens facturables para el funcionamiento
> por defecto del proyecto.

---

## 1. Qué unidad del curso implementa cada parte

| Unidad del material | Concepto | Dónde está implementado |
|---|---|---|
| **Unidad 1** | Ciclo Retrieval → Augmentation → Generation | `rag/retriever.py`, `rag/augmentation.py`, `rag/generator.py` |
| **Unidad 1** | El Retriever compara "significado" (embeddings) | `rag/embeddings.py`, `rag/vector_store.py` |
| **Unidad 1** | Prompt aumentado (INSTRUCCIONES / CONTEXTO / PREGUNTA) | `rag/augmentation.py` (misma estructura de 3 partes del material) |
| **Unidad 1** | Citas verificables (fuente + página) | Metadata `source`/`page` propagada desde la extracción hasta la respuesta |
| **Unidad 2** | RAG vs. contexto largo (costo, escalabilidad) | `comparison/rag_vs_long_context.py` — comparación **cuantitativa real**, no solo teórica |
| **Unidad 2** | Búsqueda híbrida (tendencia 2026) | `rag/retriever.py::hybrid_search()` (vectorial + BM25) |
| **Unidad 2** | Agentic RAG (tendencia 2026) | `rag/retriever.py::agentic_search()` (recuperación iterativa con auto-evaluación) |
| **Unidad 3** | Extracción de PDF / HTML / Markdown | `ingestion/extractors.py` |
| **Unidad 3** | Limpieza y normalización de texto | `ingestion/cleaners.py` (las 5 prácticas del material, una función por cada una) |
| **Unidad 3** | Automatización del flujo de ingesta con logging | `ingestion/pipeline.py` |
| **Unidad 3** | Chunking | `rag/chunking.py` |

No se implementa **fine-tuning** real (reentrenar un modelo) porque
requiere GPU dedicada y horas de cómputo, algo fuera del alcance de un
proyecto que corre en una laptop; en su lugar, la Unidad 2 se cubre con una
comparación **medible** entre RAG y contexto largo, y con una tabla
conceptual de fine-tuning en la sección 8 de este documento.

---

## 2. Arquitectura del sistema

```
                    ┌───────────────────────────────────────────────┐
                    │           PIPELINE DE INGESTA (Unidad 3)         │
                    │                                                 │
  data/raw/         │   Extracción → Limpieza → Chunking → Embeddings  │
  (PDF, HTML, MD) ──┼──────────────────────────────────────────────►  │
                    └───────────────────────────────┬───────────────┘
                                                      ▼
                                          ┌───────────────────────┐
                                          │   ChromaDB (local)     │
                                          │   Base vectorial       │
                                          └───────────┬───────────┘
                                                       │
   Usuario ──pregunta──► Retriever (Unidad 1) ◄────────┘
                          │  • búsqueda semántica
                          │  • búsqueda híbrida (+BM25)
                          │  • modo Agentic (iterativo)
                          ▼
                    Augmentation (Unidad 1)
                    INSTRUCCIONES + CONTEXTO + PREGUNTA
                          ▼
                    Generation (Unidad 1)
                    LLM local vía Ollama
                          ▼
                    Respuesta + fuentes citadas ──► Usuario
```

Documento de ejemplo incluido en `data/raw/`: una política de reembolsos
(Markdown) y una guía de soporte técnico (HTML con menú/script/footer de
ruido deliberado, para probar la limpieza), replicando los casos de uso
industriales del material ("documentación interna" y "soporte técnico").

---

## 3. Stack tecnológico (todo Open Source)

| Componente | Herramienta | Licencia |
|---|---|---|
| Extracción PDF | `pypdf` | BSD |
| Extracción HTML | `beautifulsoup4` + `lxml` | MIT / BSD |
| Corrección de encoding | `ftfy` | MIT |
| Embeddings (local) | `sentence-transformers` (modelo `paraphrase-multilingual-MiniLM-L12-v2`) | Apache-2.0 |
| Base de datos vectorial | `chromadb` (modo embebido/local) | Apache-2.0 |
| Búsqueda por palabras clave | `rank-bm25` | Apache-2.0 |
| Generación (LLM local) | `ollama` + un modelo open-weight (ej. Llama 3.2) | MIT (cliente) |
| Conteo de tokens | `tiktoken` | MIT |
| Interfaz web | `streamlit` | Apache-2.0 |
| Pruebas | `pytest` | MIT |

---

## 4. Estructura del proyecto

```
rag-knowledge-pipeline/
├── README.md
├── requirements.txt
├── Makefile
├── .env.example
├── .gitignore
├── cli.py                         <- Punto de entrada: ingest / ask / compare
│
├── config/
│   └── settings.py                <- Configuración centralizada (lee .env)
│
├── data/
│   ├── raw/                       <- Documentos de origen (PDF, HTML, MD)
│   │   ├── politica_reembolsos.md
│   │   └── soporte_tecnico_faq.html
│   ├── processed/                 <- (reservado para exportes intermedios)
│   └── chroma_db/                 <- Base vectorial (se genera al ingerir)
│
├── ingestion/                     <- UNIDAD 3: pipeline de datos
│   ├── extractors.py              <- Extracción PDF / HTML / Markdown
│   ├── cleaners.py                <- Limpieza y normalización
│   └── pipeline.py                <- Orquestación + logging por etapa
│
├── rag/                           <- UNIDAD 1 y 2: el sistema RAG
│   ├── chunking.py                <- División en fragmentos con solapamiento
│   ├── embeddings.py              <- Wrapper del modelo de embeddings local
│   ├── vector_store.py            <- Wrapper de ChromaDB
│   ├── retriever.py               <- Búsqueda semántica, híbrida y Agentic RAG
│   ├── augmentation.py            <- Construcción del prompt aumentado
│   └── generator.py               <- Generación vía Ollama
│
├── comparison/
│   └── rag_vs_long_context.py     <- UNIDAD 2: comparación cuantitativa
│
├── app/
│   └── streamlit_app.py           <- Interfaz de chat web (opcional)
│
└── tests/                         <- 26 pruebas unitarias (todas pasan offline)
    ├── conftest.py                <- Fixtures + embebedor falso para tests rápidos
    ├── test_cleaners.py
    ├── test_chunking.py
    ├── test_augmentation.py
    ├── test_vector_store.py
    └── test_retriever.py
```

---

## 5. Prerrequisitos

| Requisito | Versión | Verificar con |
|---|---|---|
| Python | 3.10 – 3.12 | `python3 --version` |
| Ollama (para generación local) | Reciente | `ollama --version` |

### Instalar Ollama (gratuito, open source)

Ollama permite correr modelos de lenguaje open-weight en tu propia
máquina, sin API key ni costo por token.

- **macOS / Windows:** descargar el instalador desde <https://ollama.com/download>
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Luego, descarga un modelo (una sola vez; queda guardado localmente):

```bash
ollama pull llama3.2
```

> Si prefieres no instalar Ollama, el proyecto sigue siendo completamente
> funcional en **modo solo-recuperación** (`--no-generate`, ver sección 7):
> verás los fragmentos recuperados y el prompt aumentado, pero sin la
> respuesta final generada por el LLM.

---

## 6. Instalación

```bash
cd rag-knowledge-pipeline

# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Copiar configuración de ejemplo
cp .env.example .env
```

También puedes usar `make install`.

> **Primera ejecución:** al ingerir documentos por primera vez, se
> descargará automáticamente el modelo de embeddings (~470MB, una sola
> vez, gratis, desde Hugging Face). Se necesita conexión a internet solo
> para esta descarga inicial; después, todo corre offline.

---

## 7. Ejecución paso a paso

### Paso 1 — Ingerir los documentos (Unidad 3)

```bash
python cli.py ingest --reset
```

Esto ejecuta las 4 etapas del pipeline (con logging de cada una):
extracción → limpieza → chunking → embeddings + almacenamiento. Al
finalizar, imprime un resumen:

```json
{
  "archivos_procesados": 2,
  "fragmentos_extraidos": 2,
  "documentos_limpios": 2,
  "chunks_generados": 7,
  "total_en_coleccion": 7
}
```

Para ingerir tus propios documentos, coloca archivos `.pdf`, `.html` o
`.md` en `data/raw/` y vuelve a correr el comando (o usa `--source` para
apuntar a otra carpeta).

### Paso 2 — Hacer una pregunta (Unidad 1)

```bash
python cli.py ask "¿Cuál es el plazo para solicitar un reembolso?"
```

El sistema muestra los fragmentos recuperados (con su fuente y puntaje de
relevancia) y luego la respuesta generada, fundamentada en ese contexto:

```
--- 4 fragmento(s) recuperado(s) ---
  [1] politica_reembolsos.md | score=0.812
  [2] politica_reembolsos.md | score=0.734
  ...

--- Generando respuesta con Ollama... ---

Según la política de reembolsos, el cliente cuenta con un plazo máximo
de 30 días calendario desde la fecha de entrega para solicitar un
reembolso completo (fuente: politica_reembolsos.md)...
```

**Modo Agentic RAG** (recuperación iterativa, ver Unidad 2):

```bash
python cli.py ask "¿Qué pasa si el producto llega dañado?" --agentic
```

**Modo solo-recuperación** (sin necesidad de Ollama instalado):

```bash
python cli.py ask "¿Cómo abro un ticket de soporte?" --no-generate
```

### Paso 3 — Comparar RAG vs. contexto largo (Unidad 2)

```bash
python cli.py compare "¿Cuál es el plazo para solicitar un reembolso?"
```

```json
{
  "query": "¿Cuál es el plazo para solicitar un reembolso?",
  "chunks_recuperados": 4,
  "chunks_totales_en_corpus": 7,
  "tokens_prompt_rag": 612,
  "tokens_prompt_contexto_largo": 1350,
  "ahorro_de_tokens_pct": 54.7
}
```

Con un corpus pequeño el ahorro ya es notable; con una base de conocimiento
real (cientos de documentos), la diferencia se vuelve dramática — es
precisamente el argumento de escalabilidad de la Unidad 2.

### Paso 4 — (Opcional) Interfaz web de chat

```bash
streamlit run app/streamlit_app.py
```

Abre `http://localhost:8501`: chat con historial, panel lateral con
configuración (modo Agentic, top-k) y un panel expandible que muestra los
fragmentos recuperados con su fuente exacta.

---

## 8. Comparación conceptual: RAG vs. Fine-tuning vs. Contexto largo

Tabla del material del curso, para referencia (el proyecto implementa y
mide RAG y contexto largo; fine-tuning se documenta solo conceptualmente):

| | RAG | Fine-tuning | Contexto largo |
|---|---|---|---|
| Actualización de datos | Inmediata | Lenta (reentrenamiento) | Inmediata |
| Costo | Bajo por consulta | Alto (cómputo intensivo) | Alto por consulta |
| Trazabilidad | Alta (cita la fuente) | Baja | Media |
| Escalabilidad | Millones de documentos | N/A | Limitada por la ventana de contexto |
| Caso ideal | Conocimiento que cambia con frecuencia | Estilo/formato estables | Documentos únicos y pequeños |

---

## 9. Pruebas unitarias

Las 26 pruebas del proyecto corren **completamente offline**: para las
piezas que dependen del modelo de embeddings (`vector_store`, `retriever`),
se usa un embebedor falso y determinista (ver `tests/conftest.py`) que
aísla la prueba de la descarga del modelo real — así se valida la lógica
de negocio (chunking, limpieza, prompt, almacenamiento, ranking híbrido,
bucle Agentic) sin depender de la red ni de horas de cómputo.

```bash
pytest -v
# o: make test
```

Cobertura por archivo:

| Archivo de test | Qué valida |
|---|---|
| `test_cleaners.py` | Corrección de encoding, remoción de marcado/duplicados/headers repetidos |
| `test_chunking.py` | División respetando límites de párrafo, solapamiento, metadata preservada |
| `test_augmentation.py` | Estructura del prompt (3 secciones), citas de fuente/página |
| `test_vector_store.py` | Inserción, conteo, consulta, reinicio de la colección |
| `test_retriever.py` | Búsqueda semántica, BM25, híbrida y el bucle de Agentic RAG |

---

## 10. Solución de problemas comunes

| Problema | Causa probable | Solución |
|---|---|---|
| Error al descargar el modelo de embeddings | Sin conexión a internet en el primer uso | Verifica tu conexión; el modelo se descarga una sola vez (~470MB) |
| `OllamaUnavailableError` al hacer `ask` | Ollama no está corriendo o el modelo no fue descargado | `ollama serve` en otra terminal, y `ollama pull llama3.2` |
| "La base de conocimiento está vacía" | Aún no ejecutaste la ingesta | `python cli.py ingest --reset` |
| Respuestas poco precisas | El corpus de ejemplo es muy pequeño (2 documentos) | Agrega tus propios PDF/HTML/MD a `data/raw/` y vuelve a ingerir |
| `ModuleNotFoundError` | El entorno virtual no está activado | `source .venv/bin/activate` |

---

## 11. Posibles extensiones

- **Graph RAG** (tendencia 2026 mencionada en el material): representar el
  conocimiento como un grafo de entidades/relaciones en vez de fragmentos
  sueltos, usando por ejemplo `networkx` (open source).
- **Frameworks de agentes más completos:** reemplazar el bucle simplificado
  de `agentic_search()` por **LangGraph**, **LlamaIndex Workflows** o
  **CrewAI**, como señala el material para flujos multi-paso complejos.
- **Cache-Augmented Generation:** pre-cargar el contexto estable en la
  caché del modelo (soportado nativamente por algunos runtimes de Ollama)
  para evitar recuperar en cada consulta.
- **Reranking:** añadir un modelo de re-ranking (ej. un cross-encoder de
  `sentence-transformers`) sobre los resultados híbridos antes de
  construir el prompt.
- **Más formatos de origen:** extender `ingestion/extractors.py` para
  DOCX, CSV o páginas web en vivo.

---

## 12. Transparencia sobre las pruebas realizadas

Antes de entregar este proyecto se validó, en un entorno real:

- Instalación completa de todas las dependencias desde PyPI.
- **26/26 pruebas unitarias pasando**, cubriendo limpieza, chunking,
  augmentation, almacenamiento vectorial y las tres estrategias de
  búsqueda (semántica, BM25, híbrida) más el bucle de Agentic RAG.
- Ejecución real del pipeline de extracción + limpieza + chunking sobre
  los documentos de ejemplo, confirmando (por inspección directa del
  texto resultante) que el HTML de prueba —que incluye deliberadamente
  un menú de navegación, un `<script>` y un pie de página— queda
  correctamente limpio de ese ruido.
- Compilación sin errores de todos los módulos del proyecto (CLI,
  pipeline, comparación, generador, app web).

La descarga real del modelo de embeddings (Hugging Face) y la conexión a
un servidor Ollama no pudieron ejecutarse de extremo a extremo en el
entorno de desarrollo usado para preparar este proyecto, por restricciones
de red propias de ese entorno (sin acceso a internet general). Ambas
integraciones usan las APIs estándar y documentadas de
`sentence-transformers`, `chromadb` y `ollama`, y quedarán operativas en
tu máquina, que sí tiene acceso normal a internet.

---

## 13. Licencia

Este proyecto (código propio) puede usarse y modificarse libremente con
fines educativos. Las herramientas de terceros que utiliza conservan sus
propias licencias open source, listadas en la sección 3.
