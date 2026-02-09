# Proyecto RAG con Gemini y LightRAG

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** utilizando:

- **Google Gemini API** para generación de texto y embeddings.
- **LightRAG** como motor de búsqueda híbrido (grafos + similitud de cosenos).
- **Python 3.12** con librerías modernas para ejecución asíncrona y manejo de configuraciones.

---

## Requisitos

### Python

- Python **3.12**

### Librerías

- `numpy`
- `python-dotenv`
- `google-genai`
- `lightrag`

---

## Instalación rápida

```bash
pip install numpy python-dotenv google-genai lightrag
```

---

## Configuración

### 1. Variables de entorno

Crear un archivo `.env` en la raíz del proyecto con tu clave de la API de Google:

```env
GOOGLE_API_KEY=tu_api_key_aqui
```

### 2. Directorio de trabajo

Definir un directorio para almacenar el grafo y los vectores:

```python
WORKING_DIR = "./path_to_graph_storage"
```

> Este directorio se crea automáticamente si no existe.

---

## Funcionalidades principales

### 1. Conexión con Gemini

- **`custom_gemini_complete`**
  Función para interactuar con el modelo **Gemini 1.5 Flash** y generar texto.

- **`custom_gemini_embed`**
  Función para generar embeddings usando el modelo **`text-embedding-004`**.

---

### 2. Integración con LightRAG

Se configura una instancia de **LightRAG** que:

- Usa **Gemini** como motor de generación (`llm_model_func`).
- Usa embeddings personalizados para indexar y recuperar documentos.
- Almacena datos en el directorio definido (`WORKING_DIR`).

---

### 3. Flujo principal (`main`)

El flujo principal realiza los siguientes pasos:

1. Inicializa los sistemas de almacenamiento.
2. Inserta un documento de ejemplo en la base de conocimiento.
3. Ejecuta una consulta híbrida (grafos + similitud).
4. Devuelve la respuesta generada por el sistema.

---

## Ejecución

Para correr el proyecto:

```bash
python main.py
```

---

## Ejemplo de salida esperada

```text
Iniciando proceso de inserción de documento...
Ejecutando consulta: En que es bueno patrick y que decia al final?
Respuesta del sistema: Patrick es muy bueno con python xd :V dios mio
```

---

## Estructura del código

- **Configuración de API y entorno**
  - Carga de variables desde `.env`.

- **Funciones personalizadas**
  - `custom_gemini_complete`: generación de texto.
  - `custom_gemini_embed`: generación de embeddings.

- **Instancia LightRAG**
  - Integra Gemini con el motor de búsqueda híbrido.

- **Función `main`**
  - Flujo completo de ingesta y consulta.

---

## Notas

- Asegúrate de tener una **clave válida de Google Gemini API**.
- El parámetro `embedding_dim = 768` corresponde a la dimensión nativa del modelo **`text-embedding-004`**.
- El modo de consulta **`hybrid`** combina búsqueda por grafo y similitud de cosenos para mejores resultados.
