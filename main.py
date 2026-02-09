import os
import asyncio
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from lightrag import LightRAG, QueryParam
from lightrag.base import EmbeddingFunc

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY no encontrada. Verifique su archivo .env")

# Instanciación del cliente oficial de Google GenAI
client = genai.Client(api_key=GOOGLE_API_KEY)

# Configuración del directorio para la persistencia del grafo y vectores
WORKING_DIR = "./path_to_graph_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

async def custom_gemini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    Interface de comunicación con el modelo de lenguaje (LLM) de Gemini.
    Gestiona el historial de conversación y las instrucciones de sistema.
    """
    messages = []
    
    # Normalización del historial al formato de contenido de Gemini (Content/Part)
    for msg in history_messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            # Mapeo de roles: LightRAG utiliza 'assistant', Gemini requiere 'model'
            if role == 'assistant':
                role = 'model'
            messages.append(types.Content(role=role, parts=[types.Part(text=content)]))
        else:
            messages.append(msg)
    
    # Inclusión de la consulta actual en el flujo de mensajes
    messages.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    
    # Configuración de parámetros de generación e instrucciones de sistema
    config = types.GenerateContentConfig()
    if system_prompt:
        config.system_instruction = system_prompt
    
    # Ejecución de la solicitud de generación de contenido
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=messages,
        config=config
    )
    return response.text

async def custom_gemini_embed(texts):
    """
    Generación de vectores de características (embeddings) para recuperación semántica.
    Soporta procesamiento por lotes para optimizar la indexación.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings_list = []
    for text in texts:
        # Uso del modelo de embedding optimizado para tareas de recuperación
        response = client.models.embed_content(
            model='text-embedding-004',
            contents=text
        )
        embedding = response.embeddings[0].values
        embeddings_list.append(embedding)
    
    # Conversión a matriz NumPy para compatibilidad con la base de datos vectorial
    embeddings = np.array(embeddings_list, dtype=np.float32)
    return embeddings

# Configuración de la instancia principal de LightRAG
# Integra el motor de búsqueda por grafos con las funciones personalizadas de Gemini
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=custom_gemini_complete,
    llm_model_name='gemini-1.5-flash',
    embedding_func=EmbeddingFunc(
        embedding_dim=768,                # Dimensión nativa de text-embedding-004
        max_token_size=8192,
        func=custom_gemini_embed
    )
)

async def main():
    """
    Flujo principal de ejecución: inicialización, ingesta de datos y consulta.
    """
    # Verificación de integridad de los sistemas de almacenamiento
    await rag.initialize_storages()
    
    # Fase de ingesta de conocimiento
    texto_ejemplo = "Patrick es muy bueno con python xd :V dios mio"
    print("Iniciando proceso de inserción de documento...")
    await rag.ainsert(texto_ejemplo)
    
    # Fase de recuperación y generación (RAG)
    query = "En que es bueno patrick y que decia al final?"
    print(f"Ejecutando consulta: {query}")
    
    # Ejecución de búsqueda híbrida combinando grafos y similitud de cosenos
    result = await rag.aquery(
        query,
        param=QueryParam(mode="hybrid")
    )
    
    print(f"Respuesta del sistema: {result}")

if __name__ == "__main__":
    asyncio.run(main())