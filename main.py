import os

import asyncio

from dotenv import load_dotenv

# Clase principal de LightRAG y parámetros de consulta
from lightrag import LightRAG, QueryParam

# Funciones específicas de LightRAG para usar Gemini:
# - gemini_model_complete: pedirle a Gemini que complete texto o genere respuestas
# - gemini_embed: convertir texto en embeddings (vectores numéricos) para búsquedas
from lightrag.llm.gemini import gemini_model_complete, gemini_embed


load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Por  favor,configurar el GOOGLE_API_KEY en tu archivo .env")

WORKING_DIR="./path_to_graph_storage"

if not  os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
""" 
who_are_you=type(QueryParam)

print("Esto es de tipo: ",who_are_you)
 """

rag = LightRAG(
    working_dir=WORKING_DIR,              # Carpeta donde se guardan los datos del grafo y los embeddings
    llm_model_func=gemini_model_complete, # Función que conecta con Gemini para generar texto o responder preguntas
    llm_model_name='gemini-1.5-flash',    # Nombre del modelo de Gemini usado
    embedding_func=gemini_embed,          # Función que convierte texto en embeddings
    embedding_model_name='models/text-embedding-004' # Modelo de Gemini que genera los embeddings
)


async def main():
    texto_ejemplo="Patrick es muy bueno  con python xd :V dios mio"

    rag.insert(texto_ejemplo)


    query="En que es bueno patrick y que decia al final?"

    print('Esta consultando : ', query)

    result = rag.query(
        query,                        # Texto de la pregunta que quieres hacer al grafo
        param=QueryParam(
            mode="hybrid"             # Modo de consulta:
                                      #   • "keyword": busca por coincidencia literal de palabras
                                      #   • "semantic": busca por significado usando embeddings
                                      #   • "hybrid": combina ambos enfoques para mayor precisión
        )
    )


    print("Resultado :", result )


    if __name__=="__main__":
        asyncio.run(main())
