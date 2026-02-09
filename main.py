import os

import asyncio

from dotenv import load_dotenv

# Clase principal de LightRAG y parámetros de consulta 
from lightrag import LightRAG, QueryParam

# Funciones específicas de LightRAG para usar Gemini: 
# - google_generativeai_model_complete: completar texto con Gemini 
# - google_generativeai_embed: generar embeddings con Gemini 
from lightrag.llm import google_generativeai_model_complete, google_generativeai_embed