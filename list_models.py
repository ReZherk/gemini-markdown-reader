import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

print("📋 Todos los modelos disponibles:")
print("=" * 60)

for model in client.models.list():
    print(f"\n🔹 Nombre: {model.name}")
    print(f"   Display Name: {model.display_name}")
    
    # Intentar mostrar los métodos soportados si existen
    if hasattr(model, 'supported_generation_methods'):
        print(f"   Métodos: {model.supported_generation_methods}")
    
    # Mostrar todos los atributos del modelo para debugging
    print(f"   Atributos: {dir(model)}")
    print("-" * 60)