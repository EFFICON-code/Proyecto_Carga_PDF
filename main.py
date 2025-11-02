import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma # <-- Usamos ChromaDB
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --- MODELOS DE DATOS ---
class SolicitudContexto(BaseModel):
    pregunta: str
    entidad: str # La etiqueta que enviaremos desde Excel (ej. "paltas")

# --- CONFIGURACIÓN E INICIALIZACIÓN ---
DIRECTORIO_DB = "db_pdot" # Nombre de la carpeta de la base de datos

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

app = FastAPI()
db = None # Nuestra "biblioteca" local
llm = None

@app.on_event("startup")
def startup_event():
    global db, llm
    try:
        print(">>> Evento de inicio: Iniciando carga de componentes...")
        
        if not (GOOGLE_API_KEY and OPENAI_API_KEY):
            raise ValueError("!!! ERROR FATAL: Faltan claves (GOOGLE o OPENAI) en las variables de entorno.")
        print(">>> Claves de API validadas.")

        print(">>> Cargando Embeddings de Google...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        
        print(f">>> Cargando Base de Datos Chroma desde '{DIRECTORIO_DB}'...")
        if not os.path.exists(DIRECTORIO_DB):
            raise FileNotFoundError(f"!!! ERROR FATAL: No se encuentra el directorio '{DIRECTORIO_DB}'. ¿Olvidaste subir la base de datos?")
            
        # Cargar la base de datos local existente
        db = Chroma(
            persist_directory=DIRECTORIO_DB, 
            embedding_function=embeddings
        )
        print(">>> Base de Datos Chroma cargada.")

        print(">>> Cargando LLM (GPT-4o)...")
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.3)
        
        print(">>> ¡Todos los componentes cargados con éxito! Servidor listo.")
    except Exception as e:
        print(f"!!! ERROR CRÍTICO DURANTE EL INICIO: {traceback.format_exc()}")
        raise e

# --- ENDPOINT PRINCIPAL DE LA API ---
@app.post("/extraer-contexto-pdot")
async def extraer_contexto(solicitud: SolicitudContexto):
    
    pregunta = solicitud.pregunta
    entidad = solicitud.entidad.lower().strip() 
    
    print(f">>> Solicitud recibida: Entidad='{entidad}', Pregunta='{pregunta}'")
    
    if db is None or llm is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible: Los componentes de IA no están cargados.")
    
    if not entidad:
        raise HTTPException(status_code=400, detail="Error: El campo 'entidad' no puede estar vacío.")

    try:
        # --- ¡LA MAGIA DEL FILTRADO (en ChromaDB)! ---
        print(f">>> Buscando contexto en ChromaDB SÓLO con la etiqueta: 'fuente': '{entidad}'...")
        
        # Usamos el filtro de metadatos
        docs_relevantes = db.similarity_search(
            pregunta, 
            k=10, # Traer los 10 fragmentos más relevantes
            filter={"fuente": entidad} # ¡EL FILTRO!
        )

        if not docs_relevantes:
            print(">>> No se encontró contexto relevante PARA ESA ENTIDAD.")
            return {"contexto": f"No se encontró información específica para la entidad '{entidad}' sobre la consulta realizada."}

        contexto_bruto = "\n\n---\n\n".join([doc.page_content for doc in docs_relevantes])

        # 2. Crear el prompt de síntesis
        prompt_sinitesis = f"""
        Actúa como un analista técnico experto en el PDOT de la entidad: {entidad}.
        A continuación se te proporciona un texto en bruto extraído del documento de planificación de {entidad} y una pregunta del usuario. 
        Tu tarea es sintetizar la información del texto en bruto para responder a la pregunta de forma clara y concisa.

        PREGUNTA DEL USUARIO:
        {pregunta}

        TEXTO EN BRUTO EXTRAÍDO (Fuente: {entidad}):
        {contexto_bruto}

        RESPUESTA SINTETISADA (Enfocada en {entidad}):
        """
        
        print(f">>> Sintetizando la respuesta con GPT-4o (Contexto: {entidad})...")
        respuesta_final = llm.invoke(prompt_sintesis)
        print(">>> Respuesta final generada.")
        
        return {"contexto": respuesta_final.content} 

    except Exception as e:
        error_detallado = traceback.format_exc()
        print(f"!!! ERROR DURANTE EL PROCESAMIENTO DE LA SOLICITUD: {error_detallado}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}\n{error_detallado}")

@app.get("/")
def read_root():
    return {"status": "El motor de SÍNTESIS de contexto PDOT (con ChromaDB) está en línea."}