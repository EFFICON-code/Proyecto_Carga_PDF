import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil # Usaremos esto para borrar la base de datos antigua

# --- Cargar las claves de API desde .env ---
load_dotenv() 
print(">>> Variables de entorno cargadas.")

# --- CONFIGURACIÓN ---
DIRECTORIO_PERSISTENTE = "db_pdot" # Nombre de la carpeta de la base de datos
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- ¡NUESTRO MAPA DE ETIQUETAS! ---
# ¡IMPORTANTE! Revisa que los nombres de archivo sean EXACTOS.
# "etiqueta_que_usará_excel": "NombreDelArchivo.pdf"
MAPA_DE_PDOTS = {
    "paltas": "PDOT_Paltas.pdf",
    "pindal": "PDOT_Pindal.pdf",
    "catamayo": "PDyOT_Catamayo.pdf",
    "chaguarpamba": "PDyOT_Chaguarpamba.pdf"
    # Añade más PDFs aquí si lo necesitas
}

# --- CÓDIGO DEL PROGRAMA ---
print(">>> INICIANDO EL PROCESO DE INDEXACION LOCAL (ChromaDB)...")

if not GOOGLE_API_KEY:
    print("!!! ERROR: No se encontró la GOOGLE_API_KEY en el archivo .env")
    exit()

# --- FASE 0: Limpiar la base de datos antigua (si existe) ---
if os.path.exists(DIRECTORIO_PERSISTENTE):
    print(f">>> Borrando la base de datos antigua '{DIRECTORIO_PERSISTENTE}'...")
    shutil.rmtree(DIRECTORIO_PERSISTENTE)
    print(">>> Base de datos antigua borrada.")

todos_los_fragmentos = [] 

# --- FASE 1: Cargar, Dividir y ETIQUETAR cada PDF ---
print(f">>> Procesando {len(MAPA_DE_PDOTS)} PDOTs...")

for etiqueta, nombre_archivo in MAPA_DE_PDOTS.items():
    print(f"\n--- Procesando: {nombre_archivo} (Etiqueta: '{etiqueta}') ---")
    
    nombre_archivo_limpio = nombre_archivo.strip()
    if not os.path.exists(nombre_archivo_limpio):
        print(f"!!! ADVERTENCIA: No se encontró el archivo '{nombre_archivo_limpio}'. Saltando...")
        continue
        
    print(f"    Cargando archivo: {nombre_archivo_limpio}...")
    loader = PyPDFLoader(nombre_archivo_limpio)
    try:
        documentos = loader.load()
    except Exception as e:
        print(f"!!! ERROR al cargar '{nombre_archivo_limpio}': {e}. Saltando este archivo.")
        continue
            
    print(f"    Cargado. {len(documentos)} páginas encontradas.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f">>> Dividido en {len(fragmentos)} fragmentos.")
    
    # ¡LA MAGIA! Añadir la etiqueta (Metadato) a cada fragmento
    for frag in fragmentos:
        frag.metadata["fuente"] = etiqueta
        frag.metadata["archivo_origen"] = nombre_archivo_limpio
        
    todos_los_fragmentos.extend(fragmentos)
    print(f">>> Fragmentos de '{etiqueta}' etiquetados y listos.")


if not todos_los_fragmentos:
    print("!!! ERROR FATAL: No se procesó ningún documento.")
    exit()

print(f"\n>>> Total de fragmentos de todos los PDFs: {len(todos_los_fragmentos)}")

# --- FASE 2: CONFIGURACIÓN DE EMBEDDINGS ---
print(">>> Configurando el modelo de embeddings de Google...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
print(">>> Modelo de embeddings listo.")

# --- FASE 3: CREACIÓN DE LA NUEVA BASE DE DATOS (ChromaDB Local) ---
print(f">>> Creando y guardando la nueva base de datos en '{DIRECTORIO_PERSISTENTE}'...")
print(">>> Este proceso puede tardar varios minutos...")

# Chroma crea la nueva carpeta 'db_pdot' con TODOS los fragmentos etiquetados
db = Chroma.from_documents(
    documents=todos_los_fragmentos, 
    embedding=embeddings,
    persist_directory=DIRECTORIO_PERSISTENTE
)

print("\n*** ¡PROCESO DE INDEXACION LOCAL COMPLETADO CON EXITO! ***")
print(f"*** Tu nueva biblioteca '{DIRECTORIO_PERSISTENTE}' está lista. ***")