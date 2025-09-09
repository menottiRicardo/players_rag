import os
# Fix tokenizers parallelism warnings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "player_stats2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
USE_FREE_EMBEDDINGS = os.getenv("USE_FREE_EMBEDDINGS", "true").lower() == "true"
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "chroma_db")

# Initialize embeddings model
embeddings_model = (
    HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    ) if USE_FREE_EMBEDDINGS 
    else OpenAIEmbeddings(model=EMBEDDING_MODEL)
)

# Initialize LLM and vector store
llm = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=PERSIST_DIRECTORY,
)
retriever = vector_store.as_retriever(search_kwargs={'k': 500})

def get_response(message, history):
    """Generate streaming response using RAG."""
    if not message:
        return
    
    # Retrieve relevant documents
    docs = retriever.invoke(message)
    
    # Build knowledge base and collect sources
    knowledge = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([
        doc.metadata.get('source', 'Unknown source') 
        for doc in docs 
        if hasattr(doc, 'metadata') and doc.metadata
    ]))
    
    # Create RAG prompt
    rag_prompt = f"""Eres un asistente de transferencias de futbol que responde preguntas basandote en el conocimiento que se te proporciona y solo en el.

Reglas:
- Debes dar respuestas largas y detalladas
- Si se hace una pregunta sobre un jugador siempre incluye una corta introducción sobre el jugador al final o como dato curioso
- Nunca digas que faltan datos ni como obtener los datos, porque estas hablando con un usuario
- Si no hay suficiente información, no inventes información
- No hagas preguntas al usuario, solo responde las preguntas
- No menciones que estas usando el conocimiento, porque estas hablando con un usuario final
- Si te piden una lista basado en una metrica solo entrega de mayor a menor

Pregunta: {message}
Historial: {history}
Conocimiento: {knowledge}"""


    # Stream response
    response_text = ""
    for chunk in llm.stream(rag_prompt):
        if hasattr(chunk, 'content'):
            response_text += chunk.content
            yield response_text
    
    # Add sources if available
    if sources:
        source_info = f"\n\n📚 **Fuentes consultadas:**\n" + "\n".join([f"• {source}" for source in sources])
        yield response_text + source_info
    else:
        yield response_text

# Create and launch chatbot
chatbot = gr.ChatInterface(
    get_response,
    title="Asistente de Analítica de Jugadores de Fútbol",
    description="Pregúntame sobre los jugadores y su rendimiento.",
    textbox=gr.Textbox(
        placeholder="Pregúntame sobre los jugadores...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

if __name__ == "__main__":
    chatbot.launch(share=True)