import os
# Fix tokenizers parallelism warnings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "all_files") # or scores
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))

# Chroma Cloud configuration
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize LLM and vector store
llm = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL)

# Validate Chroma Cloud configuration
if not CHROMA_API_KEY:
    raise ValueError(
        "CHROMA_API_KEY environment variable is required for Chroma Cloud"
    )
if not CHROMA_TENANT:
    raise ValueError(
        "CHROMA_TENANT environment variable is required for Chroma Cloud"
    )
if not CHROMA_DATABASE:
    raise ValueError(
        "CHROMA_DATABASE environment variable is required for Chroma Cloud"
    )

# Create Chroma Cloud client
chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    client=chroma_client,
)
retriever = vector_store.as_retriever(search_kwargs={'k': 200})


def get_response(message, history):
    """Generate streaming response using RAG."""
    if not message:
        return

    # Retrieve relevant documents
    docs = retriever.invoke(message)

    # Debug: Print information about retrieved documents
    print(f"Number of documents retrieved: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("---")

    # Build knowledge base and collect sources
    knowledge = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([
        doc.metadata.get('source', 'Unknown source')
        for doc in docs
        if hasattr(doc, 'metadata') and doc.metadata
    ]))

    print(f"Knowledge length: {len(knowledge)}")
    print(f"Sources: {sources}")

    # Create RAG prompt
    system_rules = [
        "Debes dar respuestas detalladas",
        ("Si se hace una pregunta sobre un jugador "
         "siempre incluye un dato curioso"),
        ("Nunca digas que faltan datos ni como obtener los datos, "
         "porque estas hablando con un usuario"),
        "Si no hay suficiente información, NUNCA inventes información",
        "No hagas preguntas al usuario, solo responde las preguntas",
        ("No menciones que estas usando el conocimiento, "
         "porque estas hablando con un usuario final"),
        ("Si te piden una lista basado en una metrica "
         "solo entrega de mayor a menor"),
        "Nunca inventes información, solo responde lo que sabes"
    ]

    rules_text = "\n    - " + "\n    - ".join(system_rules)

    prompt_intro = (
        "Eres un asistente de transferencias de futbol que responde "
        "preguntas basandote en el conocimiento que se te "
        "proporciona y no puedes usar tu informacion interna"
    )
    rag_prompt = f"""{prompt_intro}
    Reglas:{rules_text}

Pregunta: {message}
Historial: {history}
el conocimiento es: {knowledge}"""

    # Get full response
    response = llm.invoke(rag_prompt)
    response_text = (
        response.content if hasattr(response, 'content') else str(response)
    )

    # Add sources if available
    if sources:
        source_info = (
            "\n\n📚 **Fuentes consultadas:**\n" +
            "\n".join([f"• {source}" for source in sources])
        )
        response_text += source_info

    return response_text


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
