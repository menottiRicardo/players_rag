import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "player_stats")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))

# Chroma Cloud configuration
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
print(CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE)

embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# initiate the model
llm = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL)

# connect to Chroma Cloud
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    chroma_cloud_api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

# Set up the vectorstore to be the retriever
num_results = 300
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot


def get_response(message, history):
    # print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge' with source information
    knowledge = ""
    sources = []

    for doc in docs:
        knowledge += doc.page_content+"\n\n"
        # Extract source information if available
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', 'Unknown source')
            if source not in sources:
                sources.append(source)

    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        Eres un asistente que responde preguntas basándose en el conocimiento
        que se te proporciona.
        Al responder, no utilizas tu conocimiento interno,
        sino únicamente la información en la sección "El conocimiento".
        Reglas:
        - No mencionas nada al usuario sobre el conocimiento proporcionado.
        - Debes dar respuestas largas y detalladas.
        - Si se hace una pregunta sobre un jugador siempre incluye una corta introducción sobre el jugador.
        - Nunca me digas que faltan datos, siempre que se pueda, da la información que tengas.
        - No me digas como obtener los datos, porque estas hablando con un usuario.
        - Si no hay suficiente información, diga que no lo sabe, no inventes información.
        - utliza todas las metricas que tengas disponibles y no hagas comparaciones entre jugadores basados en una sola metrica.


        La pregunta: {message}

        Historial de conversación: {history}

        El conocimiento: {knowledge}

        """

        print(rag_prompt)

        # get the response from the LLM
        response = llm.invoke(rag_prompt)
        
        # Add source information to the response
        if sources:
            source_info = f"\n\n📚 **Fuentes consultadas:**\n"
            for source in sources:
                source_info += f"• {source}\n"
            return response.content + source_info
        else:
            return response.content


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

# launch the Gradio app
chatbot.launch()
