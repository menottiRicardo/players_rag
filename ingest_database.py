import time
import os
# Fix tokenizers parallelism warnings

from typing import List, Tuple, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Config:
    """Configuration class for the document ingestion process."""

    def __init__(self):
        self.data_path = os.getenv("DATA_PATH", "data")
        self.collection_name = os.getenv("COLLECTION_NAME", "player_stats2")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "500"))
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.use_free_embeddings = (
            os.getenv("USE_FREE_EMBEDDINGS", "true").lower() == "true"
        )

        # Local Chroma configuration
        self.persist_directory = os.getenv("PERSIST_DIRECTORY", "chroma_db")

        # Validate required environment variables
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration is present."""
        # For local Chroma, we only need to ensure the persist directory exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
            print(f"✅ Created persist directory: {self.persist_directory}")


class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self) -> List[Document]:
        """Load CSV documents from the data directory."""
        print("Loading CSV documents...")

        loader = DirectoryLoader(
            self.config.data_path,
            glob="**/*.csv",
            loader_cls=CSVLoader,
            loader_kwargs={
                "csv_args": {
                    "delimiter": ";",
                    "quotechar": '"'
                },
                "encoding": "utf-8"
            }
        )

        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents from CSV files")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks and filter metadata."""
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)

        print("Filtering complex metadata...")
        filtered_chunks = filter_complex_metadata(chunks)
        print(f"Metadata filtered. Total chunks: {len(filtered_chunks)}")

        return filtered_chunks


class VectorStoreManager:
    """Manages Chroma vector store operations."""

    def __init__(self, config: Config):
        self.config = config
        self.embeddings_model = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()

    def _initialize_embeddings(self):
        """Initialize embeddings model based on configuration."""
        if self.config.use_free_embeddings:
            print(
                f"🆓 Using free embedding model: {self.config.embedding_model}"
            )
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'},  # Use CPU to avoid GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            print(
                f"💰 Using OpenAI embedding model: {self.config.embedding_model}"
            )
            return OpenAIEmbeddings(model=self.config.embedding_model)

    def _initialize_vector_store(self) -> Chroma:
        """Initialize and connect to local Chroma database."""
        vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=self.config.persist_directory,
        )

        print(
            f"✅ Connected to local Chroma database at "
            f"{self.config.persist_directory}"
        )
        return vector_store

    def add_documents_batch(
        self, documents: List[Document], document_ids: List[str]
    ) -> bool:
        """Add a batch of documents to the vector store with retry logic."""
        try:
            self.vector_store.add_documents(
                documents=documents, ids=document_ids
            )
            return True
        except Exception as e:
            print(f"⚠️ First attempt failed: {e}")
            try:
                self.vector_store.add_documents(
                    documents=documents, ids=document_ids
                )
                return True
            except Exception as e2:
                print(f"❌ Retry failed: {e2}")
                return False


class DocumentIngestionPipeline:
    """Main pipeline for document ingestion."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.document_processor = DocumentProcessor(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)

    def _create_batches(
        self, chunks: List[Document]
    ) -> List[Tuple[List[Document], List[str], int]]:
        """Create batches of documents with unique IDs."""
        batches = []
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batch_uuids = uuids[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            batches.append((batch, batch_uuids, batch_num))

        return batches

    def _process_batches(
        self, batches: List[Tuple[List[Document], List[str], int]]
    ) -> None:
        """Process all batches sequentially."""
        total_chunks = sum(len(batch[0]) for batch in batches)
        print(f"Total chunks to process: {total_chunks}")
        print(f"Processing {len(batches)} batches sequentially")
        print(f"Batch size: {self.config.batch_size} chunks per batch")

        start_time = time.time()
        successful_batches = 0

        for batch_num, (batch, batch_uuids, _) in enumerate(batches, 1):
            success = self.vector_store_manager.add_documents_batch(
                batch, batch_uuids
            )

            if success:
                print(f"✅ Successfully added batch {batch_num}")
                successful_batches += 1
            else:
                print(f"❌ Failed batch {batch_num}")

            print(f"Progress: {batch_num}/{len(batches)} batches completed")

        end_time = time.time()
        total_time = end_time - start_time

        print("🎉 Processing completed!")
        print(f"✅ Successful batches: {successful_batches}/{len(batches)}")
        print(
            f"⏱️ Total time: {total_time:.2f} seconds "
            f"({total_time/60:.2f} minutes)"
        )

        if successful_batches > 0:
            total_chunks = sum(len(batch[0]) for batch in batches)
            rate = total_chunks/total_time
            print(
                f"🚀 Processed {total_chunks} chunks at "
                f"{rate:.1f} chunks/second"
            )

    def run(self) -> None:
        """Run the complete document ingestion pipeline."""
        try:
            # Load and process documents
            documents = self.document_processor.load_documents()
            chunks = self.document_processor.process_documents(documents)

            # Create batches and process them
            batches = self._create_batches(chunks)
            self._process_batches(batches)

        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            raise


def main():
    """Main function to run the document ingestion pipeline."""
    pipeline = DocumentIngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
