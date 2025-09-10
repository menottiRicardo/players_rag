# Football Player Analytics Chatbot with RAG

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that provides intelligent insights about football players using LangChain, OpenAI, and ChromaDB. The system processes comprehensive player statistics and transfer data to answer questions about player performance, transfers, and analytics.

## 🚀 Features

- **Intelligent Document Processing**: Automatically ingests and processes CSV files containing player statistics
- **Advanced RAG Implementation**: Uses LangChain for document retrieval and OpenAI for response generation
- **Vector Database**: Leverages ChromaDB for efficient similarity search and document storage
- **Interactive Web Interface**: Built with Gradio for an intuitive chat experience
- **Configurable Architecture**: Environment-based configuration for easy deployment
- **Batch Processing**: Efficient document ingestion with configurable batch sizes
- **Error Handling**: Robust error handling with retry mechanisms

## 📊 Data Sources

The system processes various types of football data:

- **Player Statistics**: Comprehensive performance metrics including goals, assists, passing, defensive actions
- **Transfer Analysis**: Historical transfer data and player movement patterns
- **Position-Specific Metrics**: Specialized statistics for different player positions (strikers, wingers, etc.)
- **Team Performance**: Contextual team success metrics and league comparisons

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Files     │───▶│  Document        │───▶│   ChromaDB      │
│   (Player Data) │    │  Processor       │    │   Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           │
│   Gradio UI     │◀───│   RAG Pipeline   │◀──────────┘
│   (Chatbot)     │    │   (LangChain)    │
└─────────────────┘    └──────────────────┘
```

## 🛠️ Technology Stack

- **LangChain**: Framework for building LLM applications
- **OpenAI**: GPT models for natural language understanding and generation
- **ChromaDB**: Vector database for document storage and retrieval
- **Gradio**: Web interface for the chatbot
- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- ChromaDB Cloud account (optional, can use local instance)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Chatbot-with-RAG-and-LangChain.git
cd Chatbot-with-RAG-and-LangChain
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file based on the template:

```bash
cp config.env.example .env
```

Edit the `.env` file with your configuration:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Chroma Cloud Configuration (optional)
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_chroma_tenant_here
CHROMA_DATABASE=your_chroma_database_here

# Application Configuration
DATA_PATH=data
CHROMA_PATH=chroma_db
COLLECTION_NAME=player_stats

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=1.0

# Document Processing Configuration
CHUNK_SIZE=300
CHUNK_OVERLAP=100
BATCH_SIZE=300
```

## 🎯 Usage

### 1. Ingest Data

First, process your CSV files and create the vector database:

```bash
python ingest_database.py
```

This script will:
- Load all CSV files from the `data/` directory
- Split documents into chunks for optimal retrieval
- Generate embeddings using OpenAI's text-embedding-3-large model
- Store the processed data in ChromaDB

### 2. Launch the Chatbot

Start the interactive chatbot interface:

```bash
python chatbot.py
```

The Gradio interface will open in your browser, allowing you to ask questions about the player data.

## 💬 Example Queries

- "Who are the top 5 goal scorers in the dataset?"
- "Which players have the highest transfer values?"
- "Compare the performance of strikers vs wingers"
- "What are the key statistics for [Player Name]?"
- "Which teams have the most successful transfers?"

## 📁 Project Structure

```
├── chatbot.py              # Main chatbot application
├── ingest_database.py      # Document processing and ingestion
├── config.env.example      # Environment configuration template
├── requirements.txt        # Python dependencies
├── data/                   # CSV data files
│   ├── players_with_transfers_cleaned CSV.csv
│   ├── strikers_scores CSV.csv
│   ├── transfer_analysis CSV.csv
│   └── wingers_scores.csv
└── README.md              # This file
```

## ⚙️ Configuration Options

### Model Settings
- **EMBEDDING_MODEL**: OpenAI embedding model (default: text-embedding-3-large)
- **LLM_MODEL**: OpenAI chat model (default: gpt-4o-mini)
- **LLM_TEMPERATURE**: Response creativity (0.0-1.0, default: 1.0)

### Processing Settings
- **CHUNK_SIZE**: Document chunk size for processing (default: 300)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 100)
- **BATCH_SIZE**: Batch size for vector database operations (default: 300)

### Storage Settings
- **DATA_PATH**: Directory containing CSV files (default: data)
- **CHROMA_PATH**: Local ChromaDB storage path (default: chroma_db)
- **COLLECTION_NAME**: ChromaDB collection name (default: player_stats)

## 🔧 Advanced Usage

### Custom Data Processing

The `DocumentProcessor` class can be extended to handle different file formats or implement custom preprocessing:

```python
from ingest_database import DocumentProcessor, Config

config = Config()
processor = DocumentProcessor(config)
documents = processor.load_documents()
chunks = processor.process_documents(documents)
```

### Batch Processing

For large datasets, you can adjust the batch size in the configuration:

```env
BATCH_SIZE=1000  # Increase for faster processing with more memory
```

### Local vs Cloud ChromaDB

The system supports both local ChromaDB instances and ChromaDB Cloud:

- **Local**: Set `CHROMA_PATH` in your `.env` file
- **Cloud**: Configure `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE`

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required environment variables are set in your `.env` file
2. **Memory Issues**: Reduce `BATCH_SIZE` if you encounter memory errors
3. **Slow Processing**: Increase `BATCH_SIZE` for faster processing (if you have sufficient memory)
4. **ChromaDB Connection**: Verify your ChromaDB configuration and network connectivity

### Error Messages

- `Missing required environment variables`: Check your `.env` file configuration
- `Failed batch X`: Network or API issues; the system will retry automatically
- `No documents found`: Ensure CSV files are present in the `data/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [OpenAI](https://openai.com/) for the language models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Gradio](https://gradio.app/) for the web interface

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Happy analyzing! ⚽📊**
