# Multi-Document RAG Chatbot

A Streamlit-based chatbot that uses Retrieval Augmented Generation (RAG) to answer questions based on multiple PDF documents. The application leverages Hugging Face's language models and FAISS for efficient document retrieval and provides context-aware responses using a conversational memory system.

## Features

- 📚 Process multiple PDF documents simultaneously
- 💬 Interactive chat interface using streamlit-chat
- 🔍 Real-time document source tracking with page references
- 🧠 Conversational memory for context-aware responses
- 🔄 Easy conversation reset functionality
- 📊 File processing status display in sidebar

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
```bash
# Create environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment:
- Create `.env` file in root directory
- Add your Hugging Face token:
```
HF_TOKEN=your_huggingface_token_here
```

5. Create data directory and add PDFs:
```bash
mkdir data
# Copy your PDF files into the data folder
```

6. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
.
├── data/               # Store your PDF documents here
├── app.py             # Main application code
├── .env               # Environment variables
├── requirements.txt   # Project dependencies
└── README.md         # Documentation
```

## Requirements

Core dependencies include:
```txt
streamlit>=1.31.0
streamlit-chat>=0.1.1
python-dotenv>=1.0.0
langchain>=0.1.0
langchain-community>=0.0.13
langchain-huggingface>=0.0.6
faiss-cpu>=1.7.4
torch>=2.1.0
transformers>=4.36.0
pypdf>=3.17.0
sentence-transformers>=2.2.2
```

For complete list, see `requirements.txt`

## How It Works

1. **Document Processing**
   - Application scans the `data` folder for PDF files
   - Each PDF is loaded and split into manageable chunks
   - Text chunks are converted to embeddings using HuggingFace
   - Embeddings are stored in a FAISS vector database

2. **Question Answering**
   - User questions are processed using the same embedding model
   - FAISS retrieves the most relevant document chunks
   - LLM generates responses based on retrieved context
   - Sources are tracked and displayed with page references

3. **Memory System**
   - Maintains conversation history for context
   - Allows for follow-up questions
   - Can be reset via sidebar button

## Model Configuration

Default settings:
- Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Temperature: 0.5
- Max Tokens: 4096
- Top K: 3

## Troubleshooting

### Common Issues

1. **Installation Problems**
   ```bash
   # If facing dependency conflicts, try:
   pip install -r requirements.txt --upgrade
   ```

2. **Authentication Errors**
   - Verify HF_TOKEN in .env
   - Ensure token has proper permissions

3. **PDF Processing Issues**
   - Check PDF file permissions
   - Verify PDF is not corrupted
   - Ensure PDF is text-based, not scanned images

4. **Memory Usage**
   - For large PDFs, monitor system memory
   - Consider reducing chunk size if needed

### Error Messages

1. "No PDF files found":
   - Check if PDFs are in the data folder
   - Verify file extensions are .pdf

2. "Failed to initialize chat assistant":
   - Check Hugging Face token
   - Verify internet connection
   - Ensure model access permissions

## Usage Tips

1. **Best Practices**
   - Place all relevant PDFs in data folder before starting
   - Use specific questions for better responses
   - Check source references for verification

2. **Performance Optimization**
   - Remove unnecessary PDFs from data folder
   - Reset conversation if context becomes too long
   - Monitor system resources for large documents

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://python.langchain.com/)
- Uses [Hugging Face](https://huggingface.co/) models
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)

## Contact

- LinkedIn: [https://www.linkedin.com/in/shubham-shnd?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app](https://www.linkedin.com/in/shubham-shnd?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
- For issues and feature requests, please use the GitHub issues page.


---

*Note: This project is for educational and demonstration purposes. Ensure you have the necessary rights and permissions for any models and documents used.*
