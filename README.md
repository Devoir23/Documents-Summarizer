# Document Summarizer RAG Appllication

This Streamlit application allows users to upload PDF documents and obtain summaries using advanced NLP techniques. It also includes a chat feature for querying documents interactively.
We have explored power of **Llama-index** + **Ollama**

## Features

- **Document Upload**: Upload PDF documents to summarize.
- **Summarization**: Automatically summarize uploaded documents.
- **Interactive Chat**: Query the document for additional information.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Devoir23/Documents-Summarizer.git
   ```
   ```bash
   cd Documents-Summarizer
   ```
2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
## Usage
1. Upload PDF documents using the sidebar.

2. Select a document and click Summarize to generate a summary.

3. Use the Chat feature to ask questions about the document.

## Code Walkthrough
Building a robust RAG application involves a lot of moving parts, here's what we have used to build the one presented here:
- **LlamaIndex** for orchestration
- **Streamlit** for creating a Chat UI
- Meta AI's **Llama 3** as the LLM through **Ollama**
- **"BAAI/bge-small-en-v1.5"** for embedding generation

### Setup LLM & Embed Model
We start by setting up the LLM & embedding model to be used by our application.
```bash
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

llama3 = Ollama(model="llama3:8b-instruct-q5_1", request_timeout=60.0)

Settings.llm = llama3
Settings.embed_model = embed_model
```
### Load Documents
Using LlamaIndex's **SimpleDirectoryReader** we provide a path to our documents & load them for further processing.
```bash
docs_for_summary = []

for file_path in glob.glob("./documents/*.pdf"):

    file_name = Path(file_path).stem
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    docs[0].doc_id = file_name
    docs_for_summary.extend(docs)
```
### Create a Summary Index
Once all the documents are loaded, we chunk them, embed them (using embedding model) & create a **doc_summary_index**.
```bash
doc_summary_index = DocumentSummaryIndex.from_documents(
    docs_for_summary,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=True,
    streaming=True,
)
```
### Generate Summary
Once the summary index is created, all you need to do is pass the document name (**doc_id** to be precise) to generate its summary.
```bash
summary = doc_summary_index.get_document_summary("RAGAs")
```

### Ask Follow Up Questions
We use the same summary index to create a **query_engine** to chat with & ask follow-up questions on the doc we just summarized.
```bash
# Setup a query engine
query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

response = query_engine.query("Why is automated evaluation important for RAG systems?")
display(Markdown(str(response)))
```
## Dependencies
- Python 3.11+
- Streamlit
- llama_index (```bash pip install llama_index```)
- Ollama
## Conclusion
In this studio, we developed a Retrieval Augmented Generation (RAG) application that allows you to summarize your documents & then allows you to chat with them for follow-up questions. Throughout this process, we learned about LlamaIndex, the go-to library for building RAG applications & Ollama for locally serving LLMs, in this case, we served Llama 3 that was recently released by MetaAI.

These techniques can similarly be applied to anchor your LLM app to various knowledge bases, such as documents, PDFs, videos, and more.

**LlamaIndex** has a variety of data loaders you can learn more about the same [here.](https://docs.llamaindex.ai/en/stable/understanding/loading/loading/)


## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Connect with me
<p align="left">
<a href="https://twitter.com/kartavyarb" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="kartavyarb" height="30" width="40" /></a>
<a href="https://instagram.com/kartavya.here" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="kartavya.here" height="30" width="40" /></a>
</p>
