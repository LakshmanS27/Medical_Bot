
# 🩺 Medical Q&A Chatbot

This project is a **Medical Question-Answering Chatbot** powered by the [MedQuAD](https://catalog.data.gov/dataset/medquad-a-medical-question-answer-dataset) dataset and Google Gemini (via LangChain).  
It allows you to ask medical questions, retrieves relevant answers from the MedQuAD knowledge base, and provides context and sources.

---

## 📂 Features

✅ Parse MedQuAD XML files into a clean CSV dataset  
✅ Preprocess the text using NLP (lemmatization, stopword removal)  
✅ Build a FAISS vector database for efficient retrieval  
✅ Use HuggingFace embeddings (`all-MiniLM-L6-v2`) for semantic search  
✅ Answer questions with context using Google Gemini (via LangChain)  
✅ Streamlit interface for interactive querying  

---

## 🚀 Setup

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Make sure you have:
- Python ≥ 3.8
- Internet connection to download `en_core_web_sm` and embeddings model

### 3️⃣ Prepare your API Key

Get your **Google API Key** from [Google AI Console](https://makersuite.google.com/app/apikey).  
Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## 🛠️ Workflow

### 📄 Parse MedQuAD Data
Convert MedQuAD XML files into a CSV:
```bash
python data_parser.py
```

### 🧹 Preprocess Dataset
Clean and lemmatize the text:
```bash
python pre.py
```

### 🗃️ Build Vector Database
Run the Streamlit app and click **"Create Knowledge Base"** to build the FAISS index.

---

## 🖥️ Run the Chatbot

Start the Streamlit app:
```bash
streamlit run main.py
```

You’ll see a web app where you can:
- Rebuild the knowledge base (if needed)
- Ask medical questions
- View answers with context and source documents

---

## 📦 Project Structure

```
├── data_parser.py          # Parses MedQuAD XML → CSV
├── pre.py                  # Preprocesses text (lemmatization, cleaning)
├── chain_module.py         # Builds vector DB & defines QA chain
├── main.py                 # Streamlit interface
├── dataset.csv             # Raw MedQuAD dataset (generated)
├── dataset_cleaned.csv     # Cleaned dataset (generated)
├── faiss_index/            # Vector database (generated)
├── .env                    # API key (you create this)
└── requirements.txt        # Python dependencies
```

---

## 📚 References

- [MedQuAD Dataset](https://catalog.data.gov/dataset/medquad-a-medical-question-answer-dataset)
- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://ai.google/discover/gemini/)
- [FAISS](https://faiss.ai/)
- [HuggingFace Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 🤝 License

This project is for educational and research purposes. Use responsibly — it does not replace professional medical advice.

---

## 👨‍💻 Author

Developed by [Your Name]. Contributions welcome!
