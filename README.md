# 🏡 Real Estate Semantic Search System

A capstone project developed as part of the B.Sc. in Industrial Engineering at **Sami Shamoon College of Engineering (SCE)**, focused on designing a smart real estate search engine powered by **vector databases** and **large language models**.

---

## 📌 Project Overview

This project builds a semantic search system that allows users to find real estate listings using **natural language queries**. By combining structured filtering, vector similarity, and **claim verification via NLI**, the system returns properties that match both **semantic meaning** and **explicit user requirements**.

The platform includes:
- Property uploading and embedding
- Query rewriting for structured extraction
- Semantic search over dense embeddings
- Logical claim verification
- Interactive frontend via Streamlit

---

## 🔍 Core Features

- **🔁 Query Rewriting**  
  Rewrites vague or complex user queries using **Gemini 2.0 Flash** to clarify constraints and extract structured filters.

- **🧠 Embedding Search**  
  Converts rewritten queries and property listings to semantic vectors using **BAAI/bge-small-en-v1.5** and retrieves top matches using **FAISS**.

- **✅ Claim Verification**  
  Each candidate result is passed through a **FLAN-T5-XL** model to determine whether it logically satisfies the user's intent — enabling true Natural Language Inference (NLI).

- **📷 Image Feature Extraction**  
  Property images are processed with **Gemini 2.0 Flash** to identify real estate-relevant objects and enrich textual descriptions.

- **📊 Evaluation Dashboard**  
  Automatically logs and visualizes precision, false negatives, semantic score distributions, and FLAN validation rates.

---

## 🗂️ Project Structure

```
.
├── modules/
│   ├── upload_section.py
│   └── search_section.py
│
├── utils/
│   ├── constraint_filter.py
│   ├── database.py
│   ├── evals_funcs.py
│   ├── features.py
│   ├── hf_config.py
│   ├── hf_loader.py
│   ├── hf_uploader.py
│   ├── inferring_filter.py
│   ├── query_rewrite.py
│   └── search_embeddings.py
│
├── streamlit_app.py
├── README.md
├── requirements.txt
```

---

## 🤖 Technologies Used

| Technology         | Purpose |
|--------------------|---------|
| **Streamlit**      | Web UI framework |
| **FAISS**          | High-performance vector index |
| **SentenceTransformers** | Semantic encoding (BAAI/bge-small-en-v1.5) |
| **Google Gemini 2.0 Flash** | Query rewriting + image item detection |
| **FLAN-T5-XL**     | Claim verification via NLI |
| **Hugging Face Hub** | Hosting models and datasets |

---

## 📊 Evaluation Methods

- Precision@k for top-N recall accuracy
- Semantic similarity histograms
- FLAN verification success rates
- Logging of semantic false negatives
- Manual inspection of edge cases

---

## 📦 Deployment

- [GitHub Repository](https://github.com/NevulaDP/Real-Estate-Search)
- [Hugging Face Dataset](https://huggingface.co/datasets/NevulaDP/real-rstate-search-db)
- [Live Streamlit App (experimental)](https://real-estate-search-fnvsaydy7qmcw5qpmkctdc.streamlit.app)

⚠️ Note: Streamlit deployment may experience delays due to model size (FLAN) and API limitations.

---

## 📚 Academic Context

This project was developed as a final B.Sc. thesis for the Department of Industrial Engineering at **Sami Shamoon College of Engineering (SCE)**.  
It draws upon recent academic research in:
- Vector databases
- Semantic search
- Embedding-based retrieval
- Natural Language Inference (NLI)

---

## 🧑‍💻 Author

**Nevo Betesh**  
B.Sc. in Industrial Engineering  
Sami Shamoon College of Engineering (SCE)  
🔗 [github.com/NevulaDP](https://github.com/NevulaDP)

---

## 📝 License

This project is released under the MIT License.  
Feel free to explore, fork, and build upon it with credit.
