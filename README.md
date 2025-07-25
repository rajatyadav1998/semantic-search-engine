# 🧠 Semantic Search Engine with FastAPI & Ngrok (Colab Ready)

This project is a **Semantic Search Engine** built using Python that allows users to find the most contextually relevant documents to their queries. Instead of relying on keyword matching, it uses **vector similarity** between embedded representations of text.

---

## 🔍 Problem Statement

Traditional search engines match keywords and fail to understand context. Semantic search solves this by understanding **meaning**, enabling smarter and more intuitive search.

---

## 🧰 Tech Stack

- `FastAPI` – For building a REST API  
- `sentence-transformers` – To embed text semantically  
- `scikit-learn` – For cosine similarity  
- `Ngrok` – To expose API to the web  
- `Google Colab` – As runtime environment  

---

## 💡 How It Works (Step-by-Step)

1. **Input Documents**: A predefined list of documents is stored.
2. **Embeddings**: We use `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to convert text to vector representations.
3. **Query Input**: User sends a query via the API.
4. **Semantic Matching**: Compute cosine similarity between query vector and document vectors.
5. **Return Result**: Return the top-N most similar documents via API response.

---

## 📦 Install Dependencies (Run this in Google Colab)

```python
!pip install fastapi nest_asyncio pyngrok uvicorn sentence-transformers scikit-learn
```

---

## 🧠 Full Working Code

```python
# 1. Install dependencies (if not already installed)
!pip install fastapi nest_asyncio pyngrok uvicorn sentence-transformers scikit-learn

# 2. Import modules
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# 3. Setup FastAPI
app = FastAPI()

# 4. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 5. Sample corpus
documents = [
    "Machine learning is a method of data analysis.",
    "Deep learning is a subset of machine learning.",
    "FastAPI is a modern web framework for APIs.",
    "Football is a popular sport.",
    "Natural Language Processing deals with text data."
]

# 6. Precompute document embeddings
doc_embeddings = model.encode(documents)

# 7. Search endpoint
@app.post("/search")
def search(query: dict):
    query_text = query["query"]
    query_embedding = model.encode([query_text])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_idx = similarities.argmax()
    return {
        "query": query_text,
        "most_similar_document": documents[top_idx],
        "similarity_score": float(similarities[top_idx])
    }

# 8. Handle event loop for Colab
nest_asyncio.apply()

# 9. Launch public URL using ngrok
public_url = ngrok.connect(8000)
print("Your API is live at:", public_url)

# 10. Start server
uvicorn.run(app, port=8000)
```

---

## 🧪 How to Test

1. Run all cells in Colab.
2. Visit `https://<your-ngrok-url>/docs`
3. Use `/search` endpoint and send:

```json
{
  "query": "Tell me about neural networks"
}
```

Example response:

```json
{
  "query": "Tell me about neural networks",
  "most_similar_document": "Deep learning is a subset of machine learning.",
  "similarity_score": 0.78
}
```

---

## 💬 How to Explain to Interviewer

- **What**: A semantic search engine that finds documents based on meaning, not keywords.
- **Why**: Keyword search fails for synonyms or similar context. Semantic solves this with embeddings.
- **How**: 
  - Used `sentence-transformers` to convert queries and documents into dense vectors.
  - Used `cosine similarity` to find the closest match.
  - Deployed using `FastAPI` + `Ngrok` inside Colab for live demo without cloud setup.
- **Where it's used**: Search engines, recommendation systems, AI assistants, resume filtering, etc.

---

## 📁 Directory Structure (if exporting)

```
semantic-search-engine/
├── app.py                  # Main FastAPI app
├── requirements.txt        # All dependencies
├── README.md               # Project explanation
```

---

## 📜 requirements.txt

```
fastapi
uvicorn
nest_asyncio
pyngrok
sentence-transformers
scikit-learn
```

---

## 👨‍💻 Author

**Rajat Yadav**  
🎓 DTU | 🏢 Ex-Wipro | 📞 9538746317  
🔗 [LinkedIn](https://www.linkedin.com/in/rajat-yadav-575b46177)

---

## 📝 License

Free for learning and demonstration purposes.
