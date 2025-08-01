# app.py — Streamlit Cloud version of RAG Travel Advisor

import streamlit as st
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from openai import OpenAI
client = OpenAI()
import os

# --- Load credentials from environment ---
openai.api_key = os.getenv("OPENAI_API_KEY")
cloud_id = os.getenv("ES_CLOUD_ID")
es_username = os.getenv("ES_USERNAME")
es_password = os.getenv("ES_PASSWORD")

# --- Connect to Elasticsearch Cloud ---
es = Elasticsearch(cloud_id=cloud_id, basic_auth=(es_username, es_password))
model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_NAME = "offbeat-destinations"

# --- RAG Pipeline ---
def rag_pipeline(query):
    query_vec = model.encode(query).tolist()

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vec}
            }
        }
    }
    response = es.search(index=INDEX_NAME, query=script_query, size=3)
    passages = [hit["_source"]["summary"] for hit in response["hits"]["hits"]]

    prompt = f"""
    You are a travel advisor specializing in offbeat Indian destinations.
    Given the descriptions below:

    {chr(10).join(passages)}

    Answer the user's query: "{query}"
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
     # Access the content correctly
    answer = response.choices[0].message.content
    return answer

# --- Streamlit UI ---
st.set_page_config(page_title="🧳 Travel Advisor RAG", layout="centered")
st.title("🌄 Offbeat India Travel Advisor")
st.markdown("Ask me anything about offbeat travel destinations in India.")

query = st.text_input("✍️ Your travel question")

if st.button("🔍 Ask") and query:
    with st.spinner("Thinking..."):
        answer = rag_pipeline(query)
    st.markdown("### 🧠 Suggested Answer:")
    st.write(answer)
