import os
import json
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from langchain.schema import Document

# ------------------- API KEYS -------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API")
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API")
if not OPENAI_API_KEY or not NVIDIA_API_KEY:
    st.error("OpenAI or NVIDIA API keys not set in Streamlit Secrets!")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------- VECTOR STORE -------------------
PERSIST_DIR = "./agroadvisory_chroma"
COLLECTION = "agroadvisory"

embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectordb = Chroma(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
    embedding_function=embedder,
)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})

# ------------------- PROMPT BUILDER -------------------
def build_enhanced_prompt(user_query, retrieved_chunks, structured_data, table_data=None):
    retrieved_context = "\n\n".join(retrieved_chunks)
    structured_context = json.dumps(structured_data, indent=2)
    table_context = ""
    if table_data:
        table_context = "\n\nAdditional Table Data (from research documents):\n"
        table_context += json.dumps(table_data, indent=2)
    return f"""
You are KrishiSaathi, a knowledgeable and empathetic agri-advisor assisting small and marginal farmers.

Farmer's Question:
{user_query}

Relevant Knowledge:
{retrieved_context}

Real-Time Structured Data:
{structured_context}

{table_context}

Provide concise, localized advice for the farmer in simple terms.
"""

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="KrishiSaathi AI Advisor", page_icon="🌾")
st.title("🌾 KrishiSaathi - AI Agri Advisory")
st.write("Ask KrishiSaathi your farming questions and get practical advice!")

query = st.text_area("Enter your question:", height=100)
submit = st.button("Get Advice")

if submit and query:
    with st.spinner("Fetching knowledge and generating advice..."):
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        retrieved_chunks = [d.page_content if isinstance(d, Document) else d for d in docs]

        structured_data = {
            "weather_forecast": {"rainfall_mm": 8, "temperature_c": 24, "humidity_percent": 70},
            "mandi_prices": {"wheat": "₹2100/quintal"},
            "govt_schemes": ["PM-Kisan support active", "Crop insurance registration open"]
        }

        table_data = {
            "fertilizer_recommendations": [
                {"crop": "wheat", "urea_kg_per_ha": 120, "expected_yield_quintals": 42}
            ]
        }

        # Build prompt
        prompt = build_enhanced_prompt(query, retrieved_chunks, structured_data, table_data)

        # Call NVIDIA LLM API
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
            stream=True
        )

        # Display advice in real-time
        st.subheader("🧑‍🌾 KrishiSaathi’s Advice:")
        advice_container = st.empty()
        advice_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                advice_text += chunk.choices[0].delta.content
                advice_container.text(advice_text)
