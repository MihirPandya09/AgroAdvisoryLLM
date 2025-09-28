# KrishiSaathi - AI-Powered Agri Advisory

KrishiSaathi is an AI-based assistant designed to provide actionable agricultural advice to small and marginal farmers in Rewa district, Madhya Pradesh. It leverages scientific research, real-time weather, mandi prices, and government schemes to give localized, practical recommendations.

---

## Features

-   Integrates scientific knowledge from research reports.
-   Uses real-time structured data:
    -   Weather forecasts (rainfall, temperature, humidity, wind)
    -   Mandi prices of relevant commodities
    -   Government schemes and agronomic events
-   Provides concise, farmer-friendly advice.
-   Retrieves relevant information using a semantic vector store (Chroma) and embeddings (HuggingFace).
-   Powered by NVIDIA LLaMA 3.1 LLM via API.

---

## Live Website

You can access the live application here: [https://agroadvisoryllm.streamlit.app/]([https://agroadvisoryllm.streamlit.app/](https://agroadvisoryllm-amrgohwydmq7mj8bhqqemz.streamlit.app/))

---

## Requirements

-   Python 3.9+
-   NVIDIA API key
-   Libraries:
    ```bash
    pip install openai langchain_community chromadb sentence-transformers python-dotenv
    ```
