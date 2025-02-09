import json
import streamlit as st
from upstash_redis import Redis
from langchain_together import Together
from fpdf import FPDF
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the embeddings model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load case database from JSON file
CASE_DATA_FILE = "C:/Users/bhuvan/Desktop/harsha/AILegalAssistant-main/AILegalAssistant-main/courtcase.json"

def load_data_and_embeddings():
    """Load JSON data and compute embeddings."""
    json_data = pd.read_json(CASE_DATA_FILE)
    combined_data = pd.concat([json_data], ignore_index=True)
    combined_data['content'] = combined_data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    embeddings = embedding_model.encode(combined_data['content'].tolist(), show_progress_bar=True)
    return combined_data, embeddings

data, embeddings = load_data_and_embeddings()

def search_with_embeddings(query, embeddings, data, top_k=5):
    """Find the most relevant cases based on cosine similarity."""
    query_embedding = embedding_model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_k_indices = scores.argsort()[-top_k:][::-1]
    return data.iloc[top_k_indices], scores[top_k_indices]

# Initialize Redis client
REDIS_URL = 'https://natural-mule-59145.upstash.io'
REDIS_TOKEN = 'AecJAAIjcDE1Yjc3YjJm'
redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN)

# Initialize LLM (Mistral AI)
MISTRAL_API_KEY = '06c0beec1bf3261e2f668e196be1ffea9a5e33b8a51901d2b777f47e0175fa4f'
mistral_llm = Together(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=MISTRAL_API_KEY
)

def generate_questions(case_summary, headnote):
    """Generate five legal questions based on case summary and headnote."""
    prompt = f"""
    Based on the provided case summary and headnote, generate five insightful legal questions:
    
    Case Summary: {case_summary}
    Headnote: {headnote}
    """
    response = mistral_llm(prompt)
    return response.strip().split("\n")

def generate_plaintiff_notice(summary, answers):
    """Generate a plaintiff notice using LLM."""
    prompt = f"""
    Using the provided case summary and responses, draft a formal legal plaintiff notice:
    
    Case Summary: {summary}
    Responses: {answers}
    """
    response = mistral_llm(prompt)
    return response.strip()

def save_pdf(content):
    """Save plaintiff notice as a PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Ensure the content is properly encoded
    encoded_content = content.encode('latin1', 'replace').decode('latin1')
    pdf.multi_cell(0, 10, encoded_content)
    
    pdf_file = "plaintiff_notice.pdf"
    pdf.output(pdf_file)
    return pdf_file


# Streamlit UI
st.title("⚖️ AI Legal Assistant")

if "case_summary" not in st.session_state:
    st.session_state.case_summary = ""
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.notice_generated = False
    st.session_state.similar_case = None

st.session_state.case_summary = st.text_area("Provide a case summary:", st.session_state.case_summary)

if st.button("Find Related Case"):
    if st.session_state.case_summary.strip():
        similar_cases, scores = search_with_embeddings(st.session_state.case_summary, embeddings, data)
        
        if not similar_cases.empty:
            st.session_state.similar_case = similar_cases.iloc[0]
            st.session_state.questions = generate_questions(
                st.session_state.case_summary, st.session_state.similar_case['content']
            )
            st.session_state.answers = {}
            st.success("Similar case found!")
            st.write(f"**Headnote:** {st.session_state.similar_case['content']}")
        else:
            st.error("No similar case found.")
    else:
        st.warning("Please enter a case summary.")

if st.session_state.questions:
    st.write("### Answer these questions:")
    for q in st.session_state.questions:
        st.session_state.answers[q] = st.text_input(q, key=q)

    if st.button("Generate Plaintiff Notice"):
        answers_text = "\n".join([f"{q}: {a}" for q, a in st.session_state.answers.items()])
        notice_text = generate_plaintiff_notice(st.session_state.case_summary, answers_text)
        pdf_file = save_pdf(notice_text)
        st.session_state.notice_generated = True
        st.session_state.pdf_file = pdf_file
        st.success("Plaintiff notice generated successfully!")

if st.session_state.notice_generated:
    st.write("### Download the Plaintiff Notice:")
    with open(st.session_state.pdf_file, "rb") as f:
        st.download_button("Download PDF", f, "plaintiff_notice.pdf", "application/pdf")
