import os
import json
import numpy as np
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# --- Load Secrets from Streamlit ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- Role Definitions ---
USER_ROLES = {
    "doctor": "a highly experienced cardiologist. Provide a detailed and clinically accurate answer using medical terminology.",
    "nurse": "a compassionate and knowledgeable nurse. Use supportive and simple language suitable for explaining to patients or families.",
    "patient": "an AI assistant for patients. Explain in very simple, non-technical terms so an average person can easily understand."
}

CUSTOM_ROLES_FILE = "custom_roles.json"

def load_custom_roles():
    if os.path.exists(CUSTOM_ROLES_FILE):
        with open(CUSTOM_ROLES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_custom_roles(custom_roles):
    with open(CUSTOM_ROLES_FILE, "w") as f:
        json.dump(custom_roles, f)

custom_roles = load_custom_roles()
all_roles = list(USER_ROLES.keys()) + list(custom_roles.keys()) + ["other"]

def get_rag_chain(role_description: str, role_name: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vectordb = FAISS.load_local(
        "faiss_db", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt_template = f"""
You are {role_description}

Answer the user's question using only the information provided in the context below. Be accurate, clear, and informative.

If the context does not contain the answer, say “I don’t know based on the available information.”
Do NOT make up information or answer from outside the context.

---------------------
Context:
{{context}}

User Question:
{{question}}

Answer (as a {role_name}):
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def get_history_path(role_key):
    return f"chat_history_{role_key}.json"

def load_history(role_key):
    path = get_history_path(role_key)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_history(role_key, history):
    with open(get_history_path(role_key), "w") as f:
        json.dump(history, f)

def get_question_embedding(question, embeddings_model):
    return embeddings_model.embed_query(question)

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Health RAG Chatbot", page_icon="🩺", layout="wide")
st.title("Health RAG Chatbot")

if "selected_role" not in st.session_state:
    st.session_state["selected_role"] = all_roles[0]
if "custom_role_desc" not in st.session_state:
    st.session_state["custom_role_desc"] = ""

role = st.selectbox("Select your role:", all_roles, index=all_roles.index(st.session_state["selected_role"]))

if role != st.session_state["selected_role"]:
    st.session_state["selected_role"] = role
    st.session_state["clear_input"] = True
    if role != "other":
        st.session_state["custom_role_desc"] = ""

if role == "other":
    custom_desc = st.text_input("Enter your custom role description:", value=st.session_state["custom_role_desc"])
    st.session_state["custom_role_desc"] = custom_desc.strip()
    if st.session_state["custom_role_desc"]:
        new_key = custom_desc.lower().replace(" ", "_")
        if new_key not in USER_ROLES and new_key not in custom_roles:
            custom_roles[new_key] = custom_desc
            save_custom_roles(custom_roles)
            st.experimental_rerun()

if role == "other" and st.session_state["custom_role_desc"]:
    role_key = st.session_state["custom_role_desc"].lower().replace(" ", "_")
    role_description = st.session_state["custom_role_desc"]
    role_name = "custom_role"
elif role in USER_ROLES:
    role_key = role
    role_description = USER_ROLES[role]
    role_name = role
else:
    role_key = role
    role_description = custom_roles.get(role, "an assistant")
    role_name = role

chat_key = f"chat_history_{role_key}"
selected_index_key = f"selected_index_{role_key}"
latest_key = f"latest_message_{role_key}"

if chat_key not in st.session_state:
    st.session_state[chat_key] = load_history(role_key)
if selected_index_key not in st.session_state:
    st.session_state[selected_index_key] = None
if latest_key not in st.session_state:
    st.session_state[latest_key] = None
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False
if "embeddings_model" not in st.session_state:
    st.session_state["embeddings_model"] = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
embeddings_model = st.session_state["embeddings_model"]

# --- Layout ---
left_col, right_col = st.columns([1, 3])

with left_col:
    st.subheader("💬 Conversations")
    history = st.session_state[chat_key]
    if history:
        for i, chat in enumerate(history):
            title = chat["question"][:40] + ("..." if len(chat["question"]) > 40 else "")
            if st.button(title, key=f"summary_{role_key}_{i}"):
                st.session_state[selected_index_key] = i
                st.session_state[latest_key] = None
    else:
        st.info("No previous chats yet for this role.")

with right_col:
    st.subheader("💡 Ask a Question")

    if st.session_state["clear_input"]:
        st.session_state["input_question"] = ""
        st.session_state["clear_input"] = False

    question = st.text_input("Your question:", key="input_question")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    chain = get_rag_chain(role_description, role_name)
                    response = chain.run(question)
                    embedding = get_question_embedding(question, embeddings_model)

                    new_entry = {
                        "question": question,
                        "answer": response,
                        "embedding": embedding
                    }

                    st.session_state[chat_key].append(new_entry)
                    save_history(role_key, st.session_state[chat_key])

                    st.session_state[latest_key] = new_entry
                    st.session_state[selected_index_key] = None
                    st.session_state["clear_input"] = True

                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Chat and Related Questions ---
    chat_col, related_col = st.columns([3, 1])

    with chat_col:
        display_chat = None
        if st.session_state[latest_key]:
            display_chat = st.session_state[latest_key]
        elif st.session_state[selected_index_key] is not None:
            idx = st.session_state[selected_index_key]
            if 0 <= idx < len(st.session_state[chat_key]):
                display_chat = st.session_state[chat_key][idx]

        if display_chat:
            st.markdown("### 🧑 You asked:")
            st.markdown(display_chat["question"])
            st.markdown("### 🤖 AI answered:")
            st.markdown(display_chat["answer"])

    with related_col:
        st.markdown("### 🔁 Related Questions")

        if display_chat:
            current_emb = display_chat.get("embedding", None)
            history = st.session_state[chat_key]

            if current_emb and len(history) > 1:
                similarities = []
                for i, chat in enumerate(history):
                    if chat is not display_chat and "embedding" in chat:
                        sim = cosine_similarity(current_emb, chat["embedding"])
                        if sim >= 0.75:
                            similarities.append((sim, i, chat))

                similarities.sort(key=lambda x: x[0], reverse=True)
                top_related = similarities[:5]

                if top_related:
                    for sim, i, chat in top_related:
                        title = chat["question"][:40] + ("..." if len(chat["question"]) > 40 else "")
                        if st.button(f"{title}", key=f"related_{role_key}_{i}"):
                            st.session_state[selected_index_key] = i
                            st.session_state[latest_key] = None
                else:
                    st.write("No highly relevant related questions.")
            else:
                st.write("Ask a question to view related ones.")
        else:
            st.write("Ask something to see suggestions.")
