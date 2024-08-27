import os
import validators
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.set_page_config(page_title="Student Exam Preparation Platform", page_icon="📚")


st.sidebar.title("Choose")
selection = st.sidebar.radio("Select a section:", ["PDF Query", "Summarize URL", "Question Paper Generation"])

# PDF Query Section
if selection == "PDF Query":
    st.markdown('<h1 class="title">I\'m hungry🍽️😋feed me PDFs📚 I\'ll help you </h1>', unsafe_allow_html=True)

    if groq_api_key:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
        
        session_id = st.text_input("Session ID", value="default_session")

        if 'store' not in st.session_state:
            st.session_state.store = {}

        uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                temppdf = "./temp.pdf"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.write("Genie:", response['answer'])
    else:
        st.warning("Please set the Groq API Key in the .env file")

# Summarize URL Section
elif selection == "Summarize URL":
    st.markdown('<h1 class="title">Your Study Partner 😊 Skip the long content—get quick summaries!</h1>', unsafe_allow_html=True)

    generic_url = st.text_input("URL", placeholder="Enter the URL here")

    prompt_template = """
    Provide a summary of the following content in 600 words:
    Content: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize"):
        if not groq_api_key.strip() or not generic_url.strip():
            st.error("Please provide the Groq API Key and URL to get started")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL. It can be a YT video URL or website URL")
        else:
            try:
                with st.spinner("Waiting..."):
                    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                        )
                    docs = loader.load()

                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success(output_summary)
            except Exception as e:
                st.error(f"Exception: {e}")

# Question Paper Generation Section
elif selection == "Question Paper Generation":
    st.markdown('<h1 class="title">I\'ll guess your question paper 😉</h1>', unsafe_allow_html=True)

    if groq_api_key:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

        session_id = st.text_input("Session ID", value="default_session")

        if 'store' not in st.session_state:
            st.session_state.store = {}

        uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                temppdf = "./temp.pdf"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.write("Genie:", response['answer'])
    else:
        st.warning("Please set the Groq API Key in the .env file")
