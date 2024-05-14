import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationTokenBufferMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")

# print(loader)

st.set_page_config(
    page_title="Llama3",
    page_icon="📃",
)

callback = False


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        if callback:
            self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        if callback:
            save_messages(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        if callback:
            self.message += token
            self.message_box.markdown(self.message)


if "groq1_messages" not in st.session_state:
    st.session_state["groq1_messages"] = []

llm = ChatGroq(
    temperature=0.1,
    model_name="Llama3-70b-8192",
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
)

if "groq1_chat_summary" not in st.session_state:
    st.session_state["groq1_chat_summary"] = []
else:
    callback = False
    for chat_list in st.session_state["groq1_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["groq1_messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_messages(message, role)


def paint_history():
    for message in st.session_state["groq1_messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def embed_file(url):
    loader = OnlinePDFLoader(url)
    pages = loader.load_and_split()
    vectorstore = FAISS.from_documents(pages, HuggingFaceEmbeddings())
    retriver = vectorstore.as_retriever()
    return retriver

with st.sidebar:
    pdf_url = st.text_input(
        "Enter the PDF URL"
    )


prompt = ChatPromptTemplate.from_template(
    """         
    Given the context, please provide a detailed and easy-to-understand explanation regarding the question. 
    Ensure to:
    1. Clarify the essence of the question, asking for additional details if necessary.
    2. Offer background information relevant to the question to set the stage for your explanation.
    3. Break down complex concepts or processes into manageable steps for clarity.
    4. Use examples and analogies to elucidate difficult concepts, making them relatable to everyday experiences.
    5. Summarize the main points and conclude your explanation, indicating areas for further exploration or where additional information might be beneficial.
    
    Aim to make your response comprehensive yet accessible, utilizing simple language to enhance understanding for all audience levels.

    Context: {context}
    Question: {question}
    """
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def save_context(question, result):
    st.session_state["groq1_chat_summary"].append(
        {
            "question": question,
            "answer": result,
        }
    )


def invoke_chain(question):
    result = chain.invoke(
        {"question": question},
    )
    save_context(message, result.content)


st.title("Groq-Llama3 Chatbot")

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

if "authentication_status" not in st.session_state:
    st.markdown("<p class='big-font'>You need to log in from the 'Home' page in the left sidebar.</p>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        Welcome!
        
        Use this chatbot to ask questions to an AI about your Questions!
        
        """
    )

    authentication_status = st.session_state["authentication_status"]
    if authentication_status:
        if pdf_url:
            retriever = embed_file(pdf_url)
            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()
            message = st.chat_input("Ask anything about pdf url...")
            if message:
                send_message(message, "human")
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                )
                with st.chat_message("ai"):
                    print(message)
                    chain.invoke(message)
    else:
        st.markdown("<p class='big-font'>You need to log in from the 'Home' page in the left sidebar.</p>", unsafe_allow_html=True)


