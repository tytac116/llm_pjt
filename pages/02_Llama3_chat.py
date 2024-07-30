import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache("cache.db"))

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


if "llama3_messages" not in st.session_state:
    st.session_state["llama3_messages"] = []


prompt_message_llammachat = """Explanation: As a hardware and software expert, your task is to provide detailed and easily understandable explanations in response to inquiries or statements. 
You are expected to demystify complex concepts related to both hardware and software, making them accessible to a broad audience. 
Your objective is to enhance the questioner's comprehension by not only simplifying explanations but also by providing relevant examples whenever possible.

Role: Hardware and Software Expert
Objective: To assist questioners in comprehensively understanding hardware and software concepts

Guidelines:
1. Break down complex technical topics into simple, digestible explanations.
2. Use clear and accessible language that can be understood by individuals without a technical background.
3. Provide real-life examples or hypothetical scenarios to illustrate your explanations and make abstract concepts tangible.
4. Address the 'how' and 'why' behind processes and technologies to deepen the questioner's understanding.
5. When discussing software, explain how it interacts with hardware to perform its functions.

Example Topics:
- How do CPUs process instructions?
- What is the role of an operating system in a computer?
- Can you explain how a smartphone uses both hardware and software to capture and process photos?
- Describe the process of compiling code into an executable program.

When answering, it's important to remember that your goal is to make the information as accessible as possible. 
Strive to not only answer the question but also to educate the questioner, providing them with a foundation that enables them to grasp more complex concepts in the future."""

prompt_translate="Translate the sentence you answered before into Korean."

def set_prompt():
    return prompt_message_llammachat


with st.sidebar:
    prompt_text = st.text_area(
        "Prompt", set_prompt(),
    )


llm = ChatOllama(
    temperature=0.1,
    model="llama3.1:latest",
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True,
)

if "llama3_chat_summary" not in st.session_state:
    st.session_state["llama3_chat_summary"] = []
    st.session_state["last_answer"] = ""
else:
    callback = False
    if st.session_state["llama3_chat_summary"]:
        st.session_state["last_answer"] = st.session_state["llama3_chat_summary"][-1]["answer"]

    for chat_list in st.session_state["llama3_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["llama3_messages"].append(
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
    for message in st.session_state["llama3_messages"]:
        send_message(message["message"], message["role"], save=False)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {prompt_text}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

def load_memory(_):
    loaded_memory = memory.load_memory_variables({})["history"]
    return loaded_memory


def save_context(question, result):
    st.session_state["llama3_chat_summary"].append(
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

@st.spinner(text="translating...")
def translate_answer():
    sentence = st.session_state["last_answer"]
    print(sentence)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                {prompt_translate}
                """,
            ),
            ("human", "{question}"),
        ]
    )
    translate_llm = ChatOllama(
        temperature=0.1,
        model="llama3.1:latest",
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    
    chain = prompt | translate_llm
    chain.invoke(
        {"question": sentence}
    )

st.title("Llama3 Chatbot")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your Questions!
    
    """
)

send_message("I'm ready! Ask away!", "ai", save=False)
paint_history()

authentication_status = st.session_state["authentication_status"]
if authentication_status:
    message = st.chat_input("Ask anything about something...")

    if message:
        send_message(message, "human")
        chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

        with st.chat_message("ai"):
            callback = True
            invoke_chain(message)
    
    if st.button("translate"):
        with st.chat_message("ai"):
            callback = True
            translate_answer()

