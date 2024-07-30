import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationTokenBufferMemory

st.set_page_config(
    page_title="ChatGPT4-mini",
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


if "gpt3_messages" not in st.session_state:
    st.session_state["gpt3_messages"] = []

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=500,
    return_messages=True,
)

if "gpt3_chat_summary" not in st.session_state:
    st.session_state["gpt3_chat_summary"] = []
else:
    callback = False
    for chat_list in st.session_state["gpt3_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["gpt3_messages"].append(
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
    for message in st.session_state["gpt3_messages"]:
        send_message(message["message"], message["role"], save=False)


#  """You are an engineering expert. explain my question in detail in Korean. 
# And provide additional definitions for technical terms.""",
with st.sidebar:
    prompt_text = st.text_area(
        "Prompt",
        """
        Assume the role of a professional engineer and provide a detailed yet easily understandable explanation for the following question. In your response, please:

        1. Begin with a brief overview to introduce the topic and its relevance in the field of engineering.
        2. Dive into the technical aspects, using clear and precise language to explain the concepts or processes involved. Highlight how these principles apply in practical engineering scenarios.
        3. Employ diagrams, equations, or real-world examples wherever possible to illustrate your points and enhance comprehension. Feel free to describe these visual aids in detail for a textual understanding.
        4. Address common misconceptions or challenges associated with the topic, offering professional insights into overcoming these issues.
        5. Conclude with a summary that encapsulates the key takeaways and suggests further resources or reading materials for those interested in delving deeper into the subject.

        Ensure your explanation balances depth of content with accessibility, aiming to educate both engineers and non-specialists alike.
        """
        )

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
    return memory.load_memory_variables({})["history"]


def save_context(question, result):
    st.session_state["gpt3_chat_summary"].append(
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


st.title("ChatGPT4-mini Chatbot")

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
