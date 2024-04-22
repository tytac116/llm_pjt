import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Groq-Llama3",
    page_icon="📃",
)


if "groq_messages" not in st.session_state:
    st.session_state["groq_messages"] = []

llm = ChatGroq(
    temperature=0.1,
    model_name="Llama3-70b-8192",
)

memory = ConversationBufferWindowMemory(
    llm=llm,
    k=2,
    return_messages=True,
)

if "groq_chat_summary" not in st.session_state:
    st.session_state["groq_chat_summary"] = []
else:
    for chat_list in st.session_state["groq_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["groq_messages"].append(
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
    for message in st.session_state["groq_messages"]:
        send_message(message["message"], message["role"], save=False)

@st.spinner("Preparing your question...")
def make_response(prompt_text, message):
    ai_message = llm(
        [
            SystemMessage(content=prompt_text),
            HumanMessage(content=message),
        ]
    )
    return ai_message.content


st.title("Groq-Llama3 Chatbot")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your Questions!
    
    """
)

with st.sidebar:
    prompt_text = st.text_area(
        "Prompt",
        """Explanation: As a hardware and software expert, your task is to provide detailed and easily understandable explanations in response to inquiries or statements. 
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
        Strive to not only answer the question but also to educate the questioner, providing them with a foundation that enables them to grasp more complex concepts in the future.
        """)

send_message("I'm ready! Ask away!", "ai", save=False)
paint_history()

authentication_status = st.session_state["authentication_status"]
if authentication_status:
    message = st.chat_input("Ask anything about something...")

    if message:
        send_message(message, "human")
        with st.chat_message("ai"):
            ai_message = make_response(prompt_text, message)
            st.write(ai_message)
            save_messages(ai_message, "ai")


           
