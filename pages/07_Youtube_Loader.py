import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders.youtube import YoutubeLoader

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
)

st.set_page_config(
    page_title="Youtube Summary GPT",
    page_icon="📆",
)

st.title("Youtube Summary GPT")

st.markdown(
    """
    Welcome to Youtube Summary GPT, provide a video and I will give you a transcript, a summary and a chat bot to ask any question about it!
    
    Get Started by providing a Youtube url in the sidebar.
    """
)


with st.sidebar:
    url = st.text_input(
        "Youtude Url"
    )

if url:
    loader = YoutubeLoader.from_youtube_url(
    url, language=["en", "ko"], translation="ko",)
    
    docs = loader.load()

    if docs:
        with st.chat_message("ai"):
            for doc in docs:
                st.markdown(doc.page_content + "\n\n")