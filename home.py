import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# secret_key = secrets.token_hex(16)
# print(secret_key)

st.set_page_config(
    page_title="LLM Collection for Office",
    page_icon="💀",
)

with open("./config.ymal") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)

name, authentication_status, username = authenticator.login()

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = authentication_status

if authentication_status:
    authenticator.logout("Logout", "main")
    st.write(f"Welcome *{name}*")
    st.markdown(
        """
# 1. ChatGPT 3.5
# 2. ChatGPT 4
# 3. Gemini Pro
# 4. Llama3
    """
    )
elif authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
