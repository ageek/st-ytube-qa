import streamlit as st

from langchain_main import creating_db, get_response
from langchain.embeddings import OpenAIEmbeddings

import textwrap

openai_api_key=''

# Page title
st.set_page_config(page_title='🦜🔗 LLM based YouTube QA Assistant')
st.title('🦜🔗 LLM based YouTube QA Assistant')

with st.sidebar:
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


with st.form('my_form', clear_on_submit=False):
  # YouTube video url
  video_url = st.text_input('Enter YouTube url:', type='default')
  # Query text
  query_text = st.text_input('Enter your question:')

  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    #create db off video url
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
    mydb = creating_db(video_url, embeddings)

    response, docs = get_response(mydb, query_text, openai_api_key, k=5)
    st.info(response)

# delete openai_api_key  
del openai_api_key
