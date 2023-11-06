###################################################################################################
#  Brian Lesko               2023-11-05
#  This code implements a chat-app, with a text similarity search engine for querying a document. 
#  Think of it as an upgrded Cmd+F search. Written in pure Python & created for learning purposes.
###################################################################################################

import pandas as pd
import numpy as np
import streamlit as st
import tiktoken as tk
import openai
#import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from about import about
from api_key import openai_api_key
openai.api_key = openai_api_key

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

@st.cache_resource
def tokenize(text):
    # use tiktoken package
    # https://platform.openai.com/tokenizer
    enc = tk.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    return tokens

@st.cache_resource
def chunk_tokens(tokens, chunk_length=40, chunk_overlap=10):
    # turn the document into chunks with some overlap
    # split the document into chunks
    chunks = []
    for i in range(0, len(tokens), chunk_length - chunk_overlap):
        chunks.append(tokens[i:i + chunk_length])
    return chunks

@st.cache_resource
def detokenize(tokens):
    enc = tk.encoding_for_model("gpt-4")
    text = enc.decode(tokens)
    return text

@st.cache_resource
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk))
    return embeddings

def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

def input_api_key():
    st.write('  ') 
    st.markdown("""---""")
    if not openai_api_key:
        openai_api_key = st.text_input("# OpenAI API Key", key="chatbot_api_key", type="password")
        col1, col2 = st.columns([1,5], gap="medium")
        with col2:
            "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    return openai_api_key

def get_text(upload):
    # if the upload is a .txt file
    if upload.name.endswith(".txt"):
        document = upload.read().decode("utf-8")

    # This is so much worse than a .txt file, uses machine vision 
    if upload.name.endswith(".docx") or upload.name.endswith(".doc") or upload.name.endswith(".pdf") or upload.name.endswith(".png") or upload.name.endswith(".jpg"):
        #!pip install docx2txt
        #!pip install python-magic
        with st.spinner('Extracting Text...'):
            document = docx2txt.process(upload)
            document = document.replace("\n", " ")
            document = document.replace("  ", " ")
            document = document.encode('ascii', 'ignore').decode()
            document = document.replace("&", "and")
    return document

def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

def get_relevant_contexts(text_chunks, query_embedding, doc_embeddings, n):
    # for each row of doc_embeddings, calculate the cosine similarity between the prompt embedding and the document embedding
    similarities = []
    for doc_embedding in doc_embeddings:
        similarities.append(cosine_similarity([query_embedding], [doc_embedding])[0][0])
        
    # the indicies of the top n most similar chunks
    idx_top_n_scores = np.argsort(similarities)[-n:][::-1]

    # Combine the top n chunks into a single string called "Context"
    context = ""
    for idx in idx_top_n_scores:
        context += text_chunks[idx] + " "

def augment_query(contexts, query):
    augmented_query = (
        f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}" + """you need to be clear and concise, sometime funny.
        If you need to make an assumption you must say so."""
    )
    return augmented_query

def generate_response(augmented_query,query):
    st.session_state.messages.append({"role": "user", "content": augmented_query})
    response = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state.messages)
    # delete the last message from the session state, so that only the prompt and response are displayed on the next run: no context
    st.session_state.messages.pop()
    st.session_state.messages.append({"role": "user", "content": query})
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    return msg

def print_embedding_info(tokens,text_chunks,doc_embeddings,document):
    st.write('  ') 
    st.subheader('Document Embeddings')
    st.write("tokens: ", np.array(tokens).size)
    st.write("word chunks: ", len(text_chunks))
    st.write("embeddings: ", np.array(doc_embeddings).shape)
    st.write('  ') 
    st.subheader('Document')
    st.write(document)

def main():
    st.set_page_config(page_title="Brian Lesko", page_icon="🤖", layout="wide")
    hide_streamlit_header_footer()

    with st.sidebar:
        about()
        if not openai.api_key: openai.api_key = input_api_key()
        upload = st.file_uploader("Upload a document")
    if upload:
        document = get_text(upload)
        tokens = tokenize(document)
        token_chunks = chunk_tokens(tokens)
        text_chunks = [detokenize(chunk) for chunk in token_chunks]
        doc_embeddings = embed_chunks(text_chunks)
        with st.sidebar:
            print_embedding_info(tokens,text_chunks,doc_embeddings,document)
        display_existing_messages()
        query = st.chat_input("Write a message")
        if query:
            st.chat_message("user").write(query)
            query_embedding = get_embedding(query)
            n = 2
            contexts = get_relevant_contexts(text_chunks,query_embedding, doc_embeddings, n)
            augmented_query = augment_query(contexts, query)
            response = generate_response(augmented_query,query)
            st.chat_message("assistant").write(response.content)
            with st.expander("Context"):
                st.write(contexts)
main()