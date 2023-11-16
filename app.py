###################################################################################################
#  Brian Lesko               2023-11-14
#  This code implements an embedding shema that is used to compare the similarity of textual data.
#  Think of it as an upgraded Cmd+F search. Written in pure Python & created for learning purposes.
###################################################################################################

import pandas as pd
import numpy as np
import streamlit as st
import tiktoken as tk
import openai
from sklearn.metrics.pairwise import cosine_similarity
from customize_gui import gui
from api_key import openai_api_key
openai.api_key = openai_api_key

gui = gui()

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

@st.cache_resource
def tokenize(text):
    enc = tk.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    return tokens

@st.cache_resource
def chunk_tokens(tokens, chunk_length=40, chunk_overlap=10):
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

def get_text(upload):
    # if the upload is a .txt file
    if upload.name.endswith(".txt"):
        document = upload.read().decode("utf-8")
    return document

def augment_query(contexts, query):
    augmented_query = (
        f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}" + """ In your answer please be clear and concise, sometime funny.
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

class document:
    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.tokens = tokenize(text)
        self.token_chunks = chunk_tokens(self.tokens, chunk_length=50, chunk_overlap=10)
        self.text_chunks = [detokenize(chunk) for chunk in self.token_chunks]
        self.chunk_embeddings = embed_chunks(self.text_chunks)
        self.embedding = get_embedding(self.text)
        self.df = pd.DataFrame({
            "name": [self.name], 
            "text": [self.text], 
            "embedding": [self.embedding], 
            "tokens": [self.tokens], 
            "token_chunks": [self.token_chunks], 
            "text_chunks": [self.text_chunks], 
            "chunk_embeddings": [self.chunk_embeddings]
            })

    def similarity_search(self, query, n=3):
        query_embedding = get_embedding(query)
        similarities = []
        for chunk_embedding in self.chunk_embeddings:
            similarities.append(cosine_similarity([query_embedding], [chunk_embedding])[0][0])
        # the indicies of the top n most similar chunks
        idx_sorted_scores = np.argsort(similarities)[::-1]
        context = ""
        for idx in idx_sorted_scores[:n]:
            context += self.text_chunks[idx] + "\n"
        return context

def main():
    gui.clean_format()
    with st.sidebar:
        gui.about(text="This code implements a text embedding similarity search. Think of it as an upgraded Cmd+F.")
        if not openai.api_key: openai.api_key = gui.input_api_key()
        upload = st.file_uploader("Upload a document")
    if upload:
        doc = document(upload.name, get_text(upload)) # document class defined above
        with st.sidebar:
            st.subheader("Document")
            st.write(doc.text)
        gui.display_existing_messages()
        query = st.chat_input("Write a message")
        if query:
            st.chat_message("user").write(query)
            contexts = doc.similarity_search(query, n=2)
            augmented_query = augment_query(contexts, query)
            response = generate_response(augmented_query,query)
            st.chat_message("assistant").write(response.content)
            with st.expander("Context"):
                st.write(contexts)
main()