###############################################################################################
# Brian Lesko
# 2023-11-05
# implements a chat-app, with text a similarity search engine
# for querying a document. Think of it as an upgrded Cmd+F search. 
# It's written in [Pure Python](). Created for Learning Purposes.
###############################################################################################

import pandas as pd
import numpy as np
import streamlit as st
import tiktoken as tk
import openai
from sklearn.metrics.pairwise import cosine_similarity

from api_key import openai_api_key
openai.api_key = openai_api_key

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Tokenize
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

# detokenize chunks
@st.cache_resource
def detokenize(tokens):
    enc = tk.encoding_for_model("gpt-4")
    text = enc.decode(tokens)
    return text

# Embed the chunks of the document
@st.cache_resource
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk))
    return embeddings

with st.sidebar:

    from about import about
    about()
    
    st.write('  ') 
    st.markdown("""---""")
    if not openai_api_key:
        openai_api_key = st.text_input("# OpenAI API Key", key="chatbot_api_key", type="password")
        col1, col2 = st.columns([1,5], gap="medium")
        with col2:
            "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    # File Upload
    upload = st.file_uploader("Upload a document")

if not upload: 
    st.text('''Upload a document to get started''')

with st.sidebar:
    if upload:
        # if the upload is a .txt file
        if upload.name.endswith(".txt"):
            document = upload.read().decode("utf-8")

        # This is so much worse than a .txt file, uses machine vision 
        if upload.name.endswith(".docx") or upload.name.endswith(".doc") or upload.name.endswith(".pdf") or upload.name.endswith(".png") or upload.name.endswith(".jpg"):
            #!pip install docx2txt
            #!pip install python-magic
            import docx2txt
            with st.spinner('Extracting Text...'):
                document = docx2txt.process(upload)
                # get rid of the extra newlines
                document = document.replace("\n", " ")
                # make sure there are no double spaces
                document = document.replace("  ", " ")
                # get rid of special characters
                document = document.encode('ascii', 'ignore').decode()
                # get rid of & characters
                document = document.replace("&", "and")

        st.write('  ') 
        st.subheader('Document Embeddings')

        tokens = tokenize(document)
        n_tokens = np.array(tokens).size
        st.write("tokens: ", n_tokens)

        token_chunks = chunk_tokens(tokens)
        n_token_chunks = len(token_chunks)

        word_chunks = [detokenize(chunk) for chunk in token_chunks]
        n_word_chunks = len(word_chunks)
        st.write("word chunks: ", n_word_chunks)

        doc_embeddings = embed_chunks(word_chunks)
        st.write("embeddings: ", np.array(doc_embeddings).shape)

        st.write('  ') 
        st.sidebar.subheader('Document')
        st.sidebar.write(document)

if upload:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display all the historical messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Write a message"):
        st.chat_message("user").write(prompt)

        prompt_embedding = get_embedding(prompt)

        # for each row of doc_embeddings, calculate the cosine similarity between the prompt embedding and the document embedding
        similarities = []
        for doc_embedding in doc_embeddings:
            similarities.append(cosine_similarity([prompt_embedding], [doc_embedding])[0][0])
        
        # the indicies of the top n most similar chunks
        n = 2
        idx_top_n_scores = np.argsort(similarities)[-n:][::-1]

        # Combine the top n chunks into a single string called "Context"
        context = ""
        for idx in idx_top_n_scores:
            context += word_chunks[idx] + " "

        # combine the original prompt and context into final_prompt
        final_prompt = """Answer the question or statement using the 
        context provided, you need to be clear and concise, sometime funny.
        If you need to make an assumption you must say so. 
        The question or statement you must answer is: """ + prompt + "." + """ remember to be concise and clear.
        
        Use this context: """ + context

        st.session_state.messages.append({"role": "user", "content": final_prompt})

        # call the LLM with the prompt and context
        response = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state.messages)

        # delete the last message from the session state, so that only the prompt and response are displayed on the next run: no context
        st.session_state.messages.pop()
        st.session_state.messages.append({"role": "user", "content": prompt})

        # write the response to the screen
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg.content)

        # Write the context used to the screen
        with st.expander("Context"):
            st.write(context)
    
