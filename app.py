import json
import boto3
import os
import sys
import streamlit as st
import numpy as np

from langchain_community.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock=boto3.client("bedrock-runtime",
                     region_name='us-east-1',
                     aws_access_key_id="AKIAQ3EGPZFSSXUT2POW",
                     aws_secret_access_key="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"
                     )

bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

def get_data():
    loader=PyPDFDirectoryLoader(r"C:\Users\snehdas\Downloads\data")
    pages=loader.load()

 
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    documents=text_splitter.split_documents(pages)
    return documents

def store_vector(documents):
    db=FAISS.from_documents(
        documents,
        bedrock_embeddings
    )
    db.save_local("faiss_index")

def get_llama3_llm():
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,region_name='us-east-1',
                model_kwargs={'max_gen_len':512})
    
    return llm

def get_mistral_llm():
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock,region_name='us-east-1',
                model_kwargs={'max_tokens':1024})
    
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
300 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response(llm,db,query):
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 6}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa_chain({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Communicate with PDF using AWS Bedrock🦜")

    user_question = st.text_input("Ask a Question from the PDF")

    with st.sidebar:
        st.title("Create or update knowledge base:")
        
        if st.button("Update"):
            with st.spinner("Processing..."):
                documents= get_data()
                store_vector(documents)
                st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization = True)
            llm=get_llama3_llm()
            
            #faiss_index = store_vector(documents)
            st.write(get_response(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization = True)
            llm=get_mistral_llm()
            
            #faiss_index = store_vector(documents)
            st.write(get_response(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()




