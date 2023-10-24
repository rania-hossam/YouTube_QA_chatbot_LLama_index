
import streamlit as st
import gradio as gr
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_hub.youtube_transcript import YoutubeTranscriptReader

from llama_index import VectorStoreIndex

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.llm_predictor import LLMPredictor
from langchain.llms import LlamaCpp

model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(
    model_name=model_name
)

llm = LlamaCpp(
   
    model_path="/Users/aiswaryaramachandran/.cache/lm-studio/models/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGUF/codeup-llama-2-13b-chat-hf.Q8_0.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    temperature=0.1,
    max_tokens=256,
    top_p=1,
    verbose=True, 
    f16_kv=True,
    n_ctx=4096,
    use_mlock=True,n_threads=4

)
llm_predictor=LLMPredictor(llm=llm)
embed_model = LangchainEmbedding(hf) 
service_context = ServiceContext.from_defaults(embed_model=embed_model,llm_predictor=llm_predictor)


def load_data(youtube_url):
    
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[youtube_url])
    
        
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

def predict(youtube_url,query):
    
    index=load_data(youtube_url)
    chat_engine = index.as_chat_engine(verbose=True,service_context=service_context,chat_mode="condense_question")
    response = chat_engine.stream_chat(query)
    output=""
    for token in response.response_gen:
        output=output+token
        yield output.strip()
    #return str(response)


### Have an input box to allow entering the youtube URL
      
interface = gr.Interface(fn=predict, 
                         inputs= [gr.inputs.Textbox(label="input youtube URL"), gr.inputs.Textbox(label="Question:")],
                         outputs =gr.outputs.Textbox(label="Chatbot Response")
                        )
interface.launch(enable_queue = True)