import gradio as gr
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document


from llama_hub.youtube_transcript import YoutubeTranscriptReader

from llama_index import VectorStoreIndex

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.llm_predictor import LLMPredictor
from langchain.llms import LlamaCpp


## For embedding the video, we will use the Hugging Face Sentence Transformers
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(
    model_name=model_name
)

### We are using LlamaCPP to load the LLAMA-2-18 8 bit quantised model in GGUF format 
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
    use_mlock=True,n_threads=4,
    stop=["Human:","User:"]

)

## Create a service context object, that will allow us to use the Hugging Face embeddings and llama 2 model as our Language model
llm_predictor=LLMPredictor(llm=llm)
embed_model = LangchainEmbedding(hf) 
service_context = ServiceContext.from_defaults(embed_model=embed_model,llm_predictor=llm_predictor)
index=None




### The load data function , takes in youtube_url and allows us to index the youtube video.

def load_data(youtube_url):
    print("In Load Data")

    if youtube_url.strip()=="":
        st.error("Enter A youtube URL")
        return None
    else:
        try:
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(ytlinks=[youtube_url])
    
        
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            return index
        except:
            print("Enter a valid youtube URL")
            st.error("Enter a valid youtube URL")
            return None

#### We will have user enter the youtube_url and press submit => which loads the index
index=None


chat_engine=None

### we initiate twp session_state object : clicked and index.
### Clicked: This is set to true when the Submit button is clicked.
### Index: This stores the vector index. By keeping this session state, we allow the index to be persistent till a new yoputube url is enteres

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'index' not in st.session_state:
    st.session_state.index=None

### click_button-> changes state to Truw when button is clicked 
def click_button():
    st.session_state.clicked = True
with st.sidebar:
    st.title("Youtube QA with Llama 2 Bot")
             
    st.subheader("Upload Documents/URL")
    youtube_url = st.sidebar.text_input('Enter Youtube URL', '')
    submit_btn=st.sidebar.button('Submit',on_click=click_button)
    ## When the submit button is clicked, load the data and set the index session_state to the loaded index
    if st.session_state.clicked: 
        print("Going to Load Data")
        index=load_data(youtube_url)
        st.session_state.index=index
        print("Index ",index)
        
        st.session_state.clicked=False # set it to false , so that load_data function is not called for every single user message



#print("Index",index)

print("Index State ",st.session_state.index)
### If the index has been loaded, create the chat_engine object
if st.session_state.index!=None:
    chat_engine=st.session_state.index.as_chat_engine(verbose=True,chat_mode="context",service_context=service_context)
    print("CHat engine",chat_engine) 
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    full_response = ''
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("Calling CHat Engine")
            if chat_engine!=None:
                response = chat_engine.stream_chat(prompt)
                placeholder = st.empty()
                
                for item in response.response_gen:
                    full_response += item
                    placeholder.markdown(full_response.strip("Assistant:"))
                placeholder.markdown(full_response)
    if full_response!="":
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)