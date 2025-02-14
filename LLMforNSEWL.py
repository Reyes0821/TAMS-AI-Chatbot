import os
import gradio as gr
from llama_index.core import GPTVectorStoreIndex, PromptHelper, load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from llama_index.readers.file.docs.base import PDFReader
from pathlib import Path


# Set the environment variables for Google Gemini
# Create a key in google gemini here: 'https://aistudio.google.com/app/apikey'
os.environ['GOOGLE_API_KEY'] = 'YOUR_GOOGLE_GEMINI_API'

# Define values of parameters
max_input_size = 8192
num_output = 2048  
max_chunk_overlap = 50
context_window = 8192
chunk_overlap_ratio = 0.2


# Set up the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.3,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    },
    convert_system_message_to_human=True
)

embeddings = LangchainEmbedding(GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv('GOOGLE_API_KEY')
))

prompt_helper = PromptHelper(context_window, num_output, chunk_overlap_ratio)
Settings.llm = llm
Settings.embed_model = embeddings
Settings.prompt_helper = prompt_helper

# This function returns the indexed data after completing the indexing process.
def data_indexing(folder_path):
    pdf_reader = PDFReader()
    # docs = pdf_reader.load_data(file=Path(file_path))
    docs = []
    
    # Iterate through all PDF files in the specified folder
    for pdf_file in Path(folder_path).glob("*.pdf"):
        print(f"Processing: {pdf_file}")
        docs.extend(pdf_reader.load_data(file=pdf_file))
    
    index = GPTVectorStoreIndex.from_documents(docs, settings=Settings)

    # Save index to disk
    index.set_index_id('vector_index')
    index.storage_context.persist(persist_dir="./nsewl_gpt_store")

    return index

# This function returns a response by querying the data store.
def data_querying(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="./nsewl_gpt_store")
    index = load_index_from_storage(storage_context, settings=Settings)
    
    print("Index loaded from storage.")
    
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
   
    return response.response 

folder_path = 'TAMS NSEWL User Guide'

# Execute the data indexing process.
index = data_indexing(folder_path)  

# Start up the Gradio web app for testing.
iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="TAMS NSEWL User Guide Chatbot")

iface.launch(share=True, debug=True)
