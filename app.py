import os
from flask import Flask, render_template, request, jsonify
from llama_index.core import GPTVectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Initialize Flask app
app = Flask(__name__)

# Set the environment variables for Google Gemini
# Create a key in google gemini here: 'https://aistudio.google.com/app/apikey'
os.environ['GOOGLE_API_KEY'] = 'YOUR_GOOGLE_GEMINI_API'  

# Set up the Google Gemini model for embeddings
embeddings = LangchainEmbedding(
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Ensure this is the correct embedding model
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
)

# Set up the Google Gemini LLM for chat 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.3
)

# Set settings for LlamaIndex to use Google Gemini models
Settings.llm = llm
Settings.embed_model = embeddings

# Load the models and indices for both CCL and NSEWL
ccl_storage_context = StorageContext.from_defaults(persist_dir="./ccl_gpt_store")
ccl_index = load_index_from_storage(ccl_storage_context, settings=Settings)
ccl_query_engine = ccl_index.as_query_engine()

nsewl_storage_context = StorageContext.from_defaults(persist_dir="./nsewl_gpt_store")
nsewl_index = load_index_from_storage(nsewl_storage_context, settings=Settings)
nsewl_query_engine = nsewl_index.as_query_engine()

# Route for home page with buttons
@app.route('/')
def home():
    return render_template('index.html')

# Route for CCL chatbot
@app.route('/ccl_chatbot', methods=['POST'])
def ccl_chatbot():
    user_input = request.json.get("message", "")
    response = ccl_query_engine.query(user_input)
    return jsonify({"response": response.response})

# Route for NSEWL chatbot
@app.route('/nsewl_chatbot', methods=['POST'])
def nsewl_chatbot():
    user_input = request.json.get("message", "")
    response = nsewl_query_engine.query(user_input)
    return jsonify({"response": response.response})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
