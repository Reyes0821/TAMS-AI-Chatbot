# TAMS AI Chatbot

## Overview
The **TAMS AI Chatbot** is designed to assist users in navigating the **Track Access Management System (TAMS)** by providing instant responses to common queries and guiding new users through system functionalities. The chatbot enhances user experience, reduces onboarding time, and ensures smooth interaction with TAMS.

## Features
- **AI-Powered Responses**: Uses LLM-based Retrieval-Augmented Generation (RAG) for accurate and relevant answers.
- **Training Assistance**: Facilitates learning for new users through interactive Q&A.

## Technologies Used
- **Google GenAI** for AI model processing.
- **Natural Language Processing (NLP)** for chatbot responses.
- **Web-based UI** for chatbot interaction.

## Installation
### Prerequisites
- Python 3.8+
- Required dependencies (listed in `requirements.txt`)

### Setup
```bash
# Clone the repository
git clone https://github.com/TAMS-AI-Chatbot.git
cd tams-ai-chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the chatbot
python app.py
```
- Access the chatbot through the web interface.
- Start asking questions related to TAMS.


## Google API Configuration
To use Google AI services, set up your API credentials:
1. Create a Google Cloud account and enable the necessary APIs.
2. Generate an API key or OAuth credentials.
3. Store your credentials in an environment variable or a configuration file.
4. Update your application settings to use the API key.

## TAMS User Guides
- **TAMS NSEWL User Guide**: This section is intentionally left empty due to company confidentiality.
- **TAMS CCL User Guide**: This section is intentionally left empty due to company confidentiality.

## Contributors
- [Reyes Aw Rui Heng]