from langchain_cohere.llms import Cohere
import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from helper import *
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMChain
from langchain_community.llms import Cohere
from yolo import *
from gtts import gTTS

import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

# Language for text-to-speech
language = 'en'

st.title('Sight 👁️')

# Initialize API keys
COHERE_API_KEY = "xnm4LRAs5n5SV3IFyL11oLdvTAlkGriL5Gjz1tBJ"
co = cohere.Client(COHERE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key="88a8b75c-3069-4111-8d30-e13f5ef3c8fd")

# Create an index if it doesn't already exist
index_name = 'vision'
# if index_name not in pc.list_indexes().names():
    
#     pc.create_index(
#         name=index_name,
#         dimension=4096,
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-central1'
#         )
#     )

# Connect to Pinecone index
index = pc.Index(index_name)

# YOLO-based object detection and processing
vision_info = yolo()
formatted_string = ', '.join(f"{key}: {item[key]}" for item in vision_info for key in item)

template = f"""I am visually impaired, I am trying to walk with the help of my computer vision model. The computer vision model gives me all persons and objects with their position in my view. 
Please guide me through to my destination. All objects are in front of me so I will be walking forward. First, you will list all the objects and their position, making sure you advise me on being careful if two objects are in the same area (left, right, center). Then if I ask a question, as to go somewhere, please help me in two to three sentences so I can find my way. Also if I ask, let me know if you see the object before. If there is no question, then respond accordingly.

+ {formatted_string}
""" + """
{prompt}
Answer:
"""

# LLM setup using Cohere
llm = Cohere(model="command-nightly", cohere_api_key=COHERE_API_KEY)

# Message history and memory
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

# Prompt setup
prompt = PromptTemplate(template=template, input_variables=['prompt'])
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# UI for displaying message history
view_messages = st.expander("View the message contents in session state")

if 'key' not in st.session_state:
    st.session_state.key = ""

# Display previous messages
for msg in msgs.messages:
    st.session_state.key = st.session_state.key + " " + str(msg.type) + ": " + str(msg.content) + ","
    if msg.type != 'system':
        st.chat_message(msg.type).write(msg.content)

# Store messages in Pinecone
add_embeddings(st.session_state.key, index)

# Get user input
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # Get response from the language model
    response = llm_chain.run(prompt)
    
    # Convert response to speech
    myobj = gTTS(text=response, lang=language, slow=False)
    myobj.save("audio.mp3")
    os.system("start audio.mp3")  # Adjust for your OS if needed
    st.chat_message("ai").write(response)

with view_messages:
    """
    Memory initialized with:
    
python
    msgs = StreamlitChatMessageHistory(key="messages")
    memory = ConversationBufferMemory(chat_memory=msgs)


    Contents of st.session_state.langchain_messages:
    """
    view_messages.json(st.session_state.langchain_messages)