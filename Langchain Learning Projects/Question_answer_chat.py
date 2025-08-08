from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import streamlit as st 
import os 

load_dotenv()

st.title('Asking Question to Loaded PDF')

model = ChatGroq(
    model = 'llama3-8b-8192',
    temperature=0.7,
    api_key = os.getenv("GROQ_API_KEY")
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following question {question}',
    input_variables=['question']
)

# Initially store the system message to the session_state
if 'message' not in st.session_state:
    st.session_state.message =[SystemMessage(content="You are a Helpful AI assistant")]
    
# Display the chat History 
for msg in st.session_state.message:
    if isinstance(msg,HumanMessage):
        with st.chat_message('User'):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message('Assistant'):
            st.write(msg.content)
            
            
# take input form user 
user_input = st.chat_input('Ask anythink')

if user_input :
    
    st.session_state.message.append(HumanMessage(content=user_input))
    with st.chat_message('User'):
        st.write(user_input)

    chain = prompt | model | parser

    with st.chat_message('Assistant'):
        with st.spinner('Generating...'):
            response = chain.invoke({'question': user_input})
            st.write(response)
        st.session_state.message.append(AIMessage(content=response))
    
    