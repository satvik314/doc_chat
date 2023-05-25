import streamlit as st
from streamlit_chat import message
import pickle
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
# from dotenv import load_dotenv
# load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

# Sidebar contents
with st.sidebar:
    st.subheader("This app will let you create custom chatbot on your documents! [👉]")
    st.markdown(
        '''
        - supports only PDF
        - page limit < 25 pages
        '''
    )
    add_vertical_space(8)
    st.markdown("""
    Want to learn how to do this? 
    - [Sign up for a free session!](https://docs.google.com/forms/d/e/1FAIpQLSeVbGaABr0wyg7FCn1Qhsj6YtWoK18G-qH0VrojoT9OHDFTyg/viewform)
    """)
    add_vertical_space(3)
    st.markdown("""
    If you want to access other apps I have created:
    - [Learn from YouTube](https://satvik314-yt-learn-main-dtw9a8.streamlit.app/)
    - [Auto To-Do](https://satvik314-auto-todo-app-wrrgo5.streamlit.app/)
    """)
    add_vertical_space(1)
    st.write("Reach out to me on [LinkedIn](https://www.linkedin.com/in/satvik-paramkusham-76a33610a/)")

def main():
    st.header("Talk to PDF! 🐒")

    st.markdown("""
    [Want to learn how to do this?](https://docs.google.com/forms/d/e/1FAIpQLSeVbGaABr0wyg7FCn1Qhsj6YtWoK18G-qH0VrojoT9OHDFTyg/viewform)
    """)

    #upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')

    if pdf is not None:
        # reading the pdf
        pdf_reader = PdfReader(pdf)

        if len(pdf_reader.pages)  > 25:
            st.error("Please load a PDF with < 25 pages.")
        else:
            # convert into raw text
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # chunking the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 100,
                length_function = len
            )
            chunks = text_splitter.split_text(text = text)

            # embeddings
            embeddings = OpenAIEmbeddings()

            VectorStore = FAISS.from_texts(chunks, embedding= embeddings)
            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                # st.write("Embeddings loaded from the disk")
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(texts= chunks,  embedding= embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

                # st.write("Embeddings Computation Complete")
#myblock
            if 'buffer_memory' not in st.session_state:
                st.session_state.buffer_memory = ConversationBufferWindowMemory(memory_key= 'chat_history', k=3, return_messages=True)

            conversation = ConversationalRetrievalChain.from_llm(OpenAI(), VectorStore.as_retriever(), memory = st.session_state.buffer_memory)
            # query = st.text_input("Ask a question")
            # if query:
            #     result = conversation({"question" : query})
            #     st.write(result["answer"])
# end
            
            if 'generated' not in st.session_state:
                st.session_state['generated'] = []
            if 'past' not in st.session_state:
                st.session_state['past'] = []

            query = st.text_input(f"You can now chat with {store_name}", key = "input", placeholder= "Ask your question!")

            if 'messages' not in st.session_state:
                st.session_state.messages = []

            if query:
                # query_short = query + "(Please answer in less than 50 words)"
                with st.spinner("typing..."):
                    messages = st.session_state['messages']
                    messages = update_chat(messages, "user", query)
                    response = conversation({"question" : query})['answer']
                    messages = update_chat(messages, "assistant", response)
                    st.session_state.past.append(query)
                    st.session_state.generated.append(response)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state['past'][i], is_user= True, key = str(i) + "_user")
                    message(st.session_state["generated"][i], key = str(i))

            st.divider()
                # with st.expander("Show Messages"):
                #     st.write(st.session_state.messages)
                            




def update_chat(messages, role, content):
    messages.append({"role" : role, "content": content})
    return messages


if __name__ == '__main__':
    main()


