from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import OpenAI
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import openai
import os
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# client = openai.Client()

def load_pdf(url):
    loader = UnstructuredPDFLoader("frugal_gpt.pdf")
    data = loader.load()
    return data

data = load_pdf("frugal_gpt.pdf")
text_splitter = TokenTextSplitter(chunk_size = 50, chunk_overlap = 0)
doc = text_splitter.split_documents(data)


embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(doc, embedding=embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages = True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(), vectordb.as_retriever(), memory = memory)

query = "What is this document about?"
result = pdf_qa({"question" : query})
print(result['answer'])

def doc_chat():
  while True:
    with get_openai_callback() as cb:
      query = input("Ask your question: ")
      result = pdf_qa({'question' : query})
      print(f'Total: {cb.total_tokens}, Prompt Tokens: {cb.prompt_tokens}, Completion Tokens: {cb.completion_tokens}')
      print(result['answer'])

