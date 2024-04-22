from dotenv import load_dotenv  # for loading env variables. pip install python-dotenv
from pathlib import Path
import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# https://docs.trychroma.com/troubleshooting#sqlite
# https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
# override sqlite 3.34 with newer version, per chromadb requirements
# this is bc alma linux doesnt yet support sqlite3.35, only 3.34
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import openai

# get openai key
home = Path(__file__).resolve().parent
load_dotenv(dotenv_path=home / '.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# find pdf file
data_dir = home / 'data'
data_file = '2024-stress-test-scenarios-20240215.pdf'
data_file_path = data_dir / data_file

loader = PyPDFLoader(data_file_path.absolute())
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # specify how many characters per chunk and overlap
documents = loader.load()
texts = text_splitter.split_documents(documents)

# select which embeddings we want to use
embeddings = OpenAIEmbeddings()

# Create embeddings for each chunk and insert into the Chroma vector database.
db = Chroma.from_documents(texts, embeddings)

# Create a language model and a retriever
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
retriever = db.as_retriever()

# Create a QA chain
chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type='stuff')

query = "Describe this year's severely adverse scenario, please sir."
query_response = chain.invoke(query)
print(query_response['result'])

