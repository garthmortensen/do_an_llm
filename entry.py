from dotenv import load_dotenv  # for loading env variables. pip install python-dotenv
from pathlib import Path  # vs OS, this is OOP path handler
import os
import logging
from datetime import datetime

# import the necessary classes from langchain
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
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import openai

# logging setup
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
home = Path(__file__).resolve().parent
log_filename = home / "logs" / f"{timestamp}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('testing')

# get openai key from uncommitted .env file
load_dotenv(dotenv_path=home / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # insert coin to continue ($5)

# load the documents from the PDF file and split them into chunks of 1000 characters with 0 character overlap
# this avoids openais 4096 token limit
data_file_path = home / "data" / "2024-stress-test-scenarios-20240215.pdf"
loader = PyPDFLoader(data_file_path.absolute())
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0
)  # specify how many characters per chunk and overlap


MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0  # temperature=0 means most deterministic
def setup_qa_chain():
    """Set up a question-answering chain

    returns:
        RetrievalQA: the retrieval question-answering chain
    """
    documents = loader.load()
    texts = text_splitter.split_documents(documents)

    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # create embeddings for each chunk and insert into the chroma vector database.
    db = Chroma.from_documents(texts, embeddings)

    # create a language model and a retriever
    llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
    retriever = db.as_retriever()

    return RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")


def ask_and_print(chain, question: str) -> None:
    """Ask a question to a chain and print the response.

    Args:
        chain: The chain object to invoke the question on.
        question (str): The question to ask.
    """
    response = chain.invoke(question)
    logger.info(f"{question}\n{response['result']}\n\n")


# create a QA chain
chain = setup_qa_chain()
ask_and_print(chain, "Explain in simple terms what stress testing is.")
ask_and_print(chain, "Give me a short summary of how this years really bad scenarios differ from previous years.")
