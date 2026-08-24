import os 
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("API Key not found")

parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
parser.add_argument("--data", nargs='+', required=True, help="path of text file")
parser.add_argument("--queries", help="path of queries file")
parser.add_argument("--output", nargs='+', help="path of output file")
args = parser.parse_args()

class TextProcessor:
    @staticmethod
    def LoadDocument(text_file: str):
        loader = TextLoader(file_path=text_file)
        return loader.load()

    @staticmethod
    def TextSplitter(document: list):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        return text_splitter.split_documents(documents=document)


class RAGPipeline:
    def __init__(self) -> None:
        pass
    
if __name__ == "__main__":
    raw_docs = TextProcessor.LoadDocument(args.data[0])
    final_chunks = TextProcessor.TextSplitter(raw_docs)

