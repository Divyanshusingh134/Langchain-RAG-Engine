import logging
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
parser.add_argument("--data", nargs='+', required=True, help="path of text file")
parser.add_argument("--queries", help="path of queries file")
parser.add_argument("--output", nargs='+', help="path of output file")
args = parser.parse_args()

def Load_Split_document(text_file: str) -> list:
    loader = TextLoader(file_path=text_file)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    final_chunks = text_splitter.split_documents(documents=document)
    logging.info(f"Successfully split into {len(final_chunks)} chunks.")
    return final_chunks

def genai(chunks: list):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768
    )
    return embeddings.embed_documents(chunks)

# class TextProcessor:
#     @staticmethod
#     def LoadDocument(text_file: str):
#         loader = TextLoader(file_path=text_file)
#         return loader.load()

#     @staticmethod
#     def TextSplitter(document: list):
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#         )
#         final_chunks  = text_splitter.split_documents(documents=document)
#         logging.info(f"Successfully split into {len(final_chunks)} chunks.")
#         return final_chunks

class RAGPipeline:
    def __init__(self) -> None:
        pass
    def run_evaluator(self, file_name: list, queries_file: str, outfile: str):
        final_chunks = Load_Split_document(args.data[0])
        vectors = genai(final_chunks)
        pass
    
if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.run_evaluator(args.data, args.queries, args.output)

    

