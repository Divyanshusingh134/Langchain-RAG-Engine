import logging
import argparse
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

store = LocalFileStore("./my_embedding_cache/")

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

# Base embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=768
)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store,
    namespace= embeddings.model
)

class RAGPipeline:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name = "Cricket-INFO",
            embedding_function=cached_embedder,
            persist_directory=".//chroma_langchain_db"
        )

    def run_evaluator(self, file_name: list, queries_file: str, outfile: str):
        all_chunks = []
        for file in file_name:
            chunks = Load_Split_document(file)
            all_chunks.extend(chunks)

        if not all_chunks:
            logging.error("No chunks loaded.")
            raise
        ids = [hashlib.md5(chunk[0].page_content.encode("utf-8")).hexdigest() for chunk in all_chunks]
        self.vector_store.add_documents(documents=all_chunks, ids=ids)
        logging.info(f"Successfully added {len(all_chunks)} chunks to Chroma.")
        pass
    
if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.run_evaluator(args.data, args.queries, args.output)

    

