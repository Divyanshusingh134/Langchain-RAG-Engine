import logging
import argparse
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class RAGPipeline:
    def __init__(self):
        self.store = LocalFileStore("./my_embedding_cache/")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )

        self.cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=self.embeddings,
            document_embedding_cache=self.store,
            namespace= self.embeddings.model
        )
        self.vector_store = Chroma(
            collection_name = "Cricket-INFO",
            embedding_function=self.cached_embedder,
            persist_directory="./chroma_langchain_db"
        )

        self.model = ChatGoogleGenerativeAI(
            model = "gemini-3.5-flash-lite",
            temperature= 0.0,
            max_retries = 3,
            max_tokens = None,
            timeout=30.0
        )

    def _Load_Split_document(self, text_file: str) -> list:
        loader = TextLoader(file_path=text_file)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )

        final_chunks = text_splitter.split_documents(documents=document)
        logging.info(f"Successfully split into {len(final_chunks)} chunks.")
        return final_chunks
    

    def run_evaluator(self, file_name: list, queries_file: str, outfile: str):
        if self.vector_store._collection.count == 0:
            all_chunks = []
            for file in file_name:
                chunks = self._Load_Split_document(file)
                all_chunks.extend(chunks)

            if not all_chunks:
                logging.error("No chunks loaded.")
                raise ValueError("No chunks loaded.")
            
            ids = [hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest() for chunk in all_chunks]
            self.vector_store.add_documents(documents=all_chunks, ids=ids)
            logging.info(f"Successfully added {self.vector_store._collection.count()} chunks to Chroma.")
        else:
            logging.info(f"Using existing database with {self.vector_store._collection.count()} chunks. Skipping ingestion.")

        try:
            with open(queries_file, "r", encoding="utf-8") as file:
                content = file.readlines()
        except FileNotFoundError:
            logging.error(f"File not found at path: {queries_file}")
            raise

        for i, query in enumerate(content):
            query = query.strip()
            if not query:
                continue

            result = self.vector_store.similarity_search(query=query, k=3)
            context_string = "\n\n".join([doc.page_content for doc in result])

            prompt = f"""
            You are a strict question-answering assistant. 
            Answer the question using ONLY the provided context. 
            If the answer is not in the context, state "I do not know."
            
            Context:
            {context_string}
            
            Question: 
            {query}
            """
            response = self.model.invoke(prompt)
            answer = response.content
            logging.info(f"Successfully generated answer for {i + 1}/{len(content)}.")

            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
    parser.add_argument("--data", nargs='+', required=True, help="path of text file")
    parser.add_argument("--queries", required=True, help="path of queries file")
    parser.add_argument("--output", nargs='+', help="path of output file")
    args = parser.parse_args()
    pipeline = RAGPipeline()
    pipeline.run_evaluator(args.data, args.queries, args.output)

    

