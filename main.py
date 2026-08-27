import sys
import logging
import argparse
import hashlib
import time
import warnings
from dotenv import load_dotenv
import pandas
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_community.document_loaders import TextLoader
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()
warnings.simplefilter('ignore')
def silence_unraisable(args):
    pass

sys.unraisablehook = silence_unraisable


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

class RAGPipeline:
    def __init__(self):
        self.store = LocalFileStore("./my_embedding_cache/")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
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
        self.evaluator_llm = LangchainLLMWrapper(self.model)
        self.evaluator_embedding = LangchainEmbeddingsWrapper(self.embeddings)

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
        if self.vector_store._collection.count() == 0:
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
                queries = file.readlines()
        except FileNotFoundError:
            logging.error(f"File not found at path: {queries_file}")
            raise
        dataset = []
        for i, query in enumerate(queries):
            query = query.strip()
            if not query:
                continue
            result = []
            for attempt in range(3):
                try:
                    result = self.vector_store.similarity_search(query=query, k=3)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(2)
            context_string = "\n\n".join([doc.page_content for doc in result])

            prompt = f"""
            You are a strict question-answering assistant. 
            Answer the question using ONLY the provided context. 
            If the answer is not in the context, state "I do not know.
            Do not start your answer with phrases like 'based on the provided context' or 'from the context'. Answer directly.
            Answer in complete sentences, not just a number or phrase."
            
            Context:
            {context_string}
            
            Question: 
            {query}
            """
            response = self.model.invoke(prompt)
            if isinstance(response.content, list):
                answer = " ".join(block["text"] if isinstance(block, dict) and "text" in block else str(block) for block in response.content)
            else:
                answer = str(response.content)

            dataset.append({
                "user_input": query,
                "retrieved_contexts": [docs.page_content for docs in result],
                "response": answer
            })
            logging.info(f"Successfully generated answer for {i + 1}/{len(queries)}.")
        evaluation_dataset = EvaluationDataset.from_list(data=dataset)
        results = evaluate(
            dataset=evaluation_dataset,
            metrics=[Faithfulness(), AnswerRelevancy(strictness=1)],
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embedding
        )

        try:
            with open(outfile, "a+") as file:
                file.write("## RAG Evaluation Results\n\n")
                df = results.to_pandas() #type: ignore
                for index, row in df.iterrows():
                    file.write(f"### Query {index}: {row['user_input']}\n\n")
                    file.write(f"** Answer:** {row['response']}\n")
                    file.write(f"- **Relevance:** {row['answer_relevancy']:.2f} | **Faithfulness:** {row['faithfulness']:.2f}\n\n")
                    file.write("---\n\n")

                file.write("## Summary\n\n")
                file.write(f"- **Total Questions:** {len(queries)}\n\n")
                if 'answer_relevancy' in df.columns:
                    file.write(f"- **Average Relevance Score:** {df['answer_relevancy'].mean():.4f}\n")
                if 'faithfulness' in df.columns:
                    file.write(f"- **Faithfulness:** {df['faithfulness'].mean() * 100:.0f}%\n")


                    

        except FileNotFoundError:
            logging.error(f"File not found at path: {queries_file}")
            raise
          
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
    parser.add_argument("--data", nargs='+', required=True, help="path of text file")
    parser.add_argument("--queries", required=True, help="path of queries file")
    parser.add_argument("--output", help="path of output file")
    
    args = parser.parse_args()
    pipeline = RAGPipeline()
    pipeline.run_evaluator(args.data, args.queries, args.output)

    

