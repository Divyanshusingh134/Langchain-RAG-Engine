import os 
import argparse
from dotenv import load_dotenv


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("API Key not found")

parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
parser.add_argument("--data", nargs='+', required=True, help="path of text file")
parser.add_argument("--queries", required=True, help="path of queries file")
parser.add_argument("--output", nargs='+', help="path of output file")

