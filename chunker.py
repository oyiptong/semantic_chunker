#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse
import sqlite3
import os
from typing import List, Dict, Any
import datetime
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np # Used for more robust string matching
# import uuid # Import uuid for generating a unique invocation ID - No longer needed

# Download the NLTK punkt tokenizer data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
     print("NLTK punkt tokenizer data not found. Downloading 'punkt'...")
     nltk.download('punkt')
except Exception as e:
    # Catch any other unexpected errors during the find process for 'punkt'
    print(f"An unexpected error occurred while checking for NLTK 'punkt' data: {e}")

# Also check for 'punkt_tab' which might be needed by sent_tokenize in some cases
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
     print("NLTK punkt_tab tokenizer data not found. Downloading 'punkt_tab'...")
     # Note: Sometimes downloading 'punkt' includes 'punkt_tab', but explicitly
     # downloading 'punkt_tab' can help if it's missing.
     # If this still fails, you might need to run nltk.download() interactively
     # and select 'punkt' or 'all' to ensure all dependencies are met.
     try:
         nltk.download('punkt_tab')
     except Exception as e:
         print(f"Could not automatically download 'punkt_tab'. Please try running 'import nltk; nltk.download(\\'punkt_tab\\')' manually.")
         print(f"Error: {e}")
except Exception as e:
    # Catch any other unexpected errors during the find process for 'punkt_tab'
    print(f"An unexpected error occurred while checking for NLTK 'punkt_tab' data: {e}")


# Try importing LangChain components
try:
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # print("LangChain not found. Install with: pip install langchain langchain-experimental langchain-community sentence-transformers") # Keep silent unless needed

# Try importing LlamaIndex components
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # print("LlamaIndex not found. Install with: pip install llama-index llama-index-embeddings-huggingface sentence-transformers") # Keep silent unless needed

# Ensure at least one library is available
if not LANGCHAIN_AVAILABLE and not LLAMAINDEX_AVAILABLE:
    raise ImportError("Neither LangChain nor LlamaIndex found. Please install at least one using the commands mentioned previously.")

def get_text_with_offsets(filepath: str) -> (str, List[int]):
    """Reads a text file and returns content and start character offset for each line."""
    content = ""
    line_offsets = []
    current_offset = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_offsets.append(current_offset)
                content += line
                current_offset += len(line)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        exit(1)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        exit(1)
    return content, line_offsets

def get_line_and_char_from_offset(offset: int, line_offsets: List[int]) -> (int, int):
    """Converts a character offset to line number and character number within that line."""
    if not line_offsets:
        return 1, offset + 1 # Handle empty file case

    # Find the largest line offset less than or equal to the given offset
    line_index = 0
    # Iterate up to the second to last element to avoid index out of bounds when checking i+1
    for i in range(len(line_offsets) - 1):
        if line_offsets[i+1] > offset:
            line_index = i
            break
    # If the loop finishes without finding a larger offset, the offset is in the last line
    # This check is needed because the loop goes up to len(line_offsets) - 1
    if offset >= line_offsets[-1]:
         line_index = len(line_offsets) - 1


    line_number = line_index + 1 # Line numbers are 1-based
    char_number = offset - line_offsets[line_index] + 1 # Character numbers are 1-based
    return line_number, char_number

def chunk_text_langchain(text: str, embedding_model_name: str) -> List[Dict[str, Any]]:
    """Chunks text using LangChain's SemanticChunker."""
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not installed. Cannot use LangChain chunking.")
        return []

    print(f"Using LangChain with embedding model: {embedding_model_name}")
    # Load the embedding model
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    except Exception as e:
        print(f"Error loading embedding model {embedding_model_name}: {e}")
        print("Please ensure the model name is correct and sentence-transformers is installed.")
        return []

    # Initialize the semantic chunker
    # breakpoint_threshold_amount=95 is a common starting point (split at 95th percentile of differences)
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_amount=95)

    # Split the text
    try:
        chunks = text_splitter.create_documents([text])
        print(f"Created {len(chunks)} chunks using LangChain.")
        return [{"text": chunk.page_content} for chunk in chunks]
    except Exception as e:
        print(f"Error during LangChain chunking: {e}")
        return []


def chunk_text_llamaindex(text: str, embedding_model_name: str) -> List[Dict[str, Any]]:
    """Chunks text using LlamaIndex's SemanticSplitterNodeParser."""
    if not LLAMAINDEX_AVAILABLE:
        print("LlamaIndex is not installed. Cannot use LlamaIndex chunking.")
        return []

    print(f"Using LlamaIndex with embedding model: {embedding_model_name}")
    # Load the embedding model
    try:
        # LlamaIndex uses HuggingFaceEmbedding for Sentence Transformers models
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    except Exception as e:
        print(f"Error loading embedding model {embedding_model_name}: {e}")
        print("Please ensure the model name is correct and sentence-transformers is installed.")
        return []

    # Initialize the semantic splitter node parser
    # breakpoint_percentile_threshold=95 is a common starting point
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, # Compare sentence by sentence
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )

    # LlamaIndex works with Document objects
    document = Document(text=text)

    # Get nodes (chunks) from the document
    try:
        nodes = splitter.get_nodes_from_documents([document])
        print(f"Created {len(nodes)} chunks using LlamaIndex.")
        return [{"text": node.get_content()} for node in nodes]
    except Exception as e:
        print(f"Error during LlamaIndex chunking: {e}")
        return []

def store_chunks_in_db(db_path: str, chunks: List[Dict[str, Any]], original_text: str, line_offsets: List[int], source_file: str, library_used: str):
    """Stores chunk data in a SQLite database with additional metadata."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create runs table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invocation_datetime TEXT,
                source_file TEXT,
                library_used TEXT
            )
        ''')

        # Create chunks table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_text TEXT,
                start_line INTEGER,
                start_char INTEGER,
                end_line INTEGER,
                end_char INTEGER,
                char_count INTEGER,
                sentence_count INTEGER,
                invocation_id INTEGER, -- Foreign key to the runs table
                FOREIGN KEY (invocation_id) REFERENCES runs(id)
            )
        ''')

        # Insert into the runs table and get the invocation_id
        invocation_dt = datetime.datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO runs (invocation_datetime, source_file, library_used)
            VALUES (?, ?, ?)
        ''', (invocation_dt, source_file, library_used))
        invocation_id = cursor.lastrowid # Get the auto-incremented ID

        # Use a pointer to track the current position in the original text
        current_original_offset = 0

        # Insert chunks
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['text']

            # Calculate character count
            char_count = len(chunk_text)

            # Calculate sentence count using NLTK
            try:
                sentences = sent_tokenize(chunk_text)
                sentence_count = len(sentences)
            except Exception as e:
                print(f"Warning: Could not tokenize sentences for chunk {i+1}: {e}. Setting sentence_count to 0.")
                sentence_count = 0

            # --- More Robust Chunk Matching ---
            # Try to find the chunk text starting from the current_original_offset
            # This assumes chunks are returned in order and is more reliable than global find.
            # We also strip leading/trailing whitespace for a more flexible match.
            search_text = original_text[current_original_offset:]
            chunk_text_stripped = chunk_text.strip()

            # Find the start index of the stripped chunk text in the remaining original text
            relative_start_index = search_text.find(chunk_text_stripped)

            if relative_start_index != -1:
                # Calculate the absolute start index in the original text
                start_index = current_original_offset + relative_start_index
                # Calculate the end index based on the original chunk text length
                end_index = start_index + len(chunk_text) - 1

                # Update the current_original_offset for the next chunk search
                current_original_offset = end_index + 1 # Start searching after the current chunk

                start_line, start_char = get_line_and_char_from_offset(start_index, line_offsets)
                end_line, end_char = get_line_and_char_from_offset(end_index, line_offsets)

                cursor.execute('''
                    INSERT INTO chunks (chunk_text, start_line, start_char, end_line, end_char, char_count, sentence_count, invocation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (chunk_text, start_line, start_char, end_line, end_char, char_count, sentence_count, invocation_id))
            else:
                # If sequential find fails, fall back to a global find as a last resort,
                # but print a warning as this is less reliable for offsets.
                print(f"Warning: Sequential find failed for chunk {i+1}. Falling back to global find.")
                start_index = original_text.find(chunk_text)
                if start_index != -1:
                     end_index = start_index + len(chunk_text) - 1
                     start_line, start_char = get_line_and_char_from_offset(start_index, line_offsets)
                     end_line, end_char = get_line_and_char_from_offset(end_index, line_offsets)
                     cursor.execute('''
                        INSERT INTO chunks (chunk_text, start_line, start_char, end_line, end_char, char_count, sentence_count, invocation_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (chunk_text, start_line, start_char, end_line, end_char, char_count, sentence_count, invocation_id))
                     # Do NOT update current_original_offset here, as global find doesn't guarantee order
                     print(f"Warning: Stored chunk {i+1} using global find, offsets might be less accurate: {chunk_text[:100]}...")
                else:
                    print(f"Error: Could not find chunk text in original document using sequential or global find. Skipping chunk {i+1}: {chunk_text[:100]}...") # Print first 100 chars

        conn.commit()
        print(f"Successfully stored {len(chunks)} chunks in {db_path}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform semantic chunking and store results in SQLite.")
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument("library", choices=["langchain", "llamaindex"], help="Library to use for chunking (langchain or llamaindex).")
    parser.add_argument("--db_path", default="chunks.db", help="Path to the SQLite database file.")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Name of the Sentence-Transformers embedding model to use.")

    args = parser.parse_args()

    # Check if the selected library is available
    if args.library == "langchain" and not LANGCHAIN_AVAILABLE:
        print("LangChain is not installed. Please install it to use this option.")
        exit(1)
    if args.library == "llamaindex" and not LLAMAINDEX_AVAILABLE:
        print("LlamaIndex is not installed. Please install it to use this option.")
        exit(1)

    # Read the input file and get line offsets
    original_text, line_offsets = get_text_with_offsets(args.input_file)

    # Perform chunking
    if args.library == "langchain":
        chunks_data = chunk_text_langchain(original_text, args.embedding_model)
    elif args.library == "llamaindex":
        chunks_data = chunk_text_llamaindex(original_text, args.embedding_model)
    else:
        chunks_data = [] # Should not happen due to argparse choices

    if not chunks_data:
        print("No chunks were generated. Exiting.")
        exit(1)

    # Store chunks in the database
    store_chunks_in_db(args.db_path, chunks_data, original_text, line_offsets, args.input_file, args.library)
