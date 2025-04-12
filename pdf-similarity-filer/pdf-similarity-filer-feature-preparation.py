#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pdfminer.six",
#     "torch",
#     "transformers",
#     "joblib",
# ]
# ///

"""
PDF Date Processor - Processes PDF files that have dates in their filenames.

This script can either:
1. Walk through a directory structure to find PDF files with YYYY-MM-DD date patterns
2. Process a single PDF file directly if it has a date pattern in its filename

The script extracts text from PDFs, and outputs directory path, date, and text content.
Processing is done in parallel using all available CPU cores when processing directories.

Usage:
    python pdf_processor.py <directory_or_file_path> [-o <output_file.json>]
"""

from os import path
from typing import List, Dict, Optional
import argparse
import contextlib
import gc
import json
import logging
import os
import re
import sys
import tempfile

import joblib
from joblib import Memory

from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch


MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cpu"

# Caching Setup
BASE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "pdf_similarity_filer")
text_memory = Memory(os.path.join(BASE_CACHE_DIR, "text_cache"), verbose=0)
embedding_memory = Memory(os.path.join(BASE_CACHE_DIR, "embedding_cache"), verbose=0)


class SentenceBERTEmbedder:
    """
    A class to generate document embeddings using Sentence-BERT models.

    This implementation can handle documents of any length by:
    1. Splitting them into manageable chunks
    2. Embedding each chunk
    3. Combining the embeddings through mean pooling
    """

    @staticmethod
    def init_logger():
        # This error may occur in {569198bd574a}
        logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        max_seq_length: int = 384,
        batch_size: int = 8,
    ):
        """
        Initialize the embedder with a specific Sentence-BERT model.

        Args:
            model_name: The name of the Sentence-BERT model to use
            device: Device to run the model on ('cuda', 'cpu', etc.)
            max_seq_length: Maximum sequence length for each chunk
            batch_size: Number of chunks to process at once
        """
        # Determine device (use GPU if available)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.

        Args:
            model_output: Model's last hidden state
            attention_mask: Attention mask to exclude padding tokens

        Returns:
            Mean pooled embeddings
        """
        # Expand attention mask to same dimensions as model output
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        )

        # Sum embeddings and mask values
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Divide summed embeddings by summed mask for mean
        return sum_embeddings / sum_mask

    def _get_chunk_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of text chunks.

        Args:
            chunks: List of text chunks to embed

        Returns:
            Array of embeddings for each chunk
        """
        # Tokenize the chunks
        encoded_input = self.tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        ##print("XXXX", {k: v.shape for (k, v) in encoded_input.items()})

        # Get model output
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Apply mean pooling
        embeddings = self._mean_pooling(
            model_output.last_hidden_state, encoded_input["attention_mask"]
        )

        # Convert to numpy array and return
        return embeddings.cpu().numpy()

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split a long document into smaller chunks that fit in the model's context.

        Args:
            text: The document text to split

        Returns:
            List of text chunks
        """
        # Simple splitting by sentences (this can be improved)
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Estimate token length (rough approximation). {569198bd574a}
            sentence_tokens = len(self.tokenizer.tokenize(sentence))
            ## if sentence_tokens > 512:
            ##     logging.warning(f"Sentence too lonag: {sentence}")

            # If adding this sentence would exceed max length, finalize current chunk
            if current_length + sentence_tokens > self.max_seq_length and current_chunk:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_tokens

        # Add final chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def embed_document(self, document: str) -> np.ndarray:
        """
        Generate an embedding for a full document, handling long documents by chunking.

        Args:
            document: Document text to embed

        Returns:
            Document embedding as numpy array
        """
        # Split document into chunks
        chunks = self._split_text_into_chunks(document)

        # If document is empty or only produces empty chunks
        if not chunks:
            # Return zeros with the expected embedding dimension
            with torch.no_grad():
                # Get embedding dimension from a dummy input
                dummy_output = self.model(
                    **self.tokenizer("dummy", return_tensors="pt").to(self.device)
                )
                embedding_dim = dummy_output.last_hidden_state.size(-1)
            return np.zeros(embedding_dim)

        # Process chunks in batches
        all_embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_embeddings = self._get_chunk_embeddings(batch_chunks)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch results
        chunk_embeddings = np.vstack(all_embeddings)

        # Average embeddings from all chunks to get document embedding
        document_embedding = np.mean(chunk_embeddings, axis=0)

        # Normalize embedding to unit length (optional but recommended)
        document_embedding = document_embedding / np.linalg.norm(document_embedding)

        return document_embedding

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple documents.

        Args:
            documents: List of document texts to embed

        Returns:
            Array of document embeddings
        """
        return np.vstack([self.embed_document(doc) for doc in documents])

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )


def find_pdf_files_with_date(root_dir):
    """
    Recursively finds PDF files with a date pattern (YYYY-MM-DD) in their filenames.

    Args:
        root_dir (str): Root directory to start the search

    Returns:
        list: List of tuples (file_path, date_string)
    """
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
    pdf_files = []

    # Walk through directory structure
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                # Check if filename contains a date
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    file_path = path.join(dirpath, filename)
                    pdf_files.append((file_path, date_str))

    return pdf_files


@text_memory.cache
def extract_text_cached(file_path: str, date_str: str, root_dir: str) -> Dict:
    """
    Extracts text from a PDF file, using joblib Memory for caching.
    Note: Caching relies on the input arguments. If the file content changes
    but path/date/root_dir remain the same, the cache won't automatically invalidate.
    Joblib Memory typically uses argument hashing.

    Args:
        file_path: Absolute path to the PDF file.
        date_str: Date string extracted from the filename.
        root_dir: The root directory of the processing job (used for relative path).

    Returns:
        dict: Dictionary with file information and extracted text/error.
    """
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    logging.debug(f"Extracting text from: {file_path}")  # Add debug log

    directory = path.dirname(file_path)
    # Calculate relative directory path
    relative_dir = (
        directory[len(root_dir) :].lstrip("/")
        if directory.startswith(root_dir)
        else directory
    )

    result = {
        "file_path": file_path,
        "directory": relative_dir,
        "date": date_str,
        "text": "",
        "error": None,
    }

    try:
        # Suppress stdout/stderr during PDF extraction (consider removing if debugging issues)
        with open(os.devnull, "w") as devnull:
            with (
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                # Use the imported pdfminer function here
                extracted_text_content = extract_text(file_path)

        cleaned_text = extracted_text_content.strip().replace("\f", "")
        if not cleaned_text:
            result["error"] = "Empty document"
        else:
            result["text"] = extracted_text_content

    except Exception as exc:
        result["error"] = f"Error extracting text: {str(exc)}"
        logging.warning(f"PDF Syntax Error for {file_path}: {exc}")
    except Exception as exc:  # Catch other potential errors during extraction
        result["error"] = f"Unexpected error during text extraction: {str(exc)}"
        logging.error(
            f"Unexpected extraction error for {file_path}: {exc}", exc_info=True
        )

    return result


def is_valid_embedding(embedding, *args, **kwargs):
    """Your custom validation function."""
    return bool(embedding)


def _cached_function_with_validation(memory, func, validate_func):
    """Decorator to add custom validation to a cached function."""
    cached_func = memory.cache(func)

    def wrapper(*args, **kwargs):
        cached_value = cached_func(*args, **kwargs)
        if validate_func(cached_value, *args, **kwargs):
            return cached_value
        else:
            logging.warning(
                f"Cached value is invalid. Re-executing function for {args} {kwargs}"
            )
            # Force re-execution and update the cache
            return func(*args, **kwargs)

    return wrapper


def validated_cache(memory, validate_func):
    """Decorator to add custom validation to a cached function."""

    def decorator(func):
        return _cached_function_with_validation(
            memory, func=func, validate_func=validate_func
        )

    return decorator


@validated_cache(embedding_memory, validate_func=is_valid_embedding)
def generate_embedding_cached(
    text: str,
    error: Optional[str],
    file_path_for_logging: str,  # Added for logging context
    model_id: str = MODEL_ID,
    device: str = DEVICE,
) -> List[float]:
    """
    Generates embedding for the given text, using joblib Memory for caching.

    Args:
        text: The extracted text content.
        error: Error message from text extraction stage, if any.
        file_path_for_logging: Original file path for logging purposes only.
        model_id: The model ID to use for embedding.
        device: The device to use for embedding.

    Returns:
        list: The generated embedding as a list of floats, or empty list if error/no text.
    """
    logging.getLogger().setLevel(logging.ERROR)
    SentenceBERTEmbedder.init_logger()

    # Don't attempt to embed an empty document.
    if error == "Empty document":
        logging.info(f"Not embedding document: {file_path_for_logging}")
        return None

    embedding_list = []

    if text and not error:
        try:
            # Create embedder instance. Consider initializing it once globally or per process
            # if performance becomes an issue due to repeated initializations.
            embedder = SentenceBERTEmbedder(
                model_name=model_id,
                device=device,
                max_seq_length=384,
                batch_size=8,  # Batch size here applies to internal chunking
            )
            embedding = embedder.embed_document(text)
            embedding_list = embedding.tolist() if embedding is not None else []
        except Exception as exc:
            logging.error(
                f"Error generating embedding for {file_path_for_logging}: {exc}",
                exc_info=True,
            )
            # Leave embedding_list empty to indicate error

    return embedding_list


def process_files_in_batches(pdf_files, root_dir, batch_size=10, n_jobs=-1):
    """
    Process files by running text extraction and embedding generation as separate parallel stages.

    Args:
        pdf_files: List of (file_path, date_str) tuples
        root_dir: Root directory of the processing job
        batch_size: Number of files to process in each batch
        n_jobs: Number of parallel jobs

    Returns:
        List of processing results (dictionaries with file info, text, error, embedding)
    """
    total_files = len(pdf_files)
    logging.info(f"Starting parallel processing for {total_files} files.")

    # Stage 1: Text Extraction (Parallel)
    logging.info(f"Stage 1: Extracting text using {n_jobs} workers...")
    with joblib.parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
        text_results = joblib.Parallel(verbose=10)(  # Increased verbosity
            joblib.delayed(extract_text_cached)(file_path, date_str, root_dir)
            for file_path, date_str in pdf_files
        )
    logging.info("Stage 1: Text extraction complete.")

    # Force garbage collection
    gc.collect()

    # Stage 2: Embedding Generation (Parallel)
    logging.info(f"Stage 2: Generating embeddings using {n_jobs} workers...")
    with joblib.parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
        embedding_results = joblib.Parallel(verbose=10)(  # Increased verbosity
            joblib.delayed(generate_embedding_cached)(
                text=data.get("text", ""),
                error=data.get("error"),
                file_path_for_logging=data.get(
                    "file_path", "unknown"
                ),  # Use file_path from text_results
                # model_id and device use defaults unless passed explicitly
            )
            for data in text_results  # Iterate through results of stage 1
        )
    logging.info("Stage 2: Embedding generation complete.")

    # Force garbage collection
    gc.collect()

    # Stage 3: Combine Results
    logging.info("Stage 3: Combining results...")
    final_results = []
    if len(text_results) != len(embedding_results):
        raise ValueError(
            "Mismatch between text extraction and embedding results count!"
        )
    else:
        for i, text_data in enumerate(text_results):
            text_data["embedding"] = embedding_results[i]
            final_results.append(text_data)

    logging.info("Stage 3: Combining results complete.")

    return final_results


def main():
    """Main function to orchestrate the PDF processing workflow."""

    # Configure logging to output to stderr
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    parser = argparse.ArgumentParser(
        description="Processes PDF files with dates in filenames, extracts text, and outputs results as JSON."
    )
    parser.add_argument(
        "path", help="Path to a PDF file or directory containing PDF files."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path to the output JSON file. If not specified, output goes to stdout.",
        default=None,
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel. Default is %(default)s (all available cores).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of files to process in each batch. Default is %(default)s.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache of previously processed files before starting.",
    )
    args = parser.parse_args()

    # Clear cache if requested
    if args.clear_cache:
        logging.info(f"Clearing joblib cache directories in {BASE_CACHE_DIR}...")
        try:
            text_memory.clear(warn=True)
            embedding_memory.clear(warn=True)
            logging.info("Joblib cache cleared.")
        except Exception as e:
            logging.error(f"Failed to clear joblib cache: {e}")

    input_path = path.abspath(args.path)

    # Check if the input is a file or directory
    if os.path.isfile(input_path):
        # Process a single file
        if not input_path.lower().endswith(".pdf"):
            logging.error(f"Error: '{input_path}' is not a PDF file")
            sys.exit(1)

        # Check if filename contains a date
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
        filename = os.path.basename(input_path)
        match = date_pattern.search(filename)

        if not match:
            logging.error(
                f"Error: '{filename}' does not contain a date pattern (YYYY-MM-DD)"
            )
            sys.exit(1)

        date_str = match.group(1)
        pdf_files = [(input_path, date_str)]
        root_dir = os.path.dirname(input_path)
        logging.info(f"Processing single PDF file: '{input_path}'")

    elif os.path.isdir(input_path):
        # Process a directory
        root_dir = input_path
        logging.info(f"Searching for PDF files with date patterns in '{root_dir}'...")
        pdf_files = find_pdf_files_with_date(root_dir)
        if not pdf_files:
            logging.info("No PDF files with date patterns found.")
            sys.exit(0)

    else:
        logging.error(f"Error: '{input_path}' is not a valid file or directory")
        sys.exit(1)

    logging.info(f"Found {len(pdf_files)} PDF files with date patterns.")

    # Process PDFs in parallel using joblib with batching for better resource utilization
    logging.info(
        f"Processing PDF files in parallel with {args.n_jobs} workers and batch size {args.batch_size}..."
    )

    # Process files in batches to better manage memory and CPU utilization
    results_data = process_files_in_batches(
        pdf_files, root_dir, batch_size=args.batch_size, n_jobs=args.n_jobs
    )

    # Output the results
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(results_data, f, indent=4)
            logging.info(
                f"\nSuccessfully processed {len(results_data)} PDF files. Results saved to '{args.output}'."
            )
        except IOError as e:
            logging.error(f"\nError writing to output file '{args.output}': {e}")
            sys.exit(1)
    else:
        # Output the results as a JSON string to stdout - THIS PRINT STAYS
        print(json.dumps(results_data, indent=4))
        logging.info(
            f"\nSuccessfully processed {len(results_data)} PDF files. Results printed to stdout."
        )

    # Print performance summary
    successful = sum(1 for r in results_data if not r.get("error"))
    failed = len(results_data) - successful
    logging.info("Performance summary:")
    logging.info(f"  - Total files processed: {len(results_data)}")
    logging.info(f"  - Successfully processed: {successful}")
    logging.info(f"  - Failed to process: {failed}")
    logging.info(f"Joblib cache stored in: {BASE_CACHE_DIR}")


if __name__ == "__main__":
    main()
