#!/usr/bin/env python3
"""
PDF Semantic Filename Extractor

This script extracts text from a PDF file and uses a pre-trained Hugging Face model
to generate a semantically meaningful filename based on the content.

Usage:
    python pdf_semantic_filename.py input.pdf
"""

import sys
import os
import re
import argparse
from pathlib import Path

import PyPDF2
from transformers import pipeline


def extract_text_from_pdf(pdf_path, max_pages=5):
    """
    Extract text from the first few pages of a PDF file.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: 5)

    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = min(len(reader.pages), max_pages)

            text = ""
            for i in range(num_pages):
                text += reader.pages[i].extract_text() + "\n"

            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def clean_text(text):
    """
    Clean and prepare text for summarization.

    Args:
        text: Raw text extracted from PDF

    Returns:
        str: Cleaned text
    """
    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove any non-alphanumeric characters except spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Truncate to first 1000 characters (model input limit)
    return text[:1000]


def generate_filename(text, model_name="facebook/bart-large-cnn"):
    """
    Generate a semantic filename using a pre-trained model.

    Args:
        text: Text to summarize
        model_name: Hugging Face model to use

    Returns:
        str: Generated filename
    """
    try:
        # Initialize summarization pipeline with a lightweight model
        summarizer = pipeline("summarization", model=model_name)

        # Generate a short summary
        summary = summarizer(text, max_length=30, min_length=5, do_sample=False)[0][
            "summary_text"
        ]

        # Clean the summary to make it suitable for a filename
        filename = re.sub(r"[^\w\s-]", "", summary).strip().lower()
        filename = re.sub(r"[-\s]+", "-", filename)

        return filename
    except Exception as e:
        print(f"Error generating filename: {e}")
        return "unnamed-document"


def main():
    """Main function to process command line arguments and rename PDF."""
    parser = argparse.ArgumentParser(
        description="Generate a semantic filename for a PDF file"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--model",
        default="facebook/bart-large-cnn",
        help="Hugging Face model to use (default: facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of pages to process (default: 5)",
    )
    parser.add_argument(
        "--output-dir", help="Directory for the renamed file (default: same as input)"
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.isfile(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' not found")
        return 1

    # Extract text from PDF
    print(f"Extracting text from '{args.pdf_path}'...")
    text = extract_text_from_pdf(args.pdf_path, args.max_pages)

    if not text:
        print("Could not extract text from the PDF")
        return 1

    # Clean text
    clean = clean_text(text)

    # Generate filename
    print(f"Generating filename using model: {args.model}")
    new_filename = generate_filename(clean, args.model)

    if new_filename == "unnamed-document":
        print("Could not generate a meaningful filename")
        return 1

    # Add .pdf extension
    new_filename += ".pdf"

    # Determine output path
    input_path = Path(args.pdf_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    else:
        output_dir = input_path.parent

    output_path = output_dir / new_filename

    # Copy file with new name
    print(f"Original filename: {input_path.name}")
    print(f"Suggested filename: {new_filename}")
    print(f"Full output path: {output_path}")

    user_input = input("Rename file? (y/n): ")
    if user_input.lower() == "y":
        try:
            # Don't overwrite existing files
            if output_path.exists():
                i = 1
                while output_path.exists():
                    stem = new_filename.rsplit(".", 1)[0]
                    output_path = output_dir / f"{stem}-{i}.pdf"
                    i += 1

            # Either copy or rename based on whether output dir is different
            if input_path.parent == output_dir:
                os.rename(input_path, output_path)
                print(f"File renamed to: {output_path.name}")
            else:
                import shutil

                shutil.copy2(input_path, output_path)
                print(f"File copied to: {output_path}")
        except Exception as e:
            print(f"Error renaming file: {e}")
            return 1
    else:
        print("Operation cancelled")

    return 0


if __name__ == "__main__":
    sys.exit(main())
