#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "PyMuPDF",
# ]
# ///

import argparse
import fitz  # PyMuPDF for PDF handling
import os


def perform_ocr_on_pdf(pdf_path, output_path, language="eng"):
    """
    Perform OCR on a PDF file using PyMuPDF's built-in OCR capabilities.

    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to save the extracted text
        language (str): Language for OCR (default: eng)
    """
    print(f"Processing PDF: {pdf_path}")

    # Check if OCR is available in PyMuPDF
    if (
        not fitz.TEXT_PRESERVE_LIGATURES
    ):  # This constant exists only if OCR is available
        raise ImportError(
            "PyMuPDF OCR capabilities not available. Make sure you have installed PyMuPDF with OCR support."
        )

    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Create a file to write the extracted text
    with open(output_path, "w", encoding="utf-8") as f:
        # Process each page
        for page_num in range(len(doc)):
            # Get the page
            page = doc.load_page(page_num)

            # Check page rotation metadata for logging purposes
            ## rotation = page.rotation rotation in [90, 270]
            print(page.rect, page.rotation)
            continue
            ##is_landscape = (page.rect.width > page.rect.height) or page.rotation rotation in [90, 270]
            orientation = "Landscape" if is_landscape else "Portrait"
            print(f"Processing page {page_num + 1}/{len(doc)} (Orientation: {orientation})")

            # Perform OCR on the page using PyMuPDF's built-in Tesseract wrapper.
            # PyMuPDF and Tesseract are expected to automatically handle page rotation
            # based on PDF metadata (page.rotation) and Tesseract's Orientation
            # and Script Detection (OSD). There isn't a direct parameter in
            # get_textpage_ocr or get_text to force a specific orientation if
            # automatic detection fails with this method.
            textpage = page.get_textpage_ocr(language=language)
            text = page.get_text(
                "text",
                flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES,
                textpage=textpage,
            )

            # Write the extracted text to file
            f.write(f"=== Page {page_num + 1} ===\n")
            f.write(text)
            f.write("\n\n")

    # Close the document
    doc.close()

    print(f"OCR completed. Text saved to: {output_path}")


def main():
    """
    Main function to parse arguments and call the OCR function.
    """
    parser = argparse.ArgumentParser(
        description="Perform OCR on a PDF file using PyMuPDF's built-in capabilities"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        default="output.txt",
        help="Path to save the extracted text (default: output.txt)",
    )
    parser.add_argument(
        "--language",
        default="eng",
        help="Language for OCR (default: eng). Use 3-letter Tesseract language codes.",
    )

    args = parser.parse_args()

    perform_ocr_on_pdf(args.pdf_path, args.output, args.language)


if __name__ == "__main__":
    main()
