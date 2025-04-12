#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers",
#     "PyMuPDF",
#     "Pillow",
# ]
# ///

import argparse
import fitz  # PyMuPDF for PDF handling
import torch
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import os


def extract_images_from_pdf(pdf_path):
    """
    Extract images from each page of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        list: List of PIL Images extracted from the PDF
    """
    # Open the PDF file
    doc = fitz.open(pdf_path)
    images = []

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Get a pixmap (image) of the page
        pix = page.get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72)
        )  # 300 DPI rendering

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

        print(f"Extracted image from page {page_num + 1}")

    return images


def perform_ocr(images, model_name="microsoft/trocr-base-printed"):
    """
    Perform OCR on a list of images using a Hugging Face TrOCR model.

    Args:
        images (list): List of PIL Images to process
        model_name (str): The Hugging Face model to use for OCR

    Returns:
        list: Extracted text for each image
    """
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and processor
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

    results = []

    # Process each image
    for i, image in enumerate(images):
        print(f"Processing image {i + 1}/{len(images)}")

        # Preprocess the image
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # Generate text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        results.append(generated_text)

    return results


def main(pdf_path, output_path, model_name):
    """
    Main function to perform OCR on a PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to save the extracted text
        model_name (str): The Hugging Face model to use for OCR
    """
    print(f"Processing PDF: {pdf_path}")

    # Extract images from PDF
    images = extract_images_from_pdf(pdf_path)

    # Perform OCR on the images
    texts = perform_ocr(images, model_name)

    # Write the extracted text to a file
    with open(output_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(f"=== Page {i + 1} ===\n")
            f.write(text)
            f.write("\n\n")

    print(f"OCR completed. Text saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform OCR on a PDF file using a Hugging Face model"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        default="output.txt",
        help="Path to save the extracted text (default: output.txt)",
    )
    parser.add_argument(
        "--model",
        default="microsoft/trocr-base-printed",
        help="Hugging Face model to use (default: microsoft/trocr-base-printed)",
    )

    args = parser.parse_args()

    main(args.pdf_path, args.output, args.model)
