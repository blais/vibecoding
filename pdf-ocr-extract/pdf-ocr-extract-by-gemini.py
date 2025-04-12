#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers",
#     "PyMuPDF",
#     "Pillow",
#     "accelerate",
# ]
# ///

import argparse
import io
import logging
import os
import time
from typing import Optional, List

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


def ocr_pdf_local(
    pdf_path: str, model_id: str = "microsoft/trocr-base-printed", dpi: int = 300
) -> Optional[str]:
    """
    Performs OCR on a PDF file using a local Hugging Face model.

    Args:
        pdf_path: Path to the PDF file.
        model_id: Name of the Hugging Face model to use
                  (e.g., "microsoft/trocr-base-printed").
        dpi: Resolution (dots per inch) to render PDF pages.
             Higher values increase quality but also memory usage and processing time.

    Returns:
        str: The extracted text concatenated from all pages, or None if error.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at {pdf_path}")
        return None

    # --- 1. Setup Device (GPU or CPU) ---
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- 2. Load Model and Processor ---
    logging.info(f"Loading OCR model: {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        # Use device_map="auto" for better memory management with accelerate
        model = AutoModelForVision2Seq.from_pretrained(
            model_id
        )  # .to(device) # Removed .to(device) as device_map handles it

    except Exception as exc:
        logging.error(f"Error loading model '{model_id}': {exc}")
        logging.error(
            "Please ensure the model name is correct and you have an internet connection for the first download."
        )
        return None
    logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    # --- 3. Process PDF ---
    extracted_text = []
    logging.info(f"Processing PDF: {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logging.error(f"Error opening PDF file: {exc}")
        return None

    total_pages = len(doc)
    page_times = []

    for page_index, page in enumerate(doc):
        page_start_time = time.time()
        logging.info(f"  Processing page {page_index + 1}/{total_pages}...")

        # --- a. Render page to an image ---
        try:
            # Get a higher resolution pixmap
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")  # Get bytes in PNG format
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        except Exception as exc:
            logging.error(f"  Error rendering page {page_index + 1} to image: {exc}")
            continue  # Skip this page

        # --- b. Perform OCR on the image ---
        try:
            # Prepare image for model
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device)

            # Generate text IDs
            generated_ids = model.generate(
                pixel_values, 
                max_length=1024,
                return_dict_in_generate=True,
                output_scores=False
            ).sequences  # Adjust max_length if needed

            # Decode text IDs
            # Check if generated_ids is valid before decoding
            if generated_ids is not None and generated_ids.shape[1] > 0:
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            else:
                # Handle cases where generation failed or produced empty output
                logging.warning(f"  OCR generation produced no output for page {page_index + 1}.")
                generated_text = f"[OCR Warning: No text generated for page {page_index + 1}]"
            extracted_text.append(generated_text)

        except Exception as exc:
            logging.error(f"  Error during OCR on page {page_index + 1}: {exc}")
            # Optionally append an error marker or skip
            extracted_text.append(f"[OCR Error on Page {page_index + 1}]")

        page_time_taken = time.time() - page_start_time
        page_times.append(page_time_taken)
        logging.info(f"  Page {page_index + 1} processed in {page_time_taken:.2f} seconds.")

    doc.close()

    # --- 4. Combine and Return Text ---
    full_text = "\n\n--- Page Break ---\n\n".join(
        extracted_text
    )  # Join pages with a separator

    end_time = time.time()
    total_time = end_time - start_time
    avg_page_time = sum(page_times) / len(page_times) if page_times else 0

    logging.info("-" * 30)
    logging.info(f"PDF processing finished.")
    logging.info(f"Total time: {total_time:.2f} seconds")
    logging.info(f"Average time per page: {avg_page_time:.2f} seconds")
    logging.info("-" * 30)

    return full_text


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform OCR on a PDF file using a local Hugging Face model."
    )
    parser.add_argument("input_pdf", type=str, help="Path to the input PDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output text file. Defaults to '<input_pdf_name>_ocr_output.txt'.",
        default=None,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,  # Defaulting to 200 as in the original example usage
        help="Resolution (dots per inch) for rendering PDF pages (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to orchestrate PDF OCR processing and output saving.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_arguments()
    input_pdf_path = args.input_pdf
    output_txt_path = args.output

    extracted_content = ocr_pdf_local(input_pdf_path, dpi=args.dpi)

    if extracted_content:
        logging.info("\n--- Extracted Text ---")
        # Optionally log shorter preview if very long? For now, log all.
        # logging.info(extracted_content[:1000] + "..." if len(extracted_content) > 1000 else extracted_content)
        logging.info(extracted_content) # Log the full content

        # Determine output path
        if output_txt_path is None:
            output_txt_path = os.path.splitext(input_pdf_path)[0] + "_ocr_output.txt"

        # Save the output to a file
        try:
            with open(output_txt_path, "w", encoding="utf-8") as output_file:
                output_file.write(extracted_content)
            logging.info(f"\n--- Output saved to: {output_txt_path} ---")
        except Exception as exc:
            logging.error(f"\n--- Error saving output file: {exc} ---")
    else:
        logging.error("OCR process failed.")


if __name__ == "__main__":
    main()
