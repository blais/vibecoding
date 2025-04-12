#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pytesseract",
#     "Pillow",
#     "pdf2image",
#     "poppler-utils",
# ]
# ///

import argparse
import os
import io
import logging
from typing import Dict, Any

import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def detect_orientation(img: Image.Image) -> Dict[str, Any]:
    """
    Detect the orientation of text in a PIL Image using Tesseract.

    Args:
        img: PIL Image object

    Returns:
        Dictionary containing orientation, rotation angle, and OSD text.
        orientation is 0, 90, 180, or 270 degrees.
    """
    # Use Tesseract to get orientation information directly from the PIL Image
    osd = pytesseract.image_to_osd(img)

    # Parse the orientation info
    rotation = 0
    orientation = 0

    for line in osd.split("\n"):
        parts = line.split(":")
        if len(parts) == 2:
            field_name = parts[0].strip()
            field_value = parts[1].strip()
            if field_name == "Rotate":
                try:
                    rotation = int(field_value)
                except ValueError:
                    print(f"Warning: Could not parse rotation value: {field_value}")
            elif field_name == "Orientation":
                try:
                    orientation = int(field_value)
                except ValueError:
                    print(f"Warning: Could not parse orientation value: {field_value}")

    return {"orientation": orientation, "rotation": rotation, "osd_text": osd}
def perform_ocr_on_pdf(
    pdf_path: str,
    output_path: str,
    language: str = "eng",
    detect_text_orientation: bool = False,
    dpi: int = 300,
) -> None:
    """
    Perform OCR on a PDF file using Pytesseract and pdf2image.

    Args:
        pdf_path: Path to the PDF file.
        output_path: Path to save the extracted text.
        language: Language for OCR (default: eng).
        detect_text_orientation: Whether to detect text orientation (default: False).
        dpi: Dots per inch for rendering PDF pages to images (default: 300).
    """
    logging.info(f"Processing PDF: {pdf_path}")

    try:
        # Convert PDF pages to PIL Images
        # Note: pdf2image requires poppler-utils to be installed on the system
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        logging.error(
            f"Error converting PDF to images: {e}; "
            "Make sure poppler-utils is installed and in your PATH."
        )
        return

    # Create a file to write the extracted text
    with open(output_path, "w", encoding="utf-8") as f:
        # Process each image (page)
        for i, img in enumerate(images):
            page_num = i + 1
            print(f"Processing page {page_num}/{len(images)}")

            f.write(f"=== Page {page_num} ===\n")  # Write header

            img_to_ocr = img
            rotation_angle = 0

            # Detect orientation if requested
            if detect_text_orientation:
                try:
                    orientation_info = detect_orientation(img)
                    # Use the 'rotation' value from OSD, which indicates how to make the text upright
                    # Note: PIL rotation is counter-clockwise, OSD might be clockwise.
                    # Tesseract's 'Rotate' usually means degrees to rotate *clockwise* for upright.
                    # PIL's rotate is counter-clockwise. So, use negative? Let's test.
                    # Common OSD rotations are 0, 90, 180, 270.
                    # If OSD says rotate 90, we need img.rotate(-90) or img.rotate(270, expand=True)
                    # If OSD says rotate 270, we need img.rotate(-270) or img.rotate(90, expand=True)
                    # Let's stick to the reported rotation and see Pytesseract handles it.
                    # UPDATE: Pytesseract's image_to_string doesn't auto-rotate based on OSD.
                    # We need to rotate the image *before* sending it to image_to_string.
                    # PIL rotates counter-clockwise. Tesseract OSD 'Rotate' is clockwise needed.
                    rotation_angle = orientation_info[
                        "rotation"
                    ]  # Clockwise rotation needed

                    # Log orientation details
                    logging.info(
                        f"Detected Orientation: {orientation_info['orientation']} degrees"
                    )
                    logging.info(
                        f"Required Clockwise Rotation (OSD): {rotation_angle} degrees"
                    )
                    logging.info(f"OSD Info:\n{orientation_info['osd_text']}")

                    # Apply the detected rotation *before* OCR if needed
                    # PIL rotates counter-clockwise, so use negative angle or (360 - angle)
                    if rotation_angle != 0:
                        print(
                            f"Applying rotation: {-rotation_angle} degrees (counter-clockwise)"
                        )
                        # Use expand=True to prevent cropping during rotation
                        img_to_ocr = img.rotate(-rotation_angle, expand=True)

                except pytesseract.TesseractError as e:
                    logging.error(
                        f"Orientation detection failed for page {page_num}: {e}"
                    )
                    # Proceed with OCR on the original image if OSD fails

            # Perform OCR on the (potentially rotated) image
            try:
                text = pytesseract.image_to_string(img_to_ocr, lang=language)
            except pytesseract.TesseractError as e:
                logging.error(f"Error during OCR on page {page_num}: {e}")
                text = f""

            # Write the extracted text to file
            f.write(text)

            # Close the image object to free memory
            if img_to_ocr != img:
                img_to_ocr.close()
            img.close()

    logging.info(f"OCR completed. Text saved to: {output_path}")


def main() -> None:
    """
    Main function to parse arguments and call the OCR function.
    """
    parser = argparse.ArgumentParser(
        description="Perform OCR on a PDF file using PyMuPDF's built-in capabilities"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        default="output.txt",
        help="Path to save the extracted text (default: output.txt)",
    )
    parser.add_argument(
        "--language",
        default="eng",
        help="Language for OCR (default: eng). Use 3-letter Tesseract language codes.",
    )
    parser.add_argument(
        "-D",
        "--no-detect-orientation",
        dest="detect_orientation",
        default=True,
        action="store_false",  # Default is False now
        help="Detect and correct text orientation in each page before OCR.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (DPI) for rendering PDF pages to images (default: 300)",
    )

    args = parser.parse_args()

    perform_ocr_on_pdf(
        args.pdf_path, args.output, args.language, args.detect_orientation, args.dpi
    )


if __name__ == "__main__":
    main()
