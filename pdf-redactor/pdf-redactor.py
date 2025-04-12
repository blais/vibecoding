#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pymupdf>=1.25.5",
#     "pypdf2>=3.0.1",
#     "reportlab>=4.3.1",
# ]
# ///

"""PDF Text Redaction Tool.

This script removes specific text from a PDF, replacing it with black rectangles.
"""

import argparse
import io
import fitz  # PyMuPDF
from typing import List, Tuple

from reportlab.pdfgen import canvas
from reportlab.lib.colors import black as reportlab_black
from reportlab.pdfbase.pdfmetrics import stringWidth


def find_text_positions(
    pdf_path: str, page_number: int, target_text: str
) -> List[dict]:
    """
    Find all occurrences of target_text in the specified page of the PDF.

    Args:
        pdf_path: Path to the PDF file
        page_number: The page number to search (0-based)
        target_text: The text to search for

    Returns:
        A list of dictionaries containing position information for each match
    """
    matches = []

    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # Search for text instances - use case-sensitive search for exact matches
    text_instances = page.search_for(target_text, quads=True)

    if not text_instances:  # If no matches found, try with more flexible search
        print(f"No exact matches found for '{target_text}'. Trying fuzzy search...")
        # Try searching for parts of the text or with different whitespace
        words = target_text.split()
        if len(words) > 1:
            # Try searching for the first few words
            search_term = " ".join(words[:2])
            text_instances = page.search_for(search_term, quads=True)

    # For each match, extract the rectangle and font information
    for quads in text_instances:
        # Convert quads to rect
        rect = quads.rect

        # Get text information at this location
        text_dict = page.get_text("dict", clip=rect)

        # Default font size if we can't determine it
        font_size = 12

        # Try to extract the actual font size from the text spans
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # Use the font size from the span
                    if "size" in span:
                        font_size = span["size"]
                        break

        # Add the match with its position information
        matches.append(
            {
                "text": target_text,
                "position": [rect.x0, rect.y0],  # Top-left corner
                "font_size": font_size,
                "rect": rect,  # Store the full rectangle for more precise redaction
            }
        )

    # If we still have no matches, try a different approach - extract all text and search
    if not matches:
        print("Still no matches. Trying full text extraction approach...")
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                line_text = ""
                line_spans = []

                # Collect all spans in this line
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    line_spans.append(span)

                # Check if our target text is in this line
                if target_text in line_text:
                    # Find approximate position based on character indices
                    start_idx = line_text.find(target_text)

                    # Find which span contains our text
                    current_pos = 0
                    for span in line_spans:
                        span_text = span.get("text", "")
                        span_len = len(span_text)

                        # If this span contains the start of our target
                        if current_pos <= start_idx < current_pos + span_len:
                            font_size = span.get("size", 12)
                            # Create a rectangle based on the span's bbox
                            span_bbox = span.get("bbox")
                            if span_bbox:
                                # Adjust rectangle to approximate the target text position
                                x0, y0, x1, y1 = span_bbox
                                # Estimate width based on character position
                                char_width = (
                                    (x1 - x0) / span_len if span_len > 0 else 10
                                )
                                target_width = len(target_text) * char_width

                                # Create rectangle for the target text
                                rect = fitz.Rect(
                                    x0 + (start_idx - current_pos) * char_width,
                                    y0,
                                    x0
                                    + (start_idx - current_pos) * char_width
                                    + target_width,
                                    y1,
                                )

                                matches.append(
                                    {
                                        "text": target_text,
                                        "position": [rect.x0, rect.y0],
                                        "font_size": font_size,
                                        "rect": rect,
                                    }
                                )
                            break

                        current_pos += span_len

    print(f"Found {len(matches)} matches for '{target_text}'")
    doc.close()
    return matches


def create_redaction_overlay(
    page_size: Tuple[float, float], matches: List[dict]
) -> io.BytesIO:
    """
    Create a PDF overlay with black rectangles at positions where target_text was found.

    Args:
        page_size: The width and height of the PDF page
        matches: List of text matches with position information

    Returns:
        BytesIO object containing the redaction overlay PDF
    """
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=page_size)
    c.setFillColor(reportlab_black)

    for match in matches:
        if "rect" in match:
            # Use the precise rectangle from PyMuPDF
            rect = match["rect"]
            # Add padding
            padding = 2

            # Make sure the rectangle has positive width and height
            width = max(rect.width, 1) + (2 * padding)
            height = max(rect.height, 1) + (2 * padding)

            # Draw the rectangle, converting from PDF to ReportLab coordinates
            c.rect(
                rect.x0 - padding,
                page_size[1] - rect.y1 - padding,  # Convert Y coordinate
                width,
                height,
                fill=True,
            )
        else:
            # Fallback to the old method if rect is not available
            x, y = match["position"]
            font_size = match["font_size"]
            text = match["text"]

            # Calculate width of target text
            target_width = max(
                stringWidth(text, "Helvetica", font_size), 10
            )  # Minimum width

            # Draw the rectangle (with some padding)
            padding = 2
            c.rect(
                x - padding,
                page_size[1] - y - padding,  # Convert from PDF to ReportLab coordinates
                target_width + (2 * padding),
                font_size + (2 * padding),
                fill=True,
            )

    c.save()
    packet.seek(0)
    return packet


def redact_pdf(input_pdf: str, output_pdf: str, target_texts: List[str]) -> None:
    """
    Remove all occurrences of target_texts from the PDF.

    Args:
        input_pdf: Path to the input PDF file
        output_pdf: Path where the redacted PDF will be saved
        target_texts: List of texts to be redacted
    """
    print(f"Redacting {len(target_texts)} expressions from {input_pdf}")
    for text in target_texts:
        print(f"  - '{text}'")

    # Open the PDF with PyMuPDF for actual text removal
    doc = fitz.open(input_pdf)

    # Process each page
    total_matches = 0
    for page_number in range(len(doc)):
        print(f"Processing page {page_number + 1}/{len(doc)}...")
        page = doc[page_number]

        # Collect all matches for all target texts on this page
        redaction_list = []
        for target_text in target_texts:
            # Find positions of target text using PyMuPDF
            matches = find_text_positions(input_pdf, page_number, target_text)
            total_matches += len(matches)

            if matches:
                print(
                    f"Found {len(matches)} matches for '{target_text}' on page {page_number + 1}"
                )

                # Add each match to the redaction list
                for match in matches:
                    if "rect" in match:
                        # Add a small padding to ensure complete text removal
                        rect = match["rect"]
                        padding = 2
                        redaction_rect = fitz.Rect(
                            rect.x0 - padding,
                            rect.y0 - padding,
                            rect.x1 + padding,
                            rect.y1 + padding,
                        )
                        # Add to redaction list - use PyMuPDF color format
                        redaction_list.append([redaction_rect, (0, 0, 0), (0, 0, 0)])

        # Apply all redactions for this page
        if redaction_list:
            for redact_params in redaction_list:
                rect, fill_color, stroke_color = redact_params
                page.add_redact_annot(
                    quad=rect,
                    text="",  # Empty text for complete removal
                    fill=fill_color,  # Already a tuple (0,0,0)
                    text_color=stroke_color,  # Already a tuple (0,0,0)
                )
            page.apply_redactions()

    # Save the redacted document
    doc.save(output_pdf)
    doc.close()

    print(f"Total matches found and removed: {total_matches}")


def generate_output_filename(input_pdf):
    """
    Generate an output filename by appending '-redacted' before the extension.

    Args:
        input_pdf: Path to the input PDF file

    Returns:
        A string with the generated output filename
    """
    # Split the input filename into base and extension
    if "." in input_pdf:
        base, ext = input_pdf.rsplit(".", 1)
        return f"{base}-redacted.{ext}"
    else:
        # If no extension, just append -redacted
        return f"{input_pdf}-redacted"


def main():
    parser = argparse.ArgumentParser(description="Redact text in a PDF file")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument(
        "--output",
        "-o",
        dest="output_pdf",
        help="Path where the redacted PDF will be saved (default: input-redacted.pdf)",
    )
    parser.add_argument(
        "target_text", nargs="+", help="Text expression(s) to be redacted"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    # Generate output filename if not provided
    output_pdf = args.output_pdf
    if not output_pdf:
        output_pdf = generate_output_filename(args.input_pdf)

    try:
        print("Starting redaction process...")
        redact_pdf(args.input_pdf, output_pdf, args.target_text)
        print(f"Redaction complete! Text removed and PDF saved to {output_pdf}")
    except Exception as e:
        print(f"Error during redaction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
