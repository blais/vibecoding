"""Daily emailer of a random law from 48 Laws of Power."""

from datetime import date
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.cloud import storage
import os
import re
import smtplib
import yaml


def read_paragraphs_from_file() -> str:
    """Read paragraphs from local file--for testing."""
    with open("The-48-Laws-of-Power-by-Robert-Greene-Book-Summary.txt") as txtfile:
        return txtfile.read()


def read_paragraphs_from_gcs(bucket_name, blob_name) -> str:
    """
    Read paragraphs from a file stored in Google Cloud Storage.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def send_email(sender_email, sender_password, recipient_email, subject, body):
    """
    Send an email using SMTP.
    """
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(message)


def select_and_send_paragraph(event, context, *, from_file: bool = False):
    """
    Cloud Function to select and send a paragraph.
    """
    # Get configuration from environment variables
    bucket_name = os.environ.get("BUCKET_NAME")
    blob_name = os.environ.get("BLOB_NAME")
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")

    try:
        # Read paragraphs
        if from_file:
            content = read_paragraphs_from_file()
        else:
            content = read_paragraphs_from_gcs(bucket_name, blob_name)

        # Split on double newlines to separate paragraphs
        paragraphs = [p.strip() for p in re.split(r"LAW ", content) if p.strip()]

        # Select paragraph
        today = date.today()
        year_days = (today - today.replace(month=1, day=1)).days
        selected_paragraph = paragraphs[year_days % len(paragraphs)]

        # Prepare email
        title = selected_paragraph.splitlines()[0]
        today = datetime.now().strftime("%Y-%m-%d")
        subject = f"Daily Law of Power - {today} - {title}"
        lines = selected_paragraph.splitlines()[1:]
        paragraph = " ".join(lines)
        body = f"{title}\n\n{paragraph}"

        # Send email
        send_email(sender_email, sender_password, recipient_email, subject, body)

        return "Email sent successfully!"

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e


def setup_yaml_environment(filename: str) -> dict[str, str]:
    with open(filename, "r") as yamlfile:
        data = yaml.safe_load(yamlfile)
    os.environ.update(data)


if __name__ == "__main__":
    # For local testing
    setup_yaml_environment("env.yaml")
    select_and_send_paragraph(None, None, from_file=True)
