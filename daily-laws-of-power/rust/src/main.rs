#![allow(unused_imports)]

use chrono::{Datelike, Local, NaiveDate};
use clap::Parser;
use google_cloud_storage::client::{Client, ClientConfig};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use lettre::message::{header::ContentType, Attachment, MultiPart, SinglePart};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{Message, SmtpTransport, Transport};
use log;
use regex::Regex;
use serde::Deserialize;
use serde_yaml::from_str;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Read};
use tokio::main;

fn read_paragraphs_from_file(filename: String) -> io::Result<String> {
    // "The-48-Laws-of-Power-by-Robert-Greene-Book-Summary.txt"
    fs::read_to_string(filename)
}

async fn read_paragraphs_from_gcs(
    bucket_name: &str,
    blob_name: &str,
) -> Result<String, Box<dyn Error>> {

    // If the credentials aren't set, we're running this locally, set the env var before creating the credentials.
    match std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        Ok(_) => (),
        Err(std::env::VarError::NotPresent) => unsafe {
            std::env::set_var("GOOGLE_APPLICATION_CREDENTIALS", "/home/blais/.google/static-website-with-ssh.json");
        },
        Err(err) => return Err(Box::new(err))
    }

    log::info!("before");
    let config = ClientConfig::default().with_auth().await?;
    // let config = default_config.with_auth().await?;
    log::info!("after1");
    let client = Client::new(config);
    log::info!("after2");
    let result = client
        .download_object(
            &GetObjectRequest {
                bucket: bucket_name.to_string(),
                object: blob_name.to_string(),
                ..Default::default()
            },
            &Range::default(),
        )
        .await?;
    log::info!("after3");
    let content = String::from_utf8(result)?;
    Ok(content)
}

fn send_email(
    sender_email: &str,
    sender_password: &str,
    recipient_email: &str,
    subject: &str,
    body: &str,
) -> Result<(), Box<dyn Error>> {
    // Create message.
    let email = Message::builder()
        .from(sender_email.parse()?)
        .to(recipient_email.parse()?)
        .subject(subject)
        .multipart(
            MultiPart::alternative().singlepart(
                SinglePart::builder()
                    .header(ContentType::TEXT_PLAIN)
                    .body(body.to_string()),
            ),
        )?;

    // Open a remote connection to gmail
    let creds = Credentials::new(sender_email.to_string(), sender_password.to_string());
    let mailer = SmtpTransport::relay("smtp.gmail.com")?
        .credentials(creds)
        .build();
    // Send the email
    mailer.send(&email)?;
    Ok(())
}

async fn select_and_send_paragraph(from_file: Option<String>) -> Result<(), Box<dyn Error>> {
    // Get configuration from environment variables
    let bucket_name = env::var("BUCKET_NAME")?;
    let blob_name = env::var("BLOB_NAME")?;
    let sender_email = env::var("SENDER_EMAIL")?;
    let sender_password = env::var("SENDER_PASSWORD")?;
    let recipient_email = env::var("RECIPIENT_EMAIL")?;

    // Read paragraphs
    let content = match from_file {
        Some(filename) => read_paragraphs_from_file(filename)?,
        None => read_paragraphs_from_gcs(&bucket_name, &blob_name).await?,
    };

    // Split on "LAW " to separate paragraphs
    let re = Regex::new(r"LAW ")?;
    let paragraphs: Vec<String> = re
        .split(&content)
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().to_string())
        .collect();

    // Select paragraph based on day of year
    let today = Local::now().date_naive();
    let year_start = NaiveDate::from_ymd_opt(today.year(), 1, 1).unwrap();
    let year_days = today.signed_duration_since(year_start).num_days() as usize;
    let selected_paragraph = &paragraphs[year_days % paragraphs.len()];

    // Prepare email
    let lines: Vec<&str> = selected_paragraph.lines().collect();
    let title = lines[0];
    let today_str = Local::now().format("%Y-%m-%d").to_string();
    let subject = format!("Daily Law of Power - {} - {}", today_str, title);

    let paragraph = lines[1..].join(" ");
    let body = format!("{}\n\n{}", title, paragraph);

    // Send email
    send_email(
        &sender_email,
        &sender_password,
        &recipient_email,
        &subject,
        &body,
    )?;

    log::info!("Email sent successfully!");
    Ok(())
}

fn set_environment_from_yaml(filename: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let yaml_content = fs::read_to_string(filename)?;
    let data: HashMap<String, String> = serde_yaml::from_str(&yaml_content)?;
    for (key, value) in &data {
        unsafe {
            env::set_var(key, value);
        }
    }
    Ok(data)
}

fn setup_logger() {
    // Setup logger.
    unsafe {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();
}

#[derive(Parser)]
#[clap(author, version, about)]
struct Args {
    #[clap(help = "Input text file with daily laws", short, long)]
    from_file: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // For local testing
    setup_logger();
    let environ = set_environment_from_yaml("env.yaml")?;
    log::info!("Environment: {:?}", environ);

    let args = Args::parse();
    select_and_send_paragraph(args.from_file).await?;
    Ok(())
}
