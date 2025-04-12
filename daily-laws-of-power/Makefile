#!/usr/bin/env make

create-topic:
	gcloud pubsub topics create daily-laws-of-power

create-trigger:
	gcloud scheduler jobs create pubsub daily-laws-of-power-job --schedule "0 6 * * *" --topic daily-laws-of-power --message-body "Send daily paragraph"

create-function:
	gcloud functions deploy select_and_send_paragraph --runtime python312 --trigger-location us-central1 --region us-central1 --trigger-topic daily-laws-of-power  --env-vars-file env.yaml --source $(PWD)  --gen2

test:
	python3 main.py
