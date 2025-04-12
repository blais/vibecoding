#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-api-python-client>=2.166.0",
#     "google-auth>=2.38.0",
#     "google-auth-oauthlib>=1.2.1",
#     "python-dateutil>=2.9.0.post0",
#     "pytz>=2025.2",
#     "requests>=2.32.3",
# ]
# ///

"""
Google Calendar Free Time Finder

This script connects to the Google Calendar API, fetches all events for the next three weeks,
and identifies free time slots between 9am and 6pm each day.

Requirements:
- google-auth
- google-auth-oauthlib
- google-api-python-client
- python-dateutil

Setup:
1. Enable Google Calendar API and download credentials.json from Google Cloud Console
2. Run the script once to authenticate and generate token.json
3. The script will output free time intervals in the format "Mon 3/17 9am-11am, 2pm-4pm"
"""

import os
import datetime
import zoneinfo
import argparse
from os import path

from dateutil import parser
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def authenticate(scopes):
    """Authenticate with Google Calendar API and return the service object."""

    secrets_filename = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    base, ext = path.splitext(secrets_filename)
    token_filename = base + "_token" + ext

    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists(token_filename):
        creds = Credentials.from_authorized_user_info(
            eval(open(token_filename, "r").read()), scopes
        )

    # If credentials don't exist or are invalid, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # If refresh fails (e.g., token revoked), force re-authentication
                creds = None  # Ensure we trigger the interactive flow below

        # If creds are still None or invalid after attempting refresh, start interactive flow
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(secrets_filename, scopes)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_filename, "w") as token:
            token.write(str(creds.to_json()))

    return creds


def get_time_boundaries(day_date):
    """Get the 9am and 6pm boundaries for a given day."""
    # Create datetime objects for 9am and 6pm
    start_time = day_date.replace(hour=9, minute=0, second=0, microsecond=0)
    end_time = day_date.replace(hour=18, minute=0, second=0, microsecond=0)
    return start_time, end_time


def get_events(service, time_min, time_max):
    """Fetch calendar events between time_min and time_max."""
    # Call the Calendar API to get events
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    return events_result.get("items", [])


def format_time_slot(start_time, end_time):
    """Format a time slot as '9am-11am' or '9:30am-11:30am'."""
    # Format start time
    if start_time.minute == 0:
        start_str = start_time.strftime("%I%p").lower().lstrip("0")
    else:
        start_str = start_time.strftime("%I:%M%p").lower().lstrip("0")

    # Format end time
    if end_time.minute == 0:
        end_str = end_time.strftime("%I%p").lower().lstrip("0")
    else:
        end_str = end_time.strftime("%I:%M%p").lower().lstrip("0")

    return f"{start_str}-{end_str}"


def format_date(date):
    """Format a date as 'Mon 3/17'."""
    return date.strftime("%a %m/%d")


def find_free_time_slots(service, start_date, num_days=21, include_weekends=False):
    """Find free time slots between 9am-6pm for the specified number of days."""
    free_time_by_day = {}

    # Iterate through each day in the specified range
    for day_offset in range(num_days):
        current_date = start_date + datetime.timedelta(days=day_offset)

        # Skip weekends if include_weekends is False
        if not include_weekends and current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            continue

        start_boundary, end_boundary = get_time_boundaries(current_date)

        # Get all events for the current day
        day_events = get_events(service, start_boundary, end_boundary)

        # Remove all day events
        day_events = [event for event in day_events if "dateTime" in event["start"]]

        # Create half-hour slots from 9am to 6pm
        half_hour_slots = []
        current_slot_start = start_boundary

        while current_slot_start < end_boundary:
            current_slot_end = current_slot_start + datetime.timedelta(minutes=30)
            if current_slot_end > end_boundary:
                current_slot_end = end_boundary

            half_hour_slots.append((current_slot_start, current_slot_end))
            current_slot_start = current_slot_end

        # Mark slots as busy if they overlap with any event
        free_slots = half_hour_slots.copy()

        for event in day_events:
            # Parse event start and end times
            event_start = parser.parse(
                event["start"].get("dateTime", event["start"].get("date"))
            )
            event_end = parser.parse(
                event["end"].get("dateTime", event["end"].get("date"))
            )

            # Adjust event times to be within our boundaries
            event_start = max(event_start, start_boundary)
            event_end = min(event_end, end_boundary)

            # Remove slots that overlap with this event
            free_slots = [
                slot
                for slot in free_slots
                if not (slot[0] < event_end and slot[1] > event_start)
            ]

        # Merge consecutive free slots
        merged_free_slots = []
        if free_slots:
            current_start = free_slots[0][0]
            current_end = free_slots[0][1]

            for i in range(1, len(free_slots)):
                # If this slot starts exactly when the previous ends, merge them
                if free_slots[i][0] == current_end:
                    current_end = free_slots[i][1]
                else:
                    # Otherwise, add the current merged slot and start a new one
                    merged_free_slots.append((current_start, current_end))
                    current_start = free_slots[i][0]
                    current_end = free_slots[i][1]

            # Add the last merged slot
            merged_free_slots.append((current_start, current_end))

        # Format the free time slots for this day
        if merged_free_slots:
            date_str = format_date(current_date)
            formatted_slots = [
                format_time_slot(start, end) for start, end in merged_free_slots
            ]
            free_time_by_day[date_str] = formatted_slots

    return free_time_by_day


def main():
    """Main function to find and display free time slots."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Find free time slots in Google Calendar"
    )
    parser.add_argument(
        "--weekends",
        "-w",
        action="store_true",
        help="Include weekend days in the output (default: False)",
    )
    parser.add_argument(
        "--today",
        "-t",
        action="store_true",
        help="Include today in the output (default: False)",
    )
    args = parser.parse_args()

    # Authenticate and get the Google Calendar service
    scopes = ["https://www.googleapis.com/auth/calendar.readonly"]
    creds = authenticate(scopes)
    service = build("calendar", "v3", credentials=creds)

    # Get today's or tomorrow's date as the start date, making it timezone-aware
    timezone = zoneinfo.ZoneInfo(
        "America/New_York"
    )  # Use the same timezone as in find_free_time_slots
    today = datetime.datetime.now(timezone)

    if args.today:
        # Use today as the start date
        start_date = today
    else:
        # Use tomorrow as the start date (default)
        start_date = today + datetime.timedelta(days=1)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Find free time slots for the next three weeks
    free_times = find_free_time_slots(
        service, start_date, include_weekends=args.weekends
    )

    # Display the results
    prev_day = None
    for date, slots in free_times.items():
        # Extract the day of week from the date string (e.g., "Mon" from "Mon 03/25")
        current_day = date.split()[0]

        # Add blank line before Monday
        if current_day == "Mon" and prev_day is not None and prev_day != "Fri":
            print()

        # Print the current day's free slots
        print(f"{date}: {', '.join(slots)} ET")

        # Add blank line after Friday
        if current_day == "Fri":
            print()

        prev_day = current_day


if __name__ == "__main__":
    main()
