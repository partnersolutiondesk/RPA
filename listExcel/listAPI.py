import requests
from datetime import datetime
import copy
import pandas as pd

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



def build_filter(field, value, operator):
    if value is None:
        return None
    return {
        "field": field,
        "value": value,
        "operator": operator
    }

def build_date_filter(start_date=None, end_date=None):

    logger.info(
        "Building date filters with start_date=%s and end_date=%s",
        start_date, end_date
    )

    date_filters = []

    start_date = normalize_date(start_date) if start_date else None
    end_date = normalize_date(end_date) if end_date else None


    logger.debug(
        "Normalized dates: start_date=%s, end_date=%s",
        start_date, end_date
    )


    if start_date:
        date_filters.append({
            "field": "endDateTime",
            "operator": "gt",
            "value": start_date
        })
        logger.debug("Added start_date filter")

    if end_date:
        date_filters.append({
            "field": "endDateTime",
            "operator": "lt",
            "value": end_date
        })
        logger.info("Date filters built successfully: %s", date_filters)

    return date_filters

def build_payload(
        device_name=None,
        status=None,
        automation_name=None,
        priority=None,
        file_name=None,
        user_name=None,
        start_date=None,
        end_date=None
):

    logger.info("Building payload: %s", device_name,status,automation_name,priority,file_name,user_name,start_date,end_date)
    filters = []

    # Optional filters
    optional_filters = [
        build_filter("deviceName", device_name, "substring"),
        build_filter("status", status, "eq"),
        build_filter("automationName", automation_name, "substring"),
        build_filter("automationPriority", priority, "eq"),
        build_filter("fileName", file_name, "substring"),
        build_filter("userName", user_name, "substring"),
    ]

    # Remove None values
    optional_filters = [f for f in optional_filters if f]

    if optional_filters:
        filters.append({
            "operator": "and",
            "operands": optional_filters
        })

    # Date filters
    date_filters = build_date_filter(start_date, end_date)
    if date_filters:
        filters.append({
            "operator": "and",
            "operands": date_filters
        })

    payload = {
        "fields": [],
        "sort": [{"field": "endDateTime", "direction": "desc"}],
        "page": {"offset": 0, "length": 10000}
    }

    if filters:
        payload["filter"] = {
            "operator": "and",
            "operands": filters
        }

    logger.info("Built payload sucessfully")

    return payload


def get_auth_token(url, username, api_key):



    auth_url = url+'/v2/authentication'
    logger.info("getting token for ", auth_url)
    payload = {
        "username": username,
        "api_key": api_key
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(auth_url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Auth failed: {response.status_code} - {response.text}")

    data = response.json()

    # Adjust this depending on API response structure
    token = data.get("token") or data.get("access_token")

    logger.info("successfully got token  ", token)

    if not token:
        logger.error("failed getting token for ", url)
        raise Exception("Token not found in response")

    return token

def call_main_api(url, payload, token):

    api_url = url+'/v3/activity/list?historical=true'
    logger.info("calling main api ", api_url)
    headers = {
        "Content-Type": "application/json",
        "X-Authorization": f"{token}"
    }

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        logger.error(f"API failed: {response.status_code} - {response.text}")
        raise Exception(f"API failed: {response.status_code} - {response.text}")

    logger.info("successfully called ", response.status_code)
    return response.json()






def fetch_all_and_save_excel(config):

    url = config.get("url")
    username = config.get("username")
    api_key = config.get("api_key")
    payload_config = config.get("payload_config")
    file_name = config.get("file_name", "output.xlsx")
    batch_size = config.get("batch_size", 10000)

    # Step 1: Auth
    token = get_auth_token(url, username, api_key)

    offset = 0
    total_count = 0
    first_write = True
    base_payload = build_payload(**(payload_config or {}))


    with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:

        while True:

            payload = copy.deepcopy(base_payload)
            payload["page"] = {
                "offset": offset,
                "length": batch_size
            }

            response = call_main_api(url, payload, token)

            # Step 2: Safe extraction
            records = []

            if isinstance(response, dict):
                records = response.get("list", [])

            if not isinstance(records, list):
                records = []

            # Keep only valid dicts
            records = [r for r in records if isinstance(r, dict)]

            count = len(records)
            logger.info(f"Fetched {count} records at offset {offset}")
            print(f"Fetched {count} records at offset {offset}")

            if count == 0:
                break

            df = pd.json_normalize(records, sep="_")

            logger.info(f"Writing to excel")

            # Step 3: Write to Excel (append logic)
            df.to_excel(
                writer,
                index=False,
                sheet_name="Sheet1",
                startrow=0 if first_write else writer.sheets["Sheet1"].max_row,
                header=first_write
            )

            first_write = False
            total_count += count

            # Stop condition
            if count < batch_size:
                break

            offset += batch_size

    logger.info(f"Total records fetched: {total_count}")
    logger.info(f"Saved to: {file_name}")

    print(f"Total records fetched: {total_count}")
    print(f"Saved to: {file_name}")

    return total_count


def normalize_date(date_str: str) -> str:
    """
    Normalizes date input to ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ

    स्वीकारed formats:
    - 'YYYY-MM-DD'
    - 'YYYY-MM-DDTHH:MM:SSZ' (returned as-is)

    Returns:
        str: ISO formatted date string
    """
    if not date_str:
        return None

    try:
        # If already in full ISO format, validate and return
        if "T" in date_str and date_str.endswith("Z"):
            datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            return date_str

        # If only date is provided
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        return parsed_date.strftime("%Y-%m-%dT00:00:00Z")

    except ValueError:
        raise ValueError(
            f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ'"
        )

#=============================================

cr_url = 'https://a-rpa.digital'

username = "user"
api_key = "api_key"



payload_config = {
    "start_date":"2026-04-06T18:30:00Z",
    "end_date":"2026-04-14T18:29:00Z"
}

file_path=r"file-path"

config = {
    "url": cr_url,
    "username": username,
    "api_key": api_key,
    "payload_config": payload_config,
    "file_name": file_path
}


all_data = fetch_all_and_save_excel(config)
print(all_data)

# print(f"Total records fetched: {len(all_data)}")


