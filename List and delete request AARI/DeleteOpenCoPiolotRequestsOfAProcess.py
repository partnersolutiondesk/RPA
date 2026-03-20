import os
import json
import csv
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

# =========================================
# Paths and .env loading
# =========================================
BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"

# Load .env (before logger so we can honor LOG_LEVEL)
loaded_env = load_dotenv(DOTENV_PATH)

# =========================================
# Logging setup
# =========================================
def setup_logger(level_name: str = "INFO") -> logging.Logger:
    """
    Create a logger that writes to app.log in the same folder as app.py,
    with rotation (1MB, 5 backups). Also logs to console.
    """
    logger = logging.getLogger("aari_app")

    # Parse level from string safely
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers (e.g., module reloads)
    if logger.handlers:
        return logger

    LOG_FILE = BASE_DIR / "app.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Optional console output (helpful during interactive runs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger

# Set log level from .env if present
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger = setup_logger(LOG_LEVEL)

# Log .env result after logger is ready
if loaded_env:
    logger.info(f".env loaded from: {DOTENV_PATH}")
else:
    logger.warning(f".env not found at: {DOTENV_PATH}. Using system environment variables if available.")

# =========================================
# Helpers
# =========================================
def _masked(value: Optional[str], keep: int = 6) -> str:
    """Mask sensitive values; keep first `keep` chars for traceability."""
    if not value:
        return "<empty>"
    if len(value) <= keep:
        return value + "***"
    return value[:keep] + "..."

def resolve_output_csv_path(env_key: str = "OUTPUT_CSV_FILENAME",
                            default_name: str = "output.csv") -> Path:
    """
    Resolve output CSV path from .env. If relative, place it next to app.py (BASE_DIR).
    If absolute, use as-is. Create parent dirs if needed.
    """
    name = os.getenv(env_key, default_name)
    name = (name or default_name).strip()
    p = Path(name)

    if p.is_absolute():
        output_path = p
    else:
        # Put relative paths inside the same repo folder as app.py
        output_path = BASE_DIR / p

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Resolved output CSV path: {output_path}")
    return output_path

# =========================================
# API functions
# =========================================
def get_access_token_aari() -> dict:
    """
    Authenticates against control room and returns token response JSON.
    Requires env vars: controlroom_URL, cr_username, cr_password
    """
    controlroom_url = os.getenv("controlroom_URL")
    cr_username = os.getenv("cr_username")
    cr_password = os.getenv("cr_password")

    logger.info("Starting authentication request to control room")
    logger.debug(f"controlroom_URL={controlroom_url}, cr_username={cr_username}")

    if not controlroom_url or not cr_username or not cr_password:
        logger.error("Missing required environment variables: controlroom_URL, cr_username, or cr_password")
        raise ValueError("Missing required environment variables for authentication")

    url = f"{controlroom_url}/v2/authentication"
    payload = {"username": cr_username, "password": cr_password}
    headers = {"Content-Type": "application/json"}

    try:
        logger.info(f"POST {url} (auth)")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        logger.info(f"Auth response status={response.status_code}")
        if response.status_code != 200:
            logger.error(f"Authentication failed: status={response.status_code}, body={response.text}")
            response.raise_for_status()

        data = response.json()
        token = data.get("token")
        logger.info(f"Auth succeeded. Token present={bool(token)}; token_prefix={_masked(token)}")
        return data
    except requests.exceptions.RequestException as e:
        logger.exception(f"Authentication request error: {e}")
        raise

def request_list_ids_to_csv(output_file: Optional[str] = None) -> str:
    """
    Fetches request IDs using pagination and writes to CSV.

    If output_file is None, it will be loaded from .env (OUTPUT_CSV_FILENAME) and
    placed in the same repo folder as app.py (BASE_DIR).
    """
    logger.info("Starting request_list_ids_to_csv")

    # Resolve output path (env-driven by default)
    if output_file is None:
        output_path = resolve_output_csv_path()
    else:
        p = Path(output_file)
        output_path = p if p.is_absolute() else (BASE_DIR / p)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output CSV path (from argument): {output_path}")

    token_json = get_access_token_aari()
    token = token_json.get("token")
    controlroom_url = os.getenv("controlroom_URL")
    process_name = os.getenv("process_name")

    if not controlroom_url or not process_name:
        logger.error("Missing controlroom_URL or process_name in environment")
        raise ValueError("Missing controlroom_URL or process_name")

    url = f"{controlroom_url}/aari/v3/requests/list"
    headers = {
        "X-Authorization": token,
        "Content-Type": "application/json"
    }

    offset = 0
    page_size = 10000
    total_written = 0

    logger.info(f"Listing requests from {url} for process_name='{process_name}' with page_size={page_size}")
    logger.info(f"Writing IDs to: {output_path}")

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id"])  # header
        logger.debug("CSV header written")

        while True:
            payload = {
                "filter": {
                    "operator": "and",
                    "operands": [
                        {"operator": "or", "operands": [{"operator": "eq", "field": "status", "value": "OPENED"}]},
                        {"operator": "or", "operands": [{"operator": "substring", "field": "process", "value": f"{process_name}"}]},
                        {"operator": "or", "operands": [{"operator": "eq", "field": "processType", "value": "PROCESS"}]}
                    ]
                },
                "attrFilter": {},
                "sort": {"field": "createdOn", "direction": "desc"},
                "page": {"offset": offset, "length": page_size}
            }

            try:
                logger.info(f"POST {url} (list) | offset={offset}, length={page_size}")
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                logger.info(f"List response status={response.status_code}")

                if response.status_code != 200:
                    logger.error(f"Error listing requests: status={response.status_code}, body={response.text}")
                    break

                data = response.json()
                records = data.get("list", [])
                count = len(records)
                logger.info(f"Fetched {count} records at offset={offset}")

                if not records:
                    logger.info("No more records found. Stopping pagination.")
                    break

                for record in records:
                    try:
                        writer.writerow([record["id"]])
                        total_written += 1
                    except Exception as e:
                        logger.exception(f"Failed to write record to CSV: {record}. Error: {e}")

                logger.info(f"Cumulative records written: {total_written}")

                # stop if last batch
                if count < page_size:
                    logger.info("Last batch received. Pagination complete.")
                    break

                offset += page_size

            except requests.exceptions.RequestException as e:
                logger.exception(f"List request error: {e}")
                break

    logger.info(f"CSV writing completed: {total_written} IDs written to {output_path}")
    return str(output_path)

def delete_records_from_csv(
    csv_path: str,
    batch_size: int = 5000,
    id_column_index: int = 0,
    has_header: bool = True,
    timeout: int = 30
) -> List[dict]:
    """
    Reads IDs from a CSV and calls delete API in batches.
    Returns a list of per-batch result dicts.
    """
    logger.info("Starting delete_records_from_csv")
    logger.info(f"CSV path: {csv_path}, batch_size={batch_size}, id_column_index={id_column_index}, has_header={has_header}")

    token_json = get_access_token_aari()
    token = token_json.get("token")
    controlroom_url = os.getenv("controlroom_URL")

    if not controlroom_url:
        logger.error("Missing controlroom_URL in environment")
        raise ValueError("Missing controlroom_URL in environment")

    api_url = f"{controlroom_url}/aari/v2/requests/delete"
    logger.info(f"Delete endpoint: {api_url}")

    def read_ids() -> List[int]:
        ids: List[int] = []
        p = Path(csv_path)
        if not p.exists():
            logger.error(f"CSV file not found: {p}")
            return ids
        try:
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)

                if has_header:
                    skipped = next(reader, None)
                    logger.debug(f"Skipped header row: {skipped}")

                for row in reader:
                    if not row or id_column_index >= len(row) or row[id_column_index].strip() == "":
                        logger.debug(f"Skipping empty/invalid row: {row}")
                        continue
                    try:
                        ids.append(int(row[id_column_index]))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid ID row: {row} | error: {e}")
        except Exception as e:
            logger.exception(f"Failed reading CSV: {e}")
        logger.info(f"Total IDs read from CSV: {len(ids)}")
        return ids

    def chunk_list(data: List[int], size: int):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    ids = read_ids()

    if not ids:
        logger.warning("No valid IDs found. Nothing to delete.")
        return []

    headers = {
        "X-Authorization": token,
        "Content-Type": "application/json"
    }

    results: List[dict] = []

    for batch in chunk_list(ids, batch_size):
        try:
            logger.info(f"Deleting batch with {len(batch)} IDs")
            response = requests.post(
                api_url,
                headers=headers,
                json={"ids": batch},
                timeout=timeout
            )

            result = {
                "status_code": response.status_code,
                "response": response.text,
                "batch_size": len(batch)
            }

            if response.status_code == 200:
                logger.info(f"Batch ({len(batch)} IDs) → {response.status_code}")
            else:
                logger.error(f"Batch delete error: status={response.status_code}, body={response.text}")

        except requests.exceptions.RequestException as e:
            result = {
                "status_code": None,
                "error": str(e),
                "batch_size": len(batch)
            }
            logger.exception(f"Batch failed: {e}")

        results.append(result)

    logger.info(f"Deletion completed. Total batches: {len(results)}")
    return results

# =========================================
# Entrypoint
# =========================================
if __name__ == "__main__":
    try:
        logger.info("=== Script start ===")
        csv_path = request_list_ids_to_csv()  # uses OUTPUT_CSV_FILENAME from .env
        logger.info(f"Data fetching completed. Output: {csv_path}")

        #IMPORTANT - THIS DELETE CODE WILL PERMANENTLY REMOVE THE REQUESTS
        # Uncomment to run deletion step using the same resolved CSV path
        # results = delete_records_from_csv(csv_path=csv_path)
        # logger.info(f"Deletion results: {results}")

        logger.info("=== Script end ===")
    except Exception as exc:
        logger.exception(f"Unhandled exception: {exc}")
        raise