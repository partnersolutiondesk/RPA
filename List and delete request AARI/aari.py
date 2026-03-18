import os
import requests
import json
import dotenv
import csv
from typing import List, Optional



dotenv.load_dotenv(r'C:\Users\syed.hasnain\Downloads\untitled1\.env')

def get_access_token_aari():
    url = f"{os.getenv('url')}/v2/authentication"
    payload = json.dumps({"username": f'{os.getenv("name")}',"password": f'{os.getenv("password")}'})
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url,headers = headers,data=payload)
    # print(response.status_code)
    # print(response.text)
    return response.json()

def request_list_ids_to_csv(output_file=r"output12.csv"):
    token_json = get_access_token_aari()
    token = token_json['token']
    url = f'{os.getenv("url")}/aari/v3/requests/list'

    headers = {
        'X-Authorization': f'{token}',
        'Content-Type': 'application/json'
    }

    offset = 0
    page_size = 10000

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id"])  # header

        while True:
            payload = {
                "filter": {
                    "operator": "and",
                    "operands": [
                        {"operator": "or", "operands": [{"operator": "eq", "field": "status", "value": "OPENED"}]},
                        {"operator": "or", "operands": [{"operator": "substring", "field": "process", "value": "CoPilotProcess copy(1)"}]},
                        {"operator": "or", "operands": [{"operator": "eq", "field": "processType", "value": "PROCESS"}]}
                    ]
                },
                "attrFilter": {},
                "sort": {"field": "createdOn", "direction": "desc"},
                "page": {"offset": offset, "length": page_size}
            }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json)

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break

            data = response.json()

            # extract IDs
            records = data.get("list", [])
            if not records:
                print("No more records found.")
                break

            for record in records:
                writer.writerow([record["id"]])

            print(f"Fetched {len(records)} records (offset={offset})")

            # stop if last batch
            if len(records) < page_size:
                print("Last batch received.")
                break

            offset += page_size


def delete_records_from_csv(
        csv_path: str,
        batch_size: int = 5000,
        id_column_index: int = 0,
        has_header: bool = True,
        timeout: int = 30
) -> List[dict]:
    """
    Reads IDs from a CSV and calls delete API in batches.

    Args:
        csv_path (str): Path to CSV file
        api_url (str): API endpoint
        token (str): X-Authorization token
        batch_size (int): Number of IDs per request
        id_column_index (int): Column index where ID exists
        has_header (bool): Skip first row if True
        timeout (int): Request timeout

    Returns:
        List[dict]: List of responses for each batch
    """
    token_json = get_access_token_aari()
    token = token_json['token']

    api_url = f"{os.getenv('url')}/aari/v2/requests/delete"


    def read_ids() -> List[int]:
        ids = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)

            if has_header:
                next(reader, None)

            for row in reader:
                if not row or row[id_column_index].strip() == "":
                    continue
                try:
                    ids.append(int(row[id_column_index]))
                except ValueError:
                    print(f"Skipping invalid ID: {row}")
        return ids

    def chunk_list(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    ids = read_ids()

    if not ids:
        print("No valid IDs found.")
        return []

    headers = {
        "X-Authorization": token,
        "Content-Type": "application/json"
    }

    results = []

    for batch in chunk_list(ids, batch_size):
        try:
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

            print(f"Batch ({len(batch)} IDs) → {response.status_code}")

        except requests.exceptions.RequestException as e:
            result = {
                "status_code": None,
                "error": str(e),
                "batch_size": len(batch)
            }
            print(f"Batch failed: {e}")

        results.append(result)

    return results


print("Data fetching completed.")



# print(request_list_ids_to_csv())

#delete_records_from_csv(
#    csv_path=r"output12.csv"
#)
