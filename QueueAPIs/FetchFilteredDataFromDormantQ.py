import requests
import time

# Input configuration
input_config = {
    "DormantQueueDetails": [
        {"QueueID": "202", "TemplateID": "217"},
        {"QueueID": "201", "TemplateID": "216"},
        {"QueueID": "200", "TemplateID": "215"}
    ]
}

# Base URLs for templates and work items
BASE_TEMPLATE_URL = "https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/workitemmodels/"
BASE_WORK_ITEM_URL = "https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/queues/"

cached_token = None
token_expiry = 0

def get_token(url, payload):
    global cached_token, token_expiry

    # Check if the token is valid
    if cached_token and time.time() < token_expiry:
        return cached_token

    try:
        # Make the POST request
        response = requests.post(url, json=payload)

        # Check the response status
        if response.status_code == 200:
            json_response = response.json()

            # Extract the token and expiry
            if 'token' in json_response:
                cached_token = json_response['token']
                token_expiry = time.time() + 3600
                return cached_token
            else:
                return "Error: 'token' attribute not found in the response."

        else:
            return f"Error: Status code {response.status_code}, Response: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Function to fetch template attributes
def fetch_template_attributes(url_token,body_token,template_id):
    url = f"{BASE_TEMPLATE_URL}{template_id}"
    token = get_token(url_token,body_token)
    headers = {"X-Authorization": token}
    try:
        response = requests.get(url,headers=headers)
        response.raise_for_status()
        data = response.json()
        return {f"col{i+1}": attr["name"] for i, attr in enumerate(data.get("attributes", []))}
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch template {template_id} from {url}: {e}")
        return {}

# Function to fetch work items with payload
def fetch_work_items(url_token, body_token,queue_id, payload_dormant):
    url = f"{BASE_WORK_ITEM_URL}{queue_id}/workitems/list"
    token = get_token(url_token,body_token)
    headers = {"X-Authorization": token}
    try:
        response = requests.post(url, json=payload_dormant,headers=headers)
        response.raise_for_status()
        json_response = response.json()
        if 'list' in json_response:
            return json_response['list']
        # return response.json()  # Assuming API returns work items in JSON format
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch work items for queue {queue_id} from {url}: {e}")
        return []

# Function to map work item columns to attribute names
def map_work_items_to_template(work_items, column_mapping):
    mapped_items = []
    for item in work_items:
        mapped_item = {column_mapping[key]: item[key] for key in column_mapping if key in item}
        mapped_items.append(mapped_item)
    return mapped_items

# Main process
def main():
    # Prepare mappings from input configuration
    url_token = ''
    body_token = ''
    attribute_mappings = {}
    for detail in input_config["DormantQueueDetails"]:
        queue_id = detail["QueueID"]
        template_id = detail["TemplateID"]
        attribute_mappings[queue_id] = fetch_template_attributes(url_token, body_token,template_id)

    # Fetch and map work items
    for detail in input_config["DormantQueueDetails"]:
        queue_id = detail["QueueID"]
        payload = {"sort":[{"field":"computedStatus","direction":"asc"}],"filter":{},"fields":[],"page":{"length":100,"offset":0}}
        # print(f"Processing queue ID: {queue_id} with payload: {payload}")

        # Fetch work items for the queue
        work_items = fetch_work_items(url_token, body_token,queue_id, payload)
        # print("Work_Items"+(str)(work_items))
        # Map columns to template attributes
        column_mapping = attribute_mappings.get(queue_id, {})
        mapped_items = map_work_items_to_template(work_items, column_mapping)

        print(f"Mapped work items for queue {queue_id}:\n", mapped_items)

if __name__ == "__main__":
    main()
