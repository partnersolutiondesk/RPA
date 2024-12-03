import requests
import time

cached_token = None
token_expiry = 0

# Fetching token for authentication
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

# Getting the data from the feeder Q
def dormant_api(url_token, body_token,url_dormant_queue, payload_dormant_queue):
    token = get_token(url_token,body_token)
    headers = {"X-Authorization": token}
    try:
        response = requests.post(url_dormant_queue, json=payload_dormant_queue, headers=headers)
        if response.status_code == 200:  # Adjust based on your API's success status code
            json_response = response.json()
            if 'list' in json_response:
                return json_response['list']
            else:
                return "Error: 'token' attribute not found in the response."
        else:
            return f"Error: Status code {response.status_code}, Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Fetching the number of devices associated with the dormant Q's
def dormant_api_devices_fetch(url_token, body_token,url_dormant_queue, payload_dormant_queue,device_url):
    token = get_token(url_token,body_token)
    headers = {"X-Authorization": token}
    try:
        response = requests.post(url_dormant_queue, json=payload_dormant_queue, headers=headers)
        if response.status_code == 200:  # Adjust based on your API's success status code
            json_response = response.json()
            if 'list' in json_response:
                # print(json_response["list"][0]["poolId"])
                response = requests.get(device_url+json_response["list"][0]["poolId"], headers=headers)
                if response.status_code==200:
                    json_response = response.json()
                device_ids = json_response.get('deviceIds', [])
                device_count = len(device_ids)
                return device_count
            else:
                return "Error: 'token' attribute not found in the response."
        else:
            return f"Error: Status code {response.status_code}, Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Function to count number of devices in active Q
def device_count_api(url_token, body_token,device_pool_url):
    token = get_token(url_token,body_token)
    headers = {"X-Authorization": token}
    try:
        # Make the POST requestet(device_pool_url ,headers=headers)
        response = requests.get(device_pool_url,headers=headers)
        if response.status_code == 200:  # Adjust based on your API's success status code
            json_response = response.json()
            device_ids = json_response.get('deviceIds', [])
            device_count = len(device_ids)
            return device_count


        else:
            return f"Error: Status code {response.status_code}, Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    # return 5

# Counting ready to run items in active Q
def count_ready_to_run(url_token, body_token,url_active, body_active):
    ##url
    ## url_active = "https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/queues/192/workitems/list"
    token = get_token(url_token,body_token)
    headers = {"X-Authorization": token}

    try:
        # Make the POST request
        response = requests.post(url_active, json=body_active, headers=headers)

        # Raise an error if the response is not successful
        response.raise_for_status()

        # Parse the response JSON
        response_data = response.json()

        # Ensure 'list' key exists in the response
        if 'list' not in response_data:
            print("Error: 'list' key not found in the response.")
            return 0

        # Count the number of items with status 'READY_TO_RUN'
        items = response_data['list']
        ready_to_run_count = sum(1 for item in items if item.get("status") == "READY_TO_RUN")

        return ready_to_run_count

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return 0

# Data that can be filled based device pool minus ready_to_run
def num_to_fill(url_token, body_token,url_active, body_active,device_pool_api):
    count = device_count_api(url_token,body_token,device_pool_api)-count_ready_to_run(url_token, body_token,url_active, body_active)
    return count

# function to compund sort and extract data of the feeder Q
def extract_sorted_items(url_token, body_token, url_dormant_queue, payload_dormant_queue, url_active, body_active, device_pool_api,queue_device_urls,payload_for_queue_devices,device_url):
    # Define the total number of items to extract
    num_items = num_to_fill(url_token, body_token, url_active, body_active, device_pool_api)

    # Collect all items from dormant queues
    all_items = []
    device_counts = []
    total_device_count = 0
    for url, payload in zip(queue_device_urls, payload_for_queue_devices):
        count = dormant_api_devices_fetch(url_token, body_token, url, payload,device_url)
        device_counts.append(count)
        # print(count)
        total_device_count += count

    # Extract and proportionally allocate items based on device count
    for queue_index, (url, payload) in enumerate(zip(url_dormant_queue, payload_dormant_queue)):
        # Get the items from the current dormant queue
        items = dormant_api(url_token, body_token, url, payload)
        # print(items)

        # Get the device count for the current queue
        queue_device_count = device_counts[queue_index]
        # Calculate the number of items to extract for this queue
        items_to_extract = int((queue_device_count / total_device_count) * num_items)
        print(items_to_extract)

        # Sort the items for the current queue
        sorted_items = sorted(
            items,
            key=lambda x: (int(x.get('col3', 0)), int(x.get('col2', 0)))
        )

        # Add the proportional number of items to the final list
        all_items.extend(sorted_items[:items_to_extract])
    print(len(all_items))

    # Combine all items and sort again globally (if needed)
    final_sorted_items = sorted(
        all_items,
        key=lambda x: (int(x.get('col3', 0)), int(x.get('col2', 0)))
    )

    # Build the result with only necessary keys
    result = [
        {key: item[key] for key in ['col1', 'col2', 'col3'] if key in item} for item in final_sorted_items
    ]

    return result

# fetching the template of the dormant Q
def queueTemplate(template_url,url_token, body_token):
    token = get_token(url_token,body_token)
    headers = {"X-Authorization": token}
    try:
        # Make the POST request
        response = requests.get(template_url, headers=headers)

        # Check the response status
        if response.status_code == 200:  # Adjust based on your API's success status code
            json_response = response.json()
            if 'attributes' in json_response:
                return json_response['attributes']
            # return json_response
            # Extract the token attribute
            # if 'attributes' in json_response:
            #     return json_response['attributes']
            # else:
            #     return "Error: 'token' attribute not found in the response."

        else:
            return f"Error: Status code {response.status_code}, Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Adding the names of columns to the feeder queue data
def transform_template_with_attributes(template_url,url_token, body_token,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url):
    # Extract the ordered column names from attributes
    response = extract_sorted_items(url_token, body_token,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url)
    attributes = queueTemplate(template_url,url_token, body_token)
    column_order = [attr['name'] for attr in attributes]

    # Generate the key mapping based on the column order
    key_mapping = {f'col{i + 1}': column_order[i] for i in range(len(column_order))}

    # Transform the response using the dynamically generated mapping
    transformed_response = [
        {key_mapping.get(key, key): value for key, value in item.items() if key != 'id'}
        for item in response
    ]
    print(transformed_response)
    return transformed_response

# Preparing payload to upload to invoking the active Q api
def prepare_payload(template_url,url_token, body_token,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url, key_transform=None, value_transform=None):
    """
    Converts input data into the desired payload format.

    Args:
        data (list of dict): The input data to transform.
        key_transform (function): Function to transform keys, if needed (optional).
        value_transform (function): Function to transform values, if needed (optional).

    Returns:
        dict: A payload formatted as per the desired structure.
    """
    data = transform_template_with_attributes(template_url,url_token, body_token,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url)
    def transform_key_value(item):
        transformed_item = {}
        for key, value in item.items():
            new_key = key_transform(key) if key_transform else key
            new_value = value_transform(value) if value_transform else value
            transformed_item[new_key] = new_value
        return transformed_item

    return {
        "workItems": [
            {"json": transform_key_value(item)} for item in data
        ]
    }

def generate_payload(template_url,url_token, body_token,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url):
    """
    Generates and returns the prepared payload.

    :param data: List of dictionaries containing raw data.
    :return: Prepared payload.
    """
    return prepare_payload(template_url,url_token, body_token,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url, key_transform=None, value_transform=value_transform)

def value_transform(value):
    """
    A function to transform values. Converts numeric strings to integers.
    """
    return int(value) if value.isdigit() else value

def invoke_api(url_token,body_token,api_url,template_url,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url):
    token = get_token(url_token,body_token)
    json_payload = generate_payload(template_url,url_token, body_token,url_dormant_queue, payload_dormant_queue,url_active,body_active,device_pool_api,queue_device_urls,payload_for_queue_devices,device_url)
    headers = {"Content-Type": "application/json","X-Authorization": token}

    try:
        # Make the POST request
        response = requests.post(api_url, json=json_payload, headers=headers)

        # Check the response status
        if response.status_code == 200:  # Adjust based on your API's success status code
            json_response = response.json()
            return json_response
        else:
            return f"Error: Status code {response.status_code}, Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"




url_token = "url_token_for_authentication"  # Replace with your API URL
body_token = {
    "Body_for_authentication"
}
dormant_queue_urls = [
    "https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/queues/193/workitems/list",
    "https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/queues/195/workitems/list",
    "https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/queues/196/workitems/list"
]
payloads_dormant_queues = [
    {"sort":[{"field":"computedStatus","direction":"asc"}],"filter":{},"fields":[],"page":{"length":100,"offset":0}},  # Payload for dormant queue 1
    {"sort":[{"field":"computedStatus","direction":"asc"}],"filter":{},"fields":[],"page":{"length":100,"offset":0}},  # Payload for dormant queue 2
    {"sort":[{"field":"computedStatus","direction":"asc"}],"filter":{},"fields":[],"page":{"length":100,"offset":0}}  # Payload for dormant queue 3
]
device_url = 'https://aa-se-ind-5.my.automationanywhere.digital/v2/devices/pools/360'
device_url_fetch = 'https://aa-se-ind-5.my.automationanywhere.digital/v2/devices/pools/'
# result_dormant = call_post_api_with_headers(url_token,body_token,url_dormant,body_dormant)
# result = get_token(api_url, body)
# print("API Response:", )
# print(device_count_api())
url_active= "https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/queues/192/workitems/list"
body_active = {"sort":[{"field":"computedStatus","direction":"asc"}],"filter":{},"fields":[],"page":{"length":100,"offset":0}}
template_url = 'https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/workitemmodels/210'
# print(extract_sorted_items(url_token, body_token,url_dormant, body_dormant,url_active,body_active,device_url))
active_post_url = 'https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/queues/192/workitems'
# print(transform_template_with_attributes(template_url,url_token, body_token,url_dormant, body_dormant,url_active,body_active,device_url))
# print(device_count_api(url_token,body_token,device_url))
queue_device_urls = ['https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/automations/list',
                     'https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/automations/list',
                     'https://aa-se-ind-5.my.automationanywhere.digital/v3/wlm/automations/list']
payload_for_queue_devices = [{"fields":[],"filter":{"operator":"eq","value":"193","field":"queueId"},"page":{"offset":0,"length":1,"total":0,"totalFilter":0},"sort":[]},
                             {"fields":[],"filter":{"operator":"eq","value":"195","field":"queueId"},"page":{"offset":0,"length":1,"total":0,"totalFilter":0},"sort":[]},
                             {"fields":[],"filter":{"operator":"eq","value":"196","field":"queueId"},"page":{"offset":0,"length":1,"total":0,"totalFilter":0},"sort":[]}]

print(invoke_api(url_token,body_token,active_post_url,template_url,dormant_queue_urls,payloads_dormant_queues,url_active,body_active,device_url,queue_device_urls,payload_for_queue_devices,device_url_fetch))
