import requests
import dotenv
import os
import json

dotenv.load_dotenv()

def get_access_token():

    url =  f"{os.getenv('instance_url')}/services/oauth2/token"
    bs4 = os.getenv("bs4")
    payload = os.getenv("payload")
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {bs4}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    access_json = response.json()

    return access_json['access_token']

def upload_files_multi(params):
    filepaths = params["filepaths"]
    case_id = params["case_id"]
    url = f"{os.getenv('instance_url')}/services/data/{os.getenv('version_number')}/composite/sobjects"
    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}"
    }

    records = []
    files_payload = {}
    opened = []

    for i, path in enumerate(filepaths, 1):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        filename = os.path.basename(path)
        part = f"file{i}"

        records.append({
            "attributes": {
                "type": "ContentVersion",
                "binaryPartName": part,
                "binaryPartNameAlias": "VersionData"
            },
            "Title": filename,
            "PathOnClient": filename,
            "FirstPublishLocationId": case_id
        })

        f = open(path, "rb")
        opened.append(f)

        files_payload[part] = (filename, f, "application/octet-stream")

    entity_json = json.dumps({
        "allOrNone": True,
        "records": records
    })

    files_payload["entity_content"] = (None, entity_json, "application/json")

    res = requests.post(url, headers=headers, files=files_payload)

    for f in opened:
        f.close()

    return res.status_code, res.json()

# Sample usage
param = {"filepaths":[r"C:\Users\syed.hasnain\Downloads\out\sampPy.py",r"C:\Users\syed.hasnain\Downloads\out\input.xlsx"],"case_id":"500dn00000JK6UXAA1"}
print(upload_files_multi(param))
