import requests
import logging
import os

from logging_setup import setup_file_logger
# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))

logger = setup_file_logger(
    base_dir=base_dir,
    filename="app.log",
    level=logging.DEBUG,
    logger_name="app",
    backupCount=30,
    utc=True,
    compress=True
)
logger.info("Application started")


payload_folder = payload = {
    "fields": [],
    "filter": {
        "field": "type",
        "value": "application/vnd.aa.directory",
        "operator": "eq"
    },
    "sort": [
        {"field": "directory", "direction": "desc"},
        {"field": "type", "direction": "desc"},
        {"field": "name", "direction": "asc"}
    ],
    "page": {
        "offset": 0,
        "length": 1000
    }
}


_token_cache = None



def get_tokens(url,user_name,api_key):
    '''
    Function for getting tokens after authentication
    '''
    logger.info("Starting authentication request")


    session = requests.Session()
    auth_url = url + '/v2/authentication'
    payload ={
        'username':user_name,
        'api_key':api_key
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = session.post(auth_url, json=payload, headers=headers)

    if response.status_code != 200:
        logger.debug(f"Auth response status: {response.status_code}")
        raise Exception(f'Auth failed: {response.status_code} - {response.text}')

    data = response.json()
    token = data.get('token') or data.get('auth_token')

    if not token:
        logger.error("Token not found in response")
        raise Exception("Token not found")

    logger.info("Authentication successful")
    return token


def get_auth_token(url,user_name,api_key):

    '''
    An utiliy function which makes the authentication token cacheable
    '''
    global _token_cache
    if _token_cache is None:
        _token_cache = get_tokens(url,user_name,api_key)
    return _token_cache


def refresh_token(url,user_name,api_key):
    '''
    An utility function for getting new tokens if the current token is expired
    '''
    global _token_cache
    _token_cache = get_tokens(url,user_name,api_key)
    return _token_cache



def fetch_folders(url,user_name,api_key,parent_id):
    logger.info(f"Fetching folders for parent_id={parent_id}")

    api_url = url + f'/v2/repository/folders/{parent_id}/list'

    token = get_auth_token(url,user_name,api_key)

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Authorization': token
    }

    response = requests.post(api_url, json=payload_folder, headers=headers)
    logger.debug(f"Response status: {response.status_code}")
    if response.status_code == 401:
        logger.warning("Token expired, refreshing token")
        token = refresh_token()
        headers['X-Authorization'] = token
        response = requests.post(api_url, json=payload_folder, headers=headers)

    response.raise_for_status()
    return response.json()



def build_tree(url,user_name,password,parent_id):

    try:
        data = fetch_folders(url,user_name,password,parent_id)
    except Exception as e:
        logger.exception(f"Failed to fetch parent {parent_id}")
        return []

    result = []

    for folder in data.get('list', []):
        logger.debug(f"Processing folder: {folder.get('id')} - {folder.get('name')}")

        try:
            child_data = fetch_folders(url,user_name,password,folder['id'])
        except Exception as e:
            logger.warning(f"Failed to fetch children for {folder['id']}: {e}")
            child_data = {"list": []}

        node = {
            "id": folder["id"],
            "name": folder["name"], 
            "folders": [
                child.get("name")
                for child in child_data.get("list", [])
            ]
        }

        result.append(node)
    logger.info(f"Completed tree for parent_id={parent_id}, nodes={len(result)}")
    return result



if __name__ == "__main__":
    root_id = '1105'
    url = 'cr-url'
    user_name= 'username'
    api_key = 'api_key'
    folder_tree = build_tree(url,user_name,api_key,root_id)
    print(folder_tree)