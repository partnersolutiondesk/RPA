# Folder Fetcher

A Python utility to parent folder and its children folder

## What it does
- get_tokens : Authenticates using username/api_key
- get_auth_token : Caches the auth token for reuse

- fetch_folders :  Fetches folder names from a remote API
- build_tree : Builds a parent -> child folder structure
- Logs activity to a file (`app.log`)


### Parameters
- url : Base url
- user_name 
- password
- id: id of the public folder

### Instructions


- Fetch your API token from the control, add it in the api_key parameter of the script
- Add your control room URL for in the URL parameter
- Get the public folder ID from the endpoint.


