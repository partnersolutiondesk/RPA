# Author: Sikha Poyyil
# Department : Global Solution Desk
# Company: Automation Anywhere
import csv
# Note: Below external libraries to be installed for script to work.
# 1. Requests-toolbelt library to be installed using below cmd for BLM apis to work
#    Command: pip install requests-toolbelt

import os
import copy
import json
import uuid
import shutil
import inspect
import logging
import re
import requests
import datetime as dt
from pathlib import Path
from requests_toolbelt.multipart.encoder import MultipartEncoder
import traceback

# Global variables defined
logger = logging.getLogger(__name__)
keep_db_schema_in_cache = None
update_expr_in_comment = None
update_error_desc_in_finally_with_given_var = None
resize_window_in_recorder_capture = None
reportdata = []
lineNumber = 0
header_entry = ['Bot Name', 'Folder path', 'Line number', 'API Uri', 'Status', 'Command Name', 'Package name']
reportdata.append(header_entry)
reportFilePath = ""


# Log the messages using logging library
def log(log_msg, log_level):
    # Automatically log the current function details.
    # Get the previous frame in the stack, otherwise it would be this function!!!
    func = inspect.currentframe().f_back.f_code

    # Dump the message + the name of this function to the log.
    logger.log(level=getattr(logging, log_level.upper(), None), msg='{0}): {1}'.format(func.co_name, log_msg))


# Initializing a logger with custom configuration
def initialize_logger(log_file_path, log_level):
    logger.setLevel(getattr(logging, log_level.upper()))
    file_handler = logging.FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s (%(message)s', datefmt='(%d-%m-%Y %I.%M.%S %p)')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log("Log file started.", 'info')


# Get user token status - valid or invalid
def token_status(cr_url, user_token):
    headers = {"accept": "application/json"}

    log("Token status: no data", 'debug')
    log("Headers: " + str(headers), 'debug')
    log("URL: " + cr_url + '/v1/authentication/token?token=' + str(user_token)[0:20], 'debug')

    response = requests.get(cr_url + '/v1/authentication/token?token=' + user_token, headers=headers)

    # Checking if response status is 200
    if response.status_code == 200:
        json_obj = response.json()
        log('Token status: ' + str(json_obj.get('valid')), 'info')
        # print('Token status: ' + str(json_obj.get('valid')))
        return str(json_obj.get('valid'))

    else:
        # Returning error json object
        error_json_obj = response.json()
        log(error_json_obj, 'info')
        # print(str(error_json_obj))
        return error_json_obj


# Generate user token in A360 Control room
def generate_token(cr_url, username, password, api_key):
    if password == '' and api_key.strip() != '':
        # Api key is not null and password is null
        data = '{ \"username\": \"' + str(username) + '\", \"apiKey\": \"' + str(api_key) + '\"}'
        log("Generate token: data = " + '{ \"username\": \"*********\", \"apiKey\": \"*********\"}', 'debug')

    elif password != '' and api_key.strip() == '':
        # Password is not null and api key is null
        data = '{ \"username\": \"' + str(username) + '\", \"password\": \"' + str(password) + '\"}'
        log("Generate token: data = " + '{ \"username\": \"*********\", \"password\": \"*********\"}', 'debug')

    elif password != '' and api_key.strip() != '':
        # Password and api key both are not null
        data = '{ \"username\": \"' + str(username) + '\", \"apiKey\": \"' + str(api_key) + '\"}'
        log("Generate token: data = " + '{ \"username\": \"*********\", \"apiKey\": \"*********\"}', 'debug')

    else:
        # Password and api key both are null
        result = json.dumps({'code': 'user.credentials.empty', 'details': '',
                             'message': 'Both password and api key are null. Please provide either one of them to generate user token'})
        log(result, 'info')
        return result

    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

    log("Headers: " + str(headers), 'debug')
    log("URL: " + cr_url + '/v2/authentication', 'debug')

    response = requests.post(cr_url + '/v2/authentication', data=data, headers=headers)

    # Checking if response status is 200
    if response.status_code == 200:
        json_obj = response.json()

        # print("User Token: " + str(json_obj.get('token'))[0:20] + ".******")
        log("User Token: " + str(json_obj.get('token'))[0:20] + ".****** generated.", 'info')
        # Returning user token
        token_ = str(json_obj.get('token'))
        log("token_token_: " + token_ + '', 'debug')
        return json_obj.get('token')
    else:
        error_json_obj = response.json()

        # print(str(error_json_obj))
        # Returning error json object
        log(str(error_json_obj), 'info')
        return error_json_obj


# Export bot from A360 Control room
def export_bot(cr_url, user_token, export_bot_name, bot_ids):
    data = '{\"name\": \"' + str(export_bot_name) + '\", \"fileIds\": ' \
           + str(bot_ids) + ', \"includePackages\": false}, \"archivePassword\": \"A360@123\"}'

    headers = {"X-Authorization": user_token, 'Content-type': 'application/json', 'Accept': 'text/plain'}

    log("Export bot: " + str(data), 'debug')
    log("Headers: " + str(
        {"X-Authorization": user_token[0:20] + ".******", 'Content-type': 'application/json', 'Accept': 'text/plain'}),
        'debug')
    log("URL: " + cr_url + '/v2/blm/export', 'debug')

    response = requests.post(cr_url + '/v2/blm/export', data=data, headers=headers)

    # Checking if response status is 202
    if response.status_code == 202:
        json_obj = response.json()
        # print("Exported package request ID: " + str(json_obj.get('requestId')))
        log("Exported package request ID: " + str(json_obj.get('requestId')), 'info')
        return json_obj.get('requestId')

    else:
        error_json_obj = response.json()
        # print(str(error_json_obj))
        log(str(error_json_obj), 'info')
        # Returning error json object
        return error_json_obj


# Get bot package export status
def bot_export_status(cr_url, request_id, user_token):
    headers = {"X-Authorization": user_token, "accept": "application/json"}

    log("Export bot status: no data", 'debug')
    log("Headers: " + str({"X-Authorization": user_token[0:20] + ".******", 'Content-type': 'application/json'}),
        'debug')
    log("URL: " + cr_url + '/v2/blm/status' + request_id, 'debug')

    response = requests.get(cr_url + '/v2/blm/status/' + request_id, headers=headers)

    # Checking if response status is 200
    if response.status_code == 200:
        json_obj = response.json()

        # Checking bot export status
        if json_obj.get('status').lower() == 'completed':
            # print("Download file ID for the exported package: " + str(json_obj.get('downloadFileId')))
            log("Download file ID for the exported package: " + str(json_obj.get('downloadFileId')), 'info')

            # Returning download file id attribute
            return json_obj.get('downloadFileId')

        else:
            # Returning wait
            return 'wait'
    else:
        error_json_obj = response.json()

        # print(str(error_json_obj))
        log(str(error_json_obj), 'info')

        # Returning error json object
        return error_json_obj


# Import bot package into A360 Control room
def bot_import(cr_url, user_token, data_folder_path, data_file_name):
    multipart_data = MultipartEncoder(
        fields={
            # a file upload field
            'upload': (
                data_file_name, open(os.path.join(data_folder_path, data_file_name), 'rb'),
                'application/x-zip-compressed'),
            # plain text fields
            'actionIfExisting': 'OVERWRITE',
            'publicWorkspace': 'true',
            'archivePassword': 'A360@123'
        }
    )

    headers = {"X-Authorization": user_token, "Content-Type": multipart_data.content_type}

    log("Import bot: " + str({
        # a file upload field
        'upload': {'File Path: ' + os.path.join(data_folder_path, data_file_name), 'application/x-zip-compressed'},
        'actionIfExisting': 'OVERWRITE',
        'publicWorkspace': 'true',
        'archivePassword': 'A360@123'
    }), 'debug')
    log("Headers: " + str(
        {"X-Authorization": user_token[0:20] + ".******", "Content-Type": multipart_data.content_type}), 'debug')
    log("URL: " + cr_url + '/v2/blm/import', 'debug')

    response = requests.post(cr_url + '/v2/blm/import', headers=headers, data=multipart_data)

    # Checking if response status is 202
    if response.status_code == 202:
        json_obj = response.json()

        # print("Imported bot package request ID: " + str(json_obj.get('requestId')))
        log("Imported bot package request ID: " + str(json_obj.get('requestId')), 'info')

        return json_obj.get('requestId')
    else:
        error_json_obj = response.json()

        # print(str(error_json_obj))
        log(str(error_json_obj), 'info')

        # Returning error json object
        return error_json_obj


# Download exported bot files into local folder
def download_file(cr_url, download_id, user_token, data_file_path):
    headers = {"X-Authorization": user_token, "accept": "*/*", "accept-encoding": "gzip;deflate;br"}

    log("Download file: no data", 'debug')
    log("Headers: " + str(
        {"X-Authorization": user_token[0:20] + ".******", "accept": "*/*", "accept-encoding": "gzip;deflate;br"}),
        'debug')
    log("URL: " + cr_url + '/v2/blm/download/' + download_id, 'debug')

    response = requests.get(cr_url + '/v2/blm/download/' + download_id, headers=headers)

    # Checking if response status is 200
    if response.status_code == 200:

        with open(data_file_path, 'wb') as output_file:
            output_file.write(response.content)

        # print("File downloaded to: " + str(data_file_path))
        log("File downloaded to: " + str(data_file_path), 'info')

        # Extracting zip file contents
        return "ok"

    else:
        error_json_obj = response.json()

        # print(str(error_json_obj))
        log(str(error_json_obj), 'info')

        # Returning error json object
        return error_json_obj


# Get file details using file path in A360 Control room
def fetch_file_details_by_path(cr_url, user_token, file_path_in_cr):
    data = '{"path": "' + file_path_in_cr + '"}'
    log('inside fetch_file_details_by_path : ' + data+ '' ,'debug')
    headers = {"X-Authorization": user_token, 'Content-type': 'application/json', 'Accept': 'text/plain'}

    log("Fetch file details by path: " + str(data), 'debug')
    log("Headers: " + str(
        {"X-Authorization": user_token[0:20] + ".******", 'Content-type': 'application/json', 'Accept': 'text/plain'}),
        'debug')
    log("URL: " + cr_url + '/v2/repository/workspaces/public/files/bypath', 'debug')

    response = requests.post(cr_url + '/v2/repository/workspaces/public/files/bypath', data=data, headers=headers)

    # Checking if response status is 200
    if response.status_code == 200:
        json_obj = response.json()

        # print("File ID (" + str(json_obj.get('id')) + ") fetched from control room for file path " + file_path_in_cr)
        log("File ID (" + str(json_obj.get('id')) + ") fetched from control room for file path " + file_path_in_cr,
            'info')

        return json_obj
    else:
        error_json_obj = response.json()

        # print(str(error_json_obj))
        log(str(error_json_obj), 'info')

        # Returning error json object
        return error_json_obj


# Export bot package from A360 Control room
def export_bot_package(args):
    try:
        cr_url = args.get('cr_url')
        user_token = args.get('user_token')
        bot_ids = args.get('bot_ids')
        data_folder_path = args.get('folder_path')
        print("hello" + data_folder_path)
        # while '\\\\' in data_folder_path:
        #     data_folder_path.replace("\\\\", "\\").replace('/', '\\')

        export_bot_name = "Export." + dt.datetime.today().strftime('%Y%m%d_%H%M%S') + ".py.utility.zip"

        if str(cr_url).endswith("/"):
            cr_url = cr_url[:-1]

        # Get token status
        tok_status = token_status(cr_url, user_token)

        # Checking token status
        if type(tok_status) == str and tok_status == 'false':
            # Returning error
            result = json.dumps({'code': 'user.token.invalid', 'details': '',
                                 'message': 'Given user token is invalid. Please generate a new one.'})
            log(str(result), 'info')
            return result
        elif type(tok_status) == dict:
            # Returning error
            result = json.dumps(tok_status)
            log(str(result), 'info')
            return result

        # Export bot from control room
        request_id = export_bot(cr_url, user_token, export_bot_name, bot_ids)

        # Checking if bot export has started
        if type(request_id) == str and request_id != '':
            download_id = 'wait'

            # Waiting for bot export to be completed
            while type(download_id) == str and download_id == 'wait':
                # Get bot export status
                download_id = bot_export_status(cr_url, request_id, user_token)

            # Checking if bot export is completed
            if type(download_id) == str and download_id != 'wait' and download_id != '':
                # Downloading file
                download_obj = download_file(cr_url, download_id, user_token,
                                             os.path.join(data_folder_path, export_bot_name))

                # Checking if download is success
                if type(download_obj) == str and download_obj == 'ok':
                    # Returning success
                    log(str(export_bot_name), 'error')
                    return export_bot_name
                else:
                    # Returning error
                    result = json.dumps(download_obj)
                    log(str(result), 'error')
                    return result
            else:
                # Returning error
                result = json.dumps(download_id)
                log(str(result), 'error')
                return result
        else:
            # Returning error
            result = json.dumps(request_id)
            log(str(result), 'error')
            return result
    except Exception as err:
        tb_str = traceback.format_exc()
        result = json.dumps({'code': 'python.exception', 'details': '', 'message': str(err), 'traceback' : tb_str})
        log(str(result), 'error')
        return result


# Export and import bots in A360 Control room
def find_bots(cr_url, username, password, api_key, file_paths_in_cr, folder_path, report):
    log('Check and create folder path (' + folder_path + ') if not available', 'debug')
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    user_token = generate_token(cr_url, username, password, api_key)
    log('Got user token : '+ user_token,'debug')
    global lineNumber
    if type(user_token) == str:
        log('type(user_token) ', 'debug')
        bot_ids = []

        for file_path in file_paths_in_cr:
            log('inside file_paths_in_cr loop ' + file_path , 'debug')
            # while '\\\\' in file_path:
            #     # log('inside while', 'debug')
            #     file_path.replace("\\\\", "\\").replace('/', '\\')
            log('before getting file details variable', 'debug')
            file_details = fetch_file_details_by_path(cr_url, user_token, file_path.replace("\\", "\\\\"))
            log('file_details ' + str(file_details), 'debug')
            if 'id' in file_details:
                if file_details['type'] == 'application/vnd.aa.directory':
                    log('Found folder "' + '" with id: ' + str(file_details['id']), 'debug')

                    res_obj = fetch_all_bot_ids_in_folder(cr_url, user_token, file_details['id'])
                    if type(res_obj) == list:
                        if len(res_obj) > 0:
                            bot_ids.extend(res_obj)
                elif file_details['type'] == 'application/vnd.aa.taskbot':
                    bot_ids.append(file_details['id'])

        log("Bot IDs to be exported: " + str(bot_ids), 'debug')

        # file_name = 'Export.20220816_154744.py.utility.zip'
        file_name = export_bot_package(
            {'cr_url': cr_url, 'user_token': user_token, 'bot_ids': bot_ids, 'folder_path': folder_path})

        # print('file_name: ' + file_name)

        log('Check and create folder path (' + os.environ['temp'] + '\\Backups\\' + file_name.replace('.zip',
                                                                                                      '') + ') if not available',
            'debug')
        Path(os.environ['temp'] + '\\Backups\\' + file_name.replace('.zip', '')).mkdir(parents=True, exist_ok=True)

        log("Unpacking archive (" + file_name + ").", 'debug')
        shutil.unpack_archive(os.environ['temp'] + '\\Backups\\' + file_name,
                              os.environ['temp'] + '\\Backups\\' + file_name.replace('.zip', ''))

        extracted_folder_path = os.environ['temp'] + '\\Backups\\' + file_name.replace('.zip', '')

        with open(extracted_folder_path + '\\manifest.json', 'r') as f:
            manifest_json = json.load(f)
            f.close()

        # log("Bot files to be edited now: " + str(manifest_json['files']), 'debug')

        local_file_paths = []

        for file in manifest_json['files']:
            if file['contentType'] == 'application/vnd.aa.taskbot':
                local_file_paths.append(extracted_folder_path + '\\' + file['path'])

        log("Bot files to be checked and edited: " + str(local_file_paths), 'debug')

        # print(local_file_paths)

        for local_file_path in local_file_paths:
            lineNumber = 0
            is_doc_updated = False
            with open(local_file_path, mode='r', encoding='utf8') as f:
                bot_json = json.load(f)

                variables = []
                if 'variables' in bot_json:
                    for variable in bot_json['variables']:
                        variables.append(variable['name'])

                if report is True:
                    for node in bot_json['nodes']:
                        get_nodes_uses_deprecated_APIs(node, file_path, local_file_path)
                        is_doc_updated = False

                else:
                    f.close()

            if is_doc_updated:
                log("Writing bot file back: " + local_file_path, 'debug')
                with open(local_file_path, mode='w', encoding='utf8') as f:
                    f.write(json.dumps(bot_json))

        if report is False:
            log("Create archive (" + extracted_folder_path + "_updated.zip).", "debug")
            shutil.make_archive(extracted_folder_path + '_updated', format='zip', root_dir=extracted_folder_path)

            if type(user_token) == str:
                return bot_import(cr_url, user_token, os.environ['temp'] + '\\Backups',
                                  file_name.replace('.zip', '_updated.zip'))
            elif type(user_token) == dict:
                result = json.dumps(user_token)
                log(str(result), 'error')
                return result

        else:
            if reportdata:
                generate_csv_report(reportdata, reportFilePath)
                return str(reportdata)
    elif type(user_token) == dict:
        result = json.dumps(user_token)
        log(str(result), 'error')
        return result


# Function to generate a PDF report
# Function to generate a PDF report as a table
def generate_csv_report(data_, filename):
    log("generate_csv_report started with data  - " + str(data_),'debug')

    # Write the updated data to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_)


def get_nodes_uses_deprecated_APIs(node, folderpath, botfile):
    log('Inside get_nodes_uses_deprecated_APIs', 'debug')
    is_doc_updated = False
    global lineNumber
    postAPIConditions = [
        'v1/authentication',
        'v1/authentication/token',
        'v1/usermanagement'
    ]
    getAPIConditions = [
        'v1/usermanagement'
    ]

    putAPIConditions = [
        'v1/usermanagement'
    ]

    deleteAPIConditions =[
        'v1/usermanagement'
    ]

    APIConditions = []
    if node['packageName'].lower() == 'rest':
        lineNumber += 1
        if node['commandName'].lower() == 'restpost':
            APIConditions = postAPIConditions
        elif node['commandName'].lower() == 'restget':
            APIConditions = getAPIConditions
        elif node['commandName'].lower() == 'restput':
            APIConditions = putAPIConditions
        elif node['commandName'].lower() == 'restdelete':
            APIConditions = deleteAPIConditions
        for attribute in node['attributes']:
            if attribute['name'].lower() == 'uri':
                if 'value' in attribute:
                    if 'expression' in attribute['value']:
                        if any(cond in attribute['value']['expression'] for cond in APIConditions):
                            log('Found ' + attribute['value']['expression'] + ' in ' + node['commandName']+' command URI at ' + str(
                                lineNumber), 'debug')
                            match = re.search(r'Automation Anywhere.*(?=\\[^\\]*$)', botfile)
                            if match:
                                botFolderinCR = match.group(0)
                            new_entry = [botfile.rsplit('\\', 1)[-1], botFolderinCR, lineNumber,
                                         attribute['value']['expression'],
                                         "Enabled" if node['disabled'] == False else "Disabled", node['commandName'],
                                         node['packageName']]
                            reportdata.append(new_entry)
                if 'string' in attribute['value']:
                        if any(cond in attribute['value']['string'] for cond in APIConditions):
                            log('Found ' + attribute['value']['string'] + '  in ' +node['commandName']+ ' command URI at ' + str(
                                lineNumber), 'debug')
                            match = re.search(r'Automation Anywhere.*(?=\\[^\\]*$)', botfile)
                            if match:
                                botFolderinCR = match.group(0)
                            new_entry = [botfile.rsplit('\\', 1)[-1], botFolderinCR, lineNumber,
                                         attribute['value']['string'],
                                         "Enabled" if node['disabled'] == False else "Disabled", node['commandName'],
                                         node['packageName']]
                            reportdata.append(new_entry)
    else:
        lineNumber += 1
    if 'children' in node:
        log('Looping through the children', 'debug')
        for sub_node in node['children']:
            res = get_nodes_uses_deprecated_APIs(sub_node, folderpath, botfile)

    if 'branches' in node:
        log('Looping through the branches', 'debug')
        for sub_node in node['branches']:
            res = get_nodes_uses_deprecated_APIs(sub_node, folderpath, botfile)
    return is_doc_updated


# Get all bot ids in a given folder
def fetch_all_bot_ids_in_folder(cr_url, user_token, folder_id):
    bot_ids = []

    data = '{"filter":{"operator":"or","operands":[{"operator":"eq","value":"application/vnd.aa.taskbot","field":"type"},{"operator":"eq","value":"application/vnd.aa.directory","field":"type"}]},"sort":[{"field":"directory","direction":"desc"},{"field":"name","direction":"asc"}]}'

    headers = {"X-Authorization": user_token, 'Content-type': 'application/json', 'Accept': 'text/plain'}

    log("Fetch all bot ids in folder: " + str(data), 'debug')
    log("Headers: " + str(
        {"X-Authorization": user_token[0:20] + ".******", 'Content-type': 'application/json', 'Accept': 'text/plain'}),
        'debug')
    log("URL: " + cr_url + '/v2/repository/folders/' + str(folder_id) + '/list', 'debug')

    response = requests.post(cr_url + '/v2/repository/folders/' + str(folder_id) + '/list', data=data, headers=headers)

    # Checking if response status is 200
    if response.status_code == 200:
        json_obj = response.json()

        log("File ID (" + str(json_obj.get('id')) + ") fetched from control room for file path.", 'info')

        if 'list' in json_obj:
            for obj in json_obj['list']:
                if obj['type'] == 'application/vnd.aa.directory':
                    res_obj = fetch_all_bot_ids_in_folder(cr_url, user_token, obj['id'])
                    if type(res_obj) == list:
                        if len(res_obj) > 0:
                            bot_ids.extend(res_obj)
                elif obj['type'] == 'application/vnd.aa.taskbot':
                    bot_ids.append(obj['id'])
        return bot_ids
    else:
        error_json_obj = response.json()

        # print(str(error_json_obj))
        log(str(error_json_obj), 'info')

        # Returning error json object
        return error_json_obj


# Update bot code if there are any necessary changes
def find_v1auth_api_usage(data, report):
    global reportFilePath
   # json_str = json.dumps(data)
    data = json.loads(data)
    # return data['cr_password']
    try:
        if 'file_or_folder_paths' not in data or data['file_or_folder_paths'] is None or data[
            'file_or_folder_paths'] == '':
            return json.dumps({'code': 'python.exception', 'details': '',
                               'message': 'Bot file or folder path is missing or empty, please provide them and run the utility again.'})
        elif 'cr_url' not in data:
            return json.dumps({'code': 'python.exception', 'details': '',
                               'message': 'Control room URL is missing, please provide and run the utility again.'})
        elif 'cr_username' not in data:
            return json.dumps({'code': 'python.exception', 'details': '',
                               'message': 'Username is missing, please provide it for authentication.'})
        elif 'cr_password' not in data and 'cr_apikey' not in data:
            return json.dumps({'code': 'python.exception', 'details': '',
                               'message': 'Both password and apikey are missing, please provide either of them for authentication.'})

        if 'report_file_path' in data:
            reportFilePath = data['report_file_path']
        log_file_path = os.environ['temp'] + "\\v1AuthAPIDeprecation_updates_utility_" + dt.datetime.today().strftime(
            '%Y%m%d_%H%M%S') + ".log" if 'log_file_path' not in data or data['log_file_path'] == '' else data[
            'log_file_path']

        log_level = 'info' if 'log_level' not in data else data['log_level']

        if 'cr_password' not in data:
            data['cr_password'] = ''
        if 'cr_apikey' not in data:
            data['cr_apikey'] = ''

        if type(data['file_or_folder_paths']) == str:
            data['file_or_folder_paths'] = data['file_or_folder_paths'].split("|")

        initialize_logger(log_file_path=log_file_path, log_level=log_level)

        result = find_bots(data['cr_url'], data['cr_username'], data['cr_password'], data['cr_apikey'],
                           data['file_or_folder_paths'], os.environ['temp'] + '\\Backups', report)

        if os.path.exists(os.environ['temp'] + '\\Backups'):
            shutil.rmtree(os.environ['temp'] + '\\Backups', ignore_errors=True)
        return str(result)
    except Exception as err:
        tb_str = traceback.format_exc()
        result = json.dumps({'code': 'python.exception', 'details': '', 'message': str(err), 'traceback': tb_str})
        log(str(result), 'info')
        return str(result)


def generateV1AuthAPIUsageReport(inputJson):
    return find_v1auth_api_usage(inputJson, True)



# Uncomment below lines to test
# update_bot_code({"file_or_folder_paths": 'Automation Anywhere\\Bots\\Test\\EPAM Migration Test', "cr_url": 'CONTROLROOM_URL', "cr_username": 'CONTROLROOM_USERNAME', "cr_password": 'CONTROLROOM_USER_PASSWORD', "cr_apikey": 'CONTROLROOM_USER_APIKEY', "log_file_path": "LOG_FILE_PATH", "log_level": "debug", "update_expr_in_comment": "True", "keep_db_schema_in_cache": "False", "update_error_desc_in_finally_with_given_var": "Exception_description"})
# print(generateV1AuthAPIUsageReport({"file_or_folder_paths": 'Automation Anywhere\\Bots\\TestFolder',
#                                     "cr_url": 'https://aa-pet-us-17.my.automationanywhere.digital',
#                                     "cr_username": 'sikha_creator',
#                                     "cr_password": 'password',
#                                     "cr_apikey": '',
#                                     "log_file_path": r"C:\Users\Sikha.P\OneDrive - Automation Anywhere Software Private Limited\AA_SIKHA\AA_SIKHA\CASES\v1AuthAPI-Report\Debug\log.log",
#                                     "log_level": "debug",
#                                     "report_file_path": r"C:\Users\Sikha.P\OneDrive - Automation Anywhere Software Private Limited\AA_SIKHA\AA_SIKHA\CASES\v1AuthAPI-Report\Debug\report.csv"
#                                     }))
