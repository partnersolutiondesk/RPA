# Bulk Download of Historical Records from the Control Room

## Overview

This utility allows users to bulk download records available in the **Historical** tab of the Control Room. You can fetch data either:

* Without filters
* With date filters
* With additional filtering criteria

The extracted data is saved to an Excel file for further analysis or reporting.

---

## Requirements

* Python 3.11+
* Required Python packages:

  ```bash
  pip install requests pandas
  ```

---

## Configuration Details

Inside the `main` function, provide the following values:

### `cr_url`

* Specify the exact Control Room URL.
* ⚠️ Do **not** include a trailing `/` at the end of the URL.

### `username`

* Provide the username of your Control Room user.

### `api_key`

* Generate an API key for your Control Room user and pass it here.

### `filepath`

* Provide the file path where the Excel file will be saved.

---

## Payload Configuration

Below are the supported payload formats for different filtering scenarios.

### 1. Payload with Only Date Filters

```python
payload_config = {
    "start_date": "2026-04-06T18:30:00Z",
    "end_date": "2026-04-14T18:29:00Z"
}
```

---

### 2. Payload with Date and Additional Filters

```python
payload_config = {
    "device_name": "ywz",
    "status": "COMPLETED",
    "start_date": "2026-04-07T18:30:00Z",
    "end_date": "2026-04-15T18:29:00Z"
}
```

---

### 3. Payload with All Filters

```python
payload_config = {
    "device_name": "ywz",
    "status": "COMPLETED",
    "automation_name": "abc",
    "priority": "PRIORITY_HIGH",
    "file_name": "abc",
    "user_name": "xyz",
    "start_date": "2026-04-07T18:30:00Z",
    "end_date": "2026-04-15T18:29:00Z"
}
```

---

## Notes

* Ensure all date values are in **ISO 8601 format (UTC)**.
* Filters are optional; include only the fields you need.
* Invalid or unsupported filters may result in API errors.

---

## Output

* The records will be downloaded and saved as an Excel file at the specified `filepath`.

---

## Summary

This tool simplifies the process of retrieving large volumes of historical data from the Control Room by:

* Supporting flexible filtering
* Enabling bulk export
* Saving results in a structured Excel format

---
