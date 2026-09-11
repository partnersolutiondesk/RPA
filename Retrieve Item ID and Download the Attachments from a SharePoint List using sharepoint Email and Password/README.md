# Retrieve Item ID and Download the Attachments from a SharePoint List using sharepoint Email and Password

A self-contained Python script that queries a SharePoint Online list via the
Microsoft Graph API, applies filter conditions, and returns the matching
item IDs (or full rows) — with an optional step to download attachments for
the matched items using SharePoint username/password auth.

Works standalone or embedded inside **Automation Anywhere 360 (AA360)** as a
Python script action.

---

## 1. Requirements

- **Python 3.9+** (3.10 / 3.11 recommended)
- Three Python packages: `requests`, `msal`, `Office365-REST-Python-Client`

---

## 2. Setup

**Step 1 — Check your Python version**

```bash
python --version
```

If it's below 3.9, install a newer version from [python.org](https://www.python.org/downloads/).

**Step 2 — (Recommended) create an isolated environment**

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**Step 3 — Install the required packages**

```bash
pip install requests msal Office365-REST-Python-Client
```

**Step 4 — Verify the install**

```bash
python -c "import requests, msal, office365; print('All libraries import OK')"
```

If that prints `All libraries import OK` with no errors, setup is complete.

---

## 3. Quick Start

```bash
python sharepoint_attachment_downloader.py
```

By default the script uses the `demo_input` list inside the `__main__`
block at the bottom of the file. Edit that list to put in your own
credentials and filter.

Quiet modes (good for piping):

```bash
python sharepoint_attachment_downloader.py --ids-only   # one item ID per line, nothing else
python sharepoint_attachment_downloader.py --ids-text   # same, via the text-wrapper helper
```

---

## 4. How to Call It

Every entry point takes one argument: a flat Python list. The first 13
slots are configuration (indices 0–12); slot 13 onward are filters.

| Index | Name                  | Type   | Required?          | Example / fill with                                                                      |
| ----- | --------------------- | ------ | ------------------ | ---------------------------------------------------------------------------------------- |
| 0     | `TENANT_ID`           | String | Yes                | Azure AD Tenant ID (GUID). Azure Portal → App Registrations → Endpoints                  |
| 1     | `CLIENT_ID`           | String | Yes                | Azure AD Application (Client) ID (GUID)                                                  |
| 2     | `CLIENT_SECRET`       | String | Yes                | Client Secret **value** (not the Secret ID) — Certificates & secrets → New client secret |
| 3     | `PROXY_URL`           | String | No                 | `""` or `None` = no proxy, or `"http://proxy.corp.com:8080"`                             |
| 4     | `SHAREPOINT_HOST`     | String | Yes                | e.g. `"contoso.sharepoint.com"` (no `https://`)                                          |
| 5     | `SITE_NAME`           | String | Yes                | e.g. `"mysite"` or `"teams/salesteam"`                                                   |
| 6     | `LIST_NAME`           | String | Yes                | Exact list display name, e.g. `"SupportTickets"`                                         |
| 7     | `PAGE_SIZE`           | Number | No                 | `0`/`None` = 200 (default)                                                               |
| 8     | `REQUEST_TIMEOUT`     | Number | No                 | `0`/`None` = 30 seconds                                                                  |
| 9     | `MAX_RETRIES`         | Number | No                 | `0`/`None` = 3 retries                                                                   |
| 10    | `SHAREPOINT_EMAIL`    | String | Only for downloads | Login used for attachment download. Leave `None` to skip                                 |
| 11    | `SHAREPOINT_PASSWORD` | String | Only for downloads | Password for the account above                                                           |
| 12    | `OUTPUT_FOLDER_PATH`  | String | Only for downloads | Local folder to save attachments, e.g. `r"C:\Output"`                                    |
| 13+   | Filter(s)             | Mixed  | No                 | See section 5                                                                            |

If a required value is missing, the script prints a clear message listing
the exact index to fill — never a cryptic `KeyError`.

> **Note:** attachments are only downloaded when `args[10]`, `args[11]`,
> **and** `args[12]` are all provided. Leave them `None` if you just want
> item IDs or rows — steps 1–5 of the run complete either way.

---

## 5. Filter Conditions — 4 ways to write them

Every slot from **index 13 onward** is a filter condition. All conditions
across all slots are combined with **AND** — there's no OR support. Each
slot must be a complete, self-contained condition; a bare value with no
column attached (e.g. just `"Active"`) is silently ignored.

You can freely mix any of the 4 methods below across different slots.

### Method 1 — JSON dict string (RECOMMENDED for multiple equality filters)

```python
args[13] = '{ "Country": "India", "Status": "Pending" }'
```

More examples:

```python
{ "Country": "India", "name": "dev", "City": "Mumbai" }
{ "amount": 12500, "is_vip": true }
{ "Created": "2024-05-01" }
```

- Text values → wrap in double quotes `"like this"`
- Numbers / booleans → no quotes: `12500`, `3.14`, `true`, `false`
- No trailing comma after the last pair
- Unlimited conditions — all AND-ed together
- Column names matched case-insensitively against the list's real names

### Method 2 — Plain string with an operator symbol (one per slot)

| String you write     | Meaning               | Works on             |
| -------------------- | --------------------- | -------------------- |
| `"Status=Closed"`    | equals                | Text / Number / Date |
| `"Status==Closed"`   | equals (same)         | Text / Number / Date |
| `"Status!=Open"`     | not equals            | Text / Number / Date |
| `"amount>100"`       | greater than          | Number / Date        |
| `"amount>=100"`      | greater than or equal | Number / Date        |
| `"amount<100"`       | less than             | Number / Date        |
| `"amount<=100"`      | less than or equal    | Number / Date        |
| `"Title~=invoice"`   | contains              | Text only            |
| `"Name^=Ticket_"`    | starts with           | Text only            |
| `"Email$=@corp.com"` | ends with             | Text only            |

Example — one condition per slot:

```python
[
    # ... first 13 config slots ...
    "Status=Pending",          # args[13]
    "amount>=12500",           # args[14]
    "Country=India",           # args[15]
    "Created>=2024-01-01",     # args[16]
]
```

### Method 3 — List / tuple

```python
["Status", "eq", "Closed"]
["amount", "ge", 12500]
["Country", "India"]          # 2-element short form: defaults to "eq"
```

Operator codes: `eq` `ne` `gt` `ge` `lt` `le` `contains` `startswith` `endswith`

### Method 4 — Full dict

```python
{"column": "Status", "operator": "eq", "value": "Closed"}
{"column": "amount", "operator": "ge", "value": 12500}
```

Alternative keys also accepted: `Column`/`name`/`field` for column, `op`
for operator, `Value` for value.

### Value types

- **Dates** — pass as `"YYYY-MM-DD"`; auto-converted to OData datetime
  format for comparison operators.
- **Booleans** — `true`/`false`/`1`/`0`/`yes`/`no` are all recognized.
- **Empty/placeholder slots** — `None`, `""`, `"none"`, `"null"`, `"n/a"`,
  `"default"`, `"-"` are silently skipped, so unused filter slots can be
  left blank safely.

---

## 6. AA360 Entry Points

| Function                                      | Input           | Output                                                              | When to use                                      |
| --------------------------------------------- | --------------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| `get_sharepoint_item_ids_aa360(args)`         | mixed-type list | `["5", "7", "12"]` — flat list of item IDs                          | You just need IDs to loop through in AA360       |
| `get_sharepoint_items_aa360(args)`            | mixed-type list | List of dicts — one per row, all discovered columns                 | You need actual column values                    |
| `get_sharepoint_download_summary_aa360(args)` | mixed-type list | `{"downloaded": 4, "items_with_attachments": 2, "failed_items": 0}` | You only care about the download result          |
| `get_sharepoint_ids_only_list(args)`          | mixed-type list | Same as `_aa360` but silent (no console output)                     | Running inside loops, no logs wanted             |
| `get_sharepoint_ids_only_text(args)`          | mixed-type list | `"5\n7\n12"` single string, `""` if 0 matches                       | Write IDs to a file, one per line                |
| `get_sharepoint_download_verbose(args)`       | mixed-type list | Full console output as one string                                   | Dump the whole run log into an AA360 Message box |

`get_sharepoint_item_ids(args)` is the internal verbose version — returns
a full dict `{ item_ids, items, columns, filter, matched, elapsed_s,
downloaded, download_summary }` and prints a formatted table.

---

## 7. Troubleshooting

**"CONFIG ERROR — missing required input values"**
Fill the slots listed (they tell you exactly `args[index] NAME`) with real values:

```python
[
  "tenant-guid",             # 0  TENANT_ID
  "app-guid",                # 1  CLIENT_ID
  "secret-value",            # 2  CLIENT_SECRET
  "",                        # 3  PROXY_URL
  "contoso.sharepoint.com",  # 4  SHAREPOINT_HOST
  "my-site",                 # 5  SITE_NAME
  "MyList",                  # 6  LIST_NAME
  0, 0, 0,                   # 7-9 performance
  "user@domain.com",         # 10 SHAREPOINT_EMAIL (optional)
  "<password>",              # 11 SHAREPOINT_PASSWORD (optional)
  r"C:\Output",              # 12 OUTPUT_FOLDER_PATH (optional)
  '{ "column" : "value" }',  # 13 FILTER
]
```

**Token failed: error codes**

| Hint in output                          | What to check                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------ |
| `Check CLIENT_ID`                       | App registration exists, value copied correctly (not the Object ID)                  |
| `Check CLIENT_SECRET`                   | Secret hasn't expired; use the secret **Value**, not Secret ID                       |
| `Check TENANT_ID`                       | Use the Directory (tenant) ID from the app Overview page                             |
| `Check API permissions + admin consent` | `Sites.Read.All` (or `Sites.Selected`) Application permission, admin consent granted |

**`Cannot reach login.microsoftonline.com`**
You're behind a corporate proxy. Set `args[3]` to `"http://proxy.corp.com:8080"` (your real proxy address).

**`404` — Site not found / List not found**
Check `SITE_NAME` (args[5]) and `LIST_NAME` (args[6]) spelling. The script lists available lists on the site when a list isn't found.

**`400 Bad Request` — filter rejected**
Usually means a filtered column isn't indexed in SharePoint:

1. Open the list → Gear icon → List settings → Indexed columns → Create a new index
2. Select each column you filter on → Create
3. Run the script again

The script also prints the direct link (`.../_layouts/15/listindex.aspx`) automatically when this happens.

**0 items matched**

- Text values are case-sensitive in SharePoint OData filters — check exact spelling/case
- Leave `args[13]+` empty to see all rows first and verify column values

**Attachment download skipped**

- Confirm `args[10]`, `args[11]`, **and** `args[12]` are all filled
- Confirm the SharePoint account has read access to the list and its attachments
- If step 6/6 fails outright with correct credentials, check whether your tenant allows legacy auth (see section 7)

**`ModuleNotFoundError: No module named 'msal'`** (or `requests` / `office365`)
The libraries aren't installed in the environment actually running the script. Run `python -c "import sys; print(sys.executable)"` to confirm you're using the venv you installed into.

**Corporate SSL / MITM proxy certificate errors**

- Install the corporate root CA on the machine running the script, or
- `pip install pip_system_certs` (lets `requests` use the Windows certificate store automatically)

---

## 9. Security & Privacy

- No credentials are hardcoded in the script — every value comes from the caller's input list. Keep `CLIENT_SECRET` and `SHAREPOINT_PASSWORD` in AA360's Credential Vault (or Azure Key Vault), never in a screenshot or email.
- Nothing is sent anywhere except: `login.microsoftonline.com` (Graph token), `graph.microsoft.com` (list data), your SharePoint site (attachment download), and your optional proxy.
- The SharePoint username/password client ID used for attachment download is Microsoft's own public, published client ID for the "SharePoint Online Management Shell" — it's not a secret and doesn't grant access on its own without valid credentials for your tenant.

---

## 10. Files

- `sharepoint_attachment_downloader.py` — the self-contained script; this is all you need
- `README.md` — this file

No other files or config needed beyond the three pip-installed packages.

---

## 11. Example AA360 Call

```python
# Python Script action in AA360.
# result will be the list of matched item IDs: e.g. ["5", "7", "12"]
import sharepoint_attachment_downloader as sp

args = [
    "<TENANT_ID>",             # 0
    "<CLIENT_ID>",             # 1
    "<CLIENT_SECRET>",         # 2   <- use AA360 Credential Vault variable here
    "",                        # 3   PROXY_URL
    "contoso.sharepoint.com",  # 4
    "mysite",                  # 5
    "SupportTickets",          # 6
    0, 0, 0,                   # 7-9 performance
    "<SHAREPOINT_EMAIL>",      # 10  <- use AA360 Credential Vault variable here
    "<SHAREPOINT_PASSWORD>",   # 11  <- use AA360 Credential Vault variable here
    r"C:\Output",              # 12
    '{ "Country": "India", "Status": "Pending", "amount": 12500 }',  # 13 FILTER
]

result = sp.get_sharepoint_item_ids_aa360(args)
```

That's the entire integration.
