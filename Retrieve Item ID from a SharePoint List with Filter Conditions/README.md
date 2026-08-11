# SharePoint List Filter

A single, self-contained Python script that queries a SharePoint Online list using the Microsoft Graph API and returns the matching item IDs (or full rows). Works standalone or embedded inside Automation Anywhere 360 (AA360) as a Python script action.

---

## 1. Requirements

- Python **3.9+** (3.10 / 3.11 recommended)
- Two Python packages:

```bash
pip install requests msal
```

That's it. No other dependencies. No database, no config files.

---

## 2. Quick Start (command-line demo)

```bash
python sharepoint_filter_download.py
```

By default the script uses the `demo_input` list inside the `__main__` block at the bottom of the file. Edit that list to put your own credentials + filter.

Quiet modes (good for piping):

```bash
python sharepoint_filter_download.py --ids-only   # one item ID per line, nothing else
python sharepoint_filter_download.py --ids-text   # same, via the text-wrapper helper
```

---

## 3. How to Call It

Every entry point in the script takes **one argument**: a Python list.
The first 10 slots are configuration (indices 0–9), slot 10+ are filters.
Nothing else needs to be passed.

### Input list layout at a glance

| Index  | Name            | Type       | Example / fill with                                                                                                                            |
| ------ | --------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 0      | TENANT_ID       | String     | Azure AD Tenant ID (GUID). Find it in Azure Portal → App Registrations → Endpoints.                                                            |
| 1      | CLIENT_ID       | String     | Azure AD Application (Client) ID (GUID).                                                                                                       |
| 2      | CLIENT_SECRET   | String     | Azure AD Client Secret **value** (not the Secret ID). Create in App Registrations → your app → Certificates & secrets → **New client secret**. |
| 3      | PROXY_URL       | String     | `""` or `None` = no proxy. Or `"http://proxy.corp.com:8080"`                                                                                   |
| 4      | SHAREPOINT_HOST | String     | e.g. `"contoso.sharepoint.com"`                                                                                                                |
| 5      | SITE_NAME       | String     | SharePoint site name or path, e.g. `"mysite"` or `"teams/salesteam"`                                                                           |
| 6      | LIST_NAME       | String     | List display name exactly as shown in SharePoint, e.g. `"SupportTickets"`                                                                      |
| 7      | PAGE_SIZE       | Number     | `0` or `None` = 200 (recommended).                                                                                                             |
| 8      | REQUEST_TIMEOUT | Number     | `0` or `None` = 30 seconds.                                                                                                                    |
| 9      | MAX_RETRIES     | Number     | `0` or `None` = 3 retries.                                                                                                                     |
| **10** | **FILTER**      | **String** | **This is what you change most of the time. See section 4.**                                                                                   |

> If a required value is missing you get a clear message listing the exact index to fill — never a cryptic `KeyError`.

---

## 4. The Filter (index 10) — 4 ways to write it

### ✅ Method 1 — One JSON dict at index 10 (RECOMMENDED)

Put **all** your conditions into a single JSON dict string at args[10]. Every key = SharePoint column name, value = what it must equal. Everything is `AND` logic, operator defaults to equals.

```python
args[10] = '{ "Country" : "India", "Status" : "Pending" }'
```

More examples:

```json
{ "Country":"India", "name":"dev", "City":"Mumbai" }
{ "amount": 12500, "is_vip": true }
{ "Created": "2024-05-01" }
```

- Text strings → wrap in **double quotes** `"like this"`
- Numbers / booleans → no quotes: `12500`, `3.14`, `true`, `false`
- No trailing comma after the last pair
- As many conditions as you want — unlimited, all AND together
- Column names matched case-insensitively to the SharePoint list's real internal names

---

### Method 2 — Simple string per slot (args 10, 11, 12…)

Use this when you need `>=`, `<`, `contains`, etc.
One filter per list slot.

| String you write     | Meaning               | Works on             |
| -------------------- | --------------------- | -------------------- |
| `"Status=Closed"`    | equals                | Text / Number / Date |
| `"Status==Closed"`   | equals (same)         | Text / Number / Date |
| `"Status!=Open"`     | not equals            | Text / Number / Date |
| `"amount>100"`       | greater than          | Number / Date        |
| `"amount>=100"`      | greater than or equal | Number / Date        |
| `"amount<100"`       | less than             | Number / Date        |
| `"amount<=100"`      | less than or equal    | Number / Date        |
| `"Title~=invoice"`   | **contains**          | Text only            |
| `"Name^=Ticket_"`    | **starts with**       | Text only            |
| `"Email$=@corp.com"` | **ends with**         | Text only            |

Example:

```python
[
    # ... first 10 config slots ...
    "Status=Pending",          # args[10]
    "amount>=12500",           # args[11]
    "Country=India",           # args[12]
    "Created>=2024-01-01",     # args[13]
]
```

---

### Method 3 — List per slot

```python
["Status", "eq", "Closed"]
["amount", "ge", 12500]
["Country", "India"]          # 2-element short form: defaults to eq
```

Operator codes: `eq ne gt ge lt le contains startswith endswith`

---

### Method 4 — Dict per slot

```python
{"column": "Status", "operator": "eq", "value": "Closed"}
{"column": "amount", "operator": "ge", "value": 12500}
```

Alternative keys also accepted: `Column/name/field` for column, `op` for operator, `Value` for value.

---

## 5. AA360 Entry Points

Call any of these from the AA360 **Python Script** action.

| Function                              | Input                        | Output (Return Value)                                                   | When to use                                |
| ------------------------------------- | ---------------------------- | ----------------------------------------------------------------------- | ------------------------------------------ |
| `get_sharepoint_item_ids_aa360(args)` | mixed-type list (see layout) | `["5", "7", "12"]` flat list of item IDs                                | You just need IDs to loop through in AA360 |
| `get_sharepoint_items_aa360(args)`    | mixed-type list              | List of dicts — one dict per row with **every discovered column** value | You need to read actual column values      |
| `get_sharepoint_ids_only_list(args)`  | mixed-type list              | Same as `_aa360` but **silent** (zero console output)                   | Running inside loops, no logs wanted       |
| `get_sharepoint_ids_only_text(args)`  | mixed-type list              | `"5\n7\n12"` single multi-line string, empty string `""` if 0 matches   | Write IDs out to a file one per line       |

> `get_sharepoint_item_ids(args)` is the internal verbose version that returns a full dict `{ item_ids, items, columns, filter, matched, elapsed_s }` and prints a formatted table.

---

## 6. SharePoint App Registration Setup (one-time)

The script needs an Azure AD App Registration with **Application permissions** (not delegated) so it can connect without a user present.

1. Go to **portal.azure.com** → **Microsoft Entra ID** → **App registrations** → **New registration**
   - Name: anything, e.g. `SharePointListReader`
   - Supported account types: single tenant
   - Click **Register**
   - Copy the **Application (client) ID** (= `CLIENT_ID`) and **Directory (tenant) ID** (= `TENANT_ID`)
2. **Certificates & secrets** → **New client secret** → choose expiry → copy the **Value** immediately (= `CLIENT_SECRET`)
3. **API permissions** → **Add a permission** → **Microsoft Graph** → **Application permissions** → add:
   - `Sites.Read.All` → click **Grant admin consent for your tenant**
   - (Or the more restrictive `Sites.Selected` if you scope it to the site.)
4. Wait 2–3 minutes for the admin consent to propagate, then run the script.

---

## 7. Troubleshooting

### "CONFIG ERROR — missing required input values"

Fill the slots listed (they tell you exactly `args[index] NAME`) with real values.
Example skeleton:

```python
[
  "tenant-guid",           # 0 TENANT_ID
  "app-guid",              # 1 CLIENT_ID
  "secret-value",          # 2 CLIENT_SECRET
  "",                      # 3 PROXY_URL
  "contoso.sharepoint.com",# 4 SHAREPOINT_HOST
  "my-site",               # 5 SITE_NAME
  "MyList",                # 6 LIST_NAME
  0, 0, 0,                 # 7-9 performance
  '{ "column" : "value" }' # 10 FILTER
]
```

### Token failed: error codes

| Hint in output                        | What to check                                                                                |
| ------------------------------------- | -------------------------------------------------------------------------------------------- |
| Check CLIENT_ID                       | App registration exists, value copied correctly (not the Object ID)                          |
| Check CLIENT_SECRET                   | Secret hasn't expired; use the secret **Value**, not Secret ID                               |
| Check TENANT_ID                       | Use Directory (tenant) ID from the app overview blade                                        |
| Check API permissions + admin consent | `Sites.Read.All` (or `Sites.Selected`) Application permission **with Admin consent granted** |

### Cannot reach login.microsoftonline.com

You are behind a corporate proxy. Set args[3] `PROXY_URL` to `"http://proxy.corp.com:8080"` (or the real proxy address).

### 404 — Site not found / List not found

Check `SITE_NAME` (args[5]) and `LIST_NAME` (args[6]) spelling. Use the exact display name. The script lists available lists on the site when a list is not found.

### 400 Bad Request — Server rejected the filter

Most common cause: **columns used in the filter are not indexed in SharePoint**.
Fix:

1. Open the SharePoint list in the browser
2. Gear icon → **List settings** → **Indexed columns** → **Create a new index**
3. Select every column you filter on, one at a time → Create
4. Run the script again

Direct link pattern:

```
https://<your-host>/<your-site>/_layouts/15/listindex.aspx
```

The script also prints this link automatically when a 400 happens.

Other possible 400 causes:

- Typo in column name (case-insensitive match is used; but SharePoint internal names sometimes differ from display names, e.g. spaces become `_x0020_`)
- Wrong operator for the column type, e.g. `contains` on a Number column

### 0 items matched

- Text values are **case-sensitive** in SharePoint OData filters — check exact spelling/case.
- Remove the filter (pass empty args[10]) to see all rows first and verify column values.

### Corporate SSL / MITM proxy causes certificate errors

Contact your IT team. Options:

- Install the corporate root CA on the machine that runs the script.
- Or export the corporate CA bundle and run:
  ```bash
  pip install pip_system_certs
  ```
  (lets `requests` use the Windows certificate store automatically.)

---

## 8. Security & Privacy

- **No credentials are hardcoded in the shipped script.** Every value is provided by the caller via the input list. Keep your `CLIENT_SECRET` in AA360's Credential Vault (or Azure KeyVault) — never paste it into a bot screenshot / email.
- MSAL error descriptions (which can echo back identifiers) are redacted before printing: `CLIENT_SECRET`, `CLIENT_ID`, `TENANT_ID`, proxy credentials, and any `Bearer <token>` strings are replaced with `<REDACTED>`/placeholders.
- Nothing is sent anywhere except the two Microsoft endpoints (`login.microsoftonline.com` for the token, `graph.microsoft.com` for the list data) plus the optional proxy you configure.

---

## 9. Files

```
sharepoint_filter_download.py   — Self-contained script (this is all you need)
README.md                       — This file
```

No other files, no config, no external dependencies beyond `requests` and `msal`.

---

## 10. Example Complete AA360 Call

```python
# Python Script action in AA360.
# result variable will be the list of matched item IDs: e.g. ["5", "7", "12"]
import sharepoint_filter_download as sp

args = [
    "<TENANT_ID>",            # 0
    "<CLIENT_ID>",            # 1
    "<CLIENT_SECRET>",        # 2   <- use AA360 Credential Vault variable here
    "",                       # 3   PROXY_URL
    "contoso.sharepoint.com", # 4
    "mysite",                 # 5
    "SupportTickets",         # 6
    0, 0, 0,                  # 7-9 performance
    '{ "Country" : "India", "Status" : "Pending" , "amount" : 12500 }'  # 10 FILTER
]

result = sp.get_sharepoint_item_ids_aa360(args)
```

That is the entire integration.
