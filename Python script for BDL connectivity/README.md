# Hive Connection Script (test_hive.py)

This script connects to a Kerberos-authenticated Hive database using Python
and performs basic operations: connect, select, insert, update, delete.

## 1. Prerequisites

### a) Python

Python 3.8 or above must be installed on the machine.
Check with:

```
python --version
```

## Quick Reference: Libraries to Install

Run this in Command Prompt before running the script:

```
pip install pyhive thrift thrift_sasl gssapi
```

If `gssapi` or `thrift_sasl` fail to install on Windows (common — they
sometimes need C++ build tools), install this instead:

```
pip install pure-sasl
```

What each library does:
| Library | Purpose |
|---|---|
| `pyhive` | Main library used to connect to and query Hive |
| `thrift` | Underlying protocol pyhive uses to talk to Hive |
| `thrift_sasl` | Handles the authentication layer (needed for Kerberos) |
| `gssapi` | Provides Kerberos support for Python |
| `pure-sasl` | Fallback if `gssapi`/`thrift_sasl` fail to install |

Note: **MIT Kerberos for Windows** is also required, but it is installed
separately (not via `pip`) — see Prerequisites below.

---

### b) MIT Kerberos for Windows

Required so the `kinit` command works on this machine.

- Download from: https://web.mit.edu/kerberos/dist/
- Install it, then confirm it's available by running:

```
where kinit
```

If this shows a path, it's installed correctly.

### c) A valid krb5.conf file

This tells Kerberos which realm/KDC (Key Distribution Center) to talk to.
Ask your DB/security team for this file, or the KDC hostname + realm name
so it can be configured. On Windows this usually goes in:

```
C:\ProgramData\MIT\Kerberos5\krb5.ini
```

### d) Kerberos credentials

Either of the following, provided by your DB/security team:

- A **keytab file** (recommended for automation/bots), or
- A **username + password** for interactive authentication

### e) Network access

Confirm the machine running this script can reach:

- The Hive server host and port (commonly `10000`)
- The Kerberos KDC (commonly port `88`)

---

## 2. Required Python Libraries

Install these before running the script:

```
pip install pyhive thrift thrift_sasl gssapi
```

If `gssapi` or `thrift_sasl` fail to install on Windows, try:

```
pip install pure-sasl
```

---

## 3. File Setup

1. Save the script as `test_hive.py` on your machine
   (e.g. in a simple folder like `C:\Scripts\test_hive.py` — avoid folders
   with spaces in the path if possible, such as certain OneDrive folders).

2. Open `test_hive.py` in Notepad (or any text editor).

3. Edit only the configuration section inside the `main()` function
   (clearly marked in the script):

```python
PRINCIPAL = "<your_username>@<YOUR_REALM>"     # e.g. "jdoe@EXAMPLE.COM"
KEYTAB_PATH = None                              # e.g. r"C:\path\to\your.keytab"
PASSWORD = None                                 # e.g. "your_password"

HOST = "<YOUR_HIVE_HOST_IP>"
PORT = <YOUR_HIVE_PORT>                         # numeric, e.g. 10000 (no quotes)
DATABASE = "<YOUR_DATABASE_NAME>"
KERBEROS_SERVICE_NAME = "<YOUR_KERBEROS_SERVICE_NAME>"   # usually "hive"

TABLE_NAME = "<YOUR_TABLE_NAME>"                # an existing table name
```

**Important rules:**

- Fill in **either** `KEYTAB_PATH` **or** `PASSWORD` — never both, and never
  leave both empty.
- `PORT` must be a plain number, not in quotes (e.g. `PORT = 10000`).
- If the table already exists, do **not** uncomment `create_table(...)`.
  Only uncomment it if you specifically need to create a new table.

---

## 4. How to Run

Open Command Prompt, go to the folder where the script is saved:

```
cd C:\Scripts
```

Then run:

```
python test_hive.py
```

---

## 5. What the Script Does (in order)

1. **Kerberos login (`kinit`)** — obtains a Kerberos ticket using your
   keytab or password.
2. **Connect** — opens a connection to Hive using that ticket.
3. **Select** — runs automatically by default, to confirm the connection
   and table access work.
4. **Insert / Update / Delete** — available as functions, but commented
   out by default. Uncomment and edit the example lines in `main()` to use
   them:

```python
# insert_row(cursor, TABLE_NAME, [7, "Grace", 58000])
# update_row(cursor, TABLE_NAME, set_clause="salary = 60000", where_clause="id = 7")
# delete_row(cursor, TABLE_NAME, where_clause="id = 7")
```

5. **Close connection** — happens automatically at the end, even if an
   error occurs partway through.

---

## 6. Reading the Output

- `[OK]` messages mean that step succeeded.
- `[ERROR]` messages mean that step failed — read the message for details.
- `[FATAL]` means the script stopped completely due to an error.

Common issues:
| Message contains | Likely cause |
|---|---|
| `kinit not found` | MIT Kerberos isn't installed or not in PATH |
| `kinit failed` | Wrong principal, password, or keytab path |
| `Connection to Hive failed` | Wrong host/port, firewall blocking access, or Kerberos service name mismatch |
| `create_table failed` / `insert/update/delete failed` | Check table name, column names, or SQL syntax in the values passed |

---

## 7. Notes

- `UPDATE` and `DELETE` only work on Hive tables that are **transactional**
  (created with `TBLPROPERTIES ('transactional'='true')` and bucketed).
  If your existing table isn't set up this way, those operations may fail —
  check with your DB team.
- Never share your password or keytab file with anyone outside your
  organization. Keep credentials only in your local copy of the script.
