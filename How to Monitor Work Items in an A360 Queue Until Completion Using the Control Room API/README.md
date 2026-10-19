# How to Monitor Work Items in an A360 Queue Until Completion Using the Control Room API

A single-file Python script that watches an Automation Anywhere (A360) Workload Management
(WLM) queue until **every work item for the day is finished**, then downloads every item and
builds a formatted **Excel report**.

> Script file: `WLM_MONITOR_BOT.py`
> Entry point for A360: `generateReport(args)`

---

## 1. What it does

1. Logs in to the Control Room.
2. Waits until today's work items actually exist in the queue (queue empty at start is **not** "finished").
3. Polls the queue every ~45 sec until nothing is still **New / Ready to run / Active / On hold**
   (up to `MAX_RUNTIME_HOURS`, default 8 h).
4. Warns in the log every hour while items are still waiting, and lists Active items that look stuck.
5. Downloads **all** work items for the day in pages of 400.
6. Builds a formatted Excel report with a **Summary** sheet and a **Work Items** sheet.
7. E-mailing the report is **not** included — add that on your side (see the comment in `run_report()`).

---

## 2. Requirements

| Item     | Requirement                                              |
| -------- | -------------------------------------------------------- |
| Python   | **3.8 or newer** (3.10+ recommended)                     |
| Packages | `requests`, `openpyxl`                                   |
| Network  | HTTPS access to your Control Room URL                    |
| Access   | A Control Room user (or API key) that can read the queue |

---

## 3. Installation (one-liners)

Check Python:

```powershell
python --version
```

Install the two required packages (pick the matching line):

```powershell
pip install requests openpyxl
```

```powershell
# If you keep several Pythons, install into the exact interpreter:
python -m pip install requests openpyxl
```

```powershell
# Recommended: install into the same Python that A360 uses:
"C:\path\to\python.exe" -m pip install requests openpyxl
```

Verify the install:

```powershell
python -c "import requests, openpyxl; print('OK', requests.__version__, openpyxl.__version__)"
```

---

## 4. Configuration

Everything you normally need to change is in the **`# 1. CONFIGURATION`** block near the top of
the script. No environment variables or extra files are needed.

| Setting                                      | Default      | What it means                                                                   |
| -------------------------------------------- | ------------ | ------------------------------------------------------------------------------- |
| `WAIT_FOR_READY_TO_RUN`                      | `True`       | "Ready to run" items count as still waiting                                     |
| `WAIT_FOR_ON_HOLD`                           | `True`       | Keep polling until "On hold" items are released/finished                        |
| `POLL_INTERVAL_SEC`                          | `45`         | Seconds between queue checks                                                    |
| `MAX_RUNTIME_HOURS`                          | `8`          | Give up after this long → partial report (`0` = no limit)                       |
| `INGESTION_WAIT_MAX_SEC`                     | `900`        | How long to wait for new items to appear at start (0 = don't wait)              |
| `QUIET_PERIOD_SEC`                           | `300`        | "0 waiting" must hold this long before "finished" (guards against wave-loading) |
| `MAX_POLL_OUTAGE_MIN`                        | `30`         | Give up only if the Control Room has failed this long in a row                  |
| `STUCK_ACTIVE_MIN`                           | `60`         | Flag an Active item unchanged this long as "possibly stuck"                     |
| `RATE_LIMIT_CALLS` / `RATE_LIMIT_PERIOD_SEC` | `300` / `60` | Self-imposed API request limit (stay under your CR limit)                       |
| `PAGE_SIZE`                                  | `400`        | Items per page (CR max). Auto-falls back to 200, then 100                       |
| `INCLUDE_ITEM_DATA`                          | `True`       | Add the item's own data fields as `Data_*` columns                              |
| `VERIFY_SSL`                                 | `True`       | Keep `True` (only `False` if a company proxy forces it)                         |

**Do NOT type credentials in the script.** They arrive from A360 through `args[...]`.

---

## 5. How to run

### A. From Automation Anywhere (A360 "Python Script" package)

- **Function to run:** `generateReport`
- **Argument (list):** `[date, cr_url, username, password, api_key, queue_id, output_path]`

| Index     | Name     | Type | Example / Notes                                                                                                   |
| --------- | -------- | ---- | ----------------------------------------------------------------------------------------------------------------- |
| `args[0]` | date     | text | `"2026-10-19"` or `"19/10/2026"` (`""` = today)                                                                   |
| `args[1]` | cr_url   | text | `"https://xxx.my.automationanywhere.digital"`                                                                     |
| `args[2]` | username | text | Control Room user name                                                                                            |
| `args[3]` | password | text | Password (`""` allowed **if** an API key is given)                                                                |
| `args[4]` | api_key  | text | API key (optional when a password is given)                                                                       |
| `args[5]` | queue_id | text | Numeric queue ID, e.g. `"100025611"`                                                                              |
| `args[6]` | output   | text | A folder (`"C:\WLM_Reports"`, created if missing) or a full `.xlsx` name. `""` = `wlm_reports` next to the script |

**Login rule:** password given → password is used (API key = fallback); no password → API key is used;
neither → clear error message.

**Return value:** a one-line text summary — outcome, exit code, report path and totals.

### B. Locally (for testing)

Edit the last line of the script with real values, then run:

```powershell
python WLM_MONITOR_BOT.py.py
```

The default `__main__` block already calls:

```python
print(generateReport(["", "https://YOUR-CR.my.automationanywhere.digital",
                      "USER", "PASSWORD", "", "100025611", r"C:\WLM_Reports"]))
```

---

## 6. Output

Reports and logs are written to the folder from `args[6]` (or `wlm_reports` next to the script):

```text
WLM_Consolidated_Report_<YYYYMMDD_HHMMSS>.xlsx      # full report
WLM_Consolidated_Report_<YYYYMMDD_HHMMSS>_PARTIAL.xlsx  # partial (timeout / stop / error)
WLM_Monitor_<YYYYMMDD>.log                          # run log
```

**Excel sheets:**

- **Summary** — run outcome, queue, window, key numbers (completed / failed / waiting / attention),
  success rate, timing, status breakdown table + bar chart, data warnings.
- **Work Items** — one row per item: `WorkItem_ID`, `Status`, `Category`, `Result`, `Error_Message`,
  `Created_On`, `Updated_On`, `Created_To_Updated_Min`, `Created_By`, `Updated_By`, `Comment`,
  plus optional `Data_*` columns.

> The _Failures & Attention_, _Error Summary_ and _Run Log_ sheets are still in the code but
> **commented out**. Search for `COMMENTED OUT` to switch them back on.

---

## 7. How each status is treated

| Control Room status       | Treatment                                     |
| ------------------------- | --------------------------------------------- |
| New, Ready to run, Active | Still waiting → keep polling                  |
| On hold                   | Still waiting (`WAIT_FOR_ON_HOLD = True`)     |
| Completed                 | Success                                       |
| Failed                    | Failed (bot / processing error)               |
| Data error                | Data error (shown separately)                 |
| Unknown                   | Needs attention (flagged, **not** waited for) |

---

## 8. Exit codes (for Jenkins / schedulers)

| Code  | Meaning                                                           |
| ----- | ----------------------------------------------------------------- |
| `0`   | All items finished, report created                                |
| `1`   | Fatal error (bad login, no permission, queue not found, CR down…) |
| `2`   | Timed out — PARTIAL report created (some items still waiting)     |
| `3`   | No work items appeared in the queue within the wait time          |
| `130` | Stopped by user / scheduler (partial report created)              |

---

## 9. Built-in protections (good to know)

- **Proactive token refresh** — re-login every 15 min before the ~20 min token dies; a 401 also re-logins.
- **Retries with back-off** — for network errors, timeouts, HTTP 408/423/425/429/5xx and non-JSON replies.
- **Confirmed "done"** — "0 pending" must be seen twice **and** hold for a 5 min quiet period with an unchanged total.
- **Ingestion wait** — an empty queue at start is not treated as "finished".
- **Max run-time guard** — partial report (`_PARTIAL`) + alert.
- **Time-based outage tolerance** — 30 min of consecutive failures, not a few bad polls.
- **Final safety re-check** — the whole queue is re-read before declaring "all done".
- **Safe paging** — follows the server's page size; falls back 400 → 200 → 100.
- **Ctrl+C / SIGTERM** — still produces a partial report.
- **Excel-safe output** — no formula injection, no illegal characters, daylight-saving aware times.

---

## 10. Troubleshooting

| Symptom                                    | Fix                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| `ModuleNotFoundError: requests / openpyxl` | `pip install requests openpyxl` into the same Python that runs the script |
| Login fails / no token                     | Check `args[2]` user and `args[3]` password (or `args[4]` API key)        |
| `Queue ID must be numeric`                 | Use the numeric ID, e.g. `"40"`, not the queue name                       |
| `Cannot write to the output folder`        | Check the folder path in `args[6]` and your permissions                   |
| Report saved with a time suffix            | The `.xlsx` was open in Excel; close it and re-run                        |
| Frequent `429 too many requests`           | Lower `RATE_LIMIT_CALLS` / raise `RATE_LIMIT_PERIOD_SEC`                  |
| SSL error behind a proxy                   | Set `VERIFY_SSL = False` only if your company proxy requires it           |

---

## 11. A360 API endpoints used

| Purpose         | Endpoint                                   |
| --------------- | ------------------------------------------ |
| Login           | `/v2/authentication`                       |
| Logout          | `/v1/authentication/logout`                |
| List work items | `/v3/wlm/queues/{queue_id}/workitems/list` |
