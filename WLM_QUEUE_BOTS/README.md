# Parallel Bot Execution Pattern — AA 360

**FeederBot → QueueBot → CheckerBot → ReporterBot**

Orchestrates parallel Automation Anywhere 360 bot executions using a Work Item queue. Progress tracking is handled via a shared CSV file (`populated.csv`) — no Excel dependency.

---

## Bot Architecture

| Bot       | Name                  | Responsibility                                                                                                                          |
| --------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Bot-A** | FeederBot             | Reads source CSV, pushes work items into the queue, calculates estimated completion time, and schedules Bot-C via the Control Room API. |
| **Bot-B** | QueueBot              | Runs in parallel; processes one work item per instance and appends `workitem_id, status` to `populated.csv`.                            |
| **Bot-C** | CheckerBot            | Triggered at scheduled time; removes the header row, then polls the CSV row count until it matches `nCount`, then triggers Bot-D.       |
| **Bot-D** | ReporterBot (tesBotD) | Runs once Bot-C confirms all items are processed and generates the final report.                                                        |

---

## End-to-End Flow

```
Bot-A (FeederBot)
  ├── Reads source CSV
  ├── Pushes all rows as Work Items → Queue
  └── Schedules Bot-C via Control Room API (passes total row count as nCount)

Bot-B (QueueBot) × N parallel instances
  ├── Claims one Work Item per instance
  ├── Executes business logic
  └── Appends workitem_id,Successful or workitem_id,False to populated.csv

Bot-C (CheckerBot)  ← triggered at scheduled time
  ├── Removes header row from populated.csv
  ├── Polls CSV row count every 6 seconds
  └── When count == nCount → triggers Bot-D

Bot-D (ReporterBot)
  └── Generates final report
```

---

## Prerequisites

### AA 360 Packages Required

| Package                   | Version  | Used By             |
| ------------------------- | -------- | ------------------- |
| CSV/TXT Package           | ≥ 2.0    | Bot-A, Bot-B, Bot-C |
| Workload (Work Items)     | ≥ 3.0    | Bot-A               |
| REST Web Services         | ≥ 2.0    | Bot-A               |
| JSON Utilities            | ≥ 2.0    | Bot-A               |
| Data Table Package        | ≥ 2.0    | Bot-C               |
| DateTime Package          | ≥ 2.0    | Bot-A               |
| Logging Package           | ≥ 2.0    | Bot-B               |
| Boolean Package           | ≥ 1.0    | Bot-B, Bot-C        |
| Error Handler (Try/Catch) | Built-in | Bot-B               |
| Task Bot Runner           | Built-in | Bot-C               |

### Shared CSV — `populated.csv`

Must exist on a shared network path accessible by all runner machines before the first run.

- Bot-B appends one row per work item (success or failure)
- Bot-C removes the header row, then compares row count directly against `nCount`

> All Bot-B runner machines must map the shared network drive with the **same drive letter** and have **write permission**.

---

## Bot-A — FeederBot

Entry point. Configure these variables before each run:

| Variable         | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `$csvPath$`      | Full path to the source CSV                             |
| `$nTimeTaken$`   | Estimated processing time per item (minutes)            |
| `$nDeviceCount$` | Number of licensed runner devices assigned to the queue |
| `$cr-url$`       | Base HTTPS URL of your Control Room                     |

**Scheduling formula:**

```
EstimatedTime = (rowCount × nTimeTaken) ÷ nDeviceCount
```

Calls `POST /v1/authentication` to get a Bearer token, then `POST /v3/automations/deploy` to schedule Bot-C.

> **Note:** Timezone in the API payload must be `Asia/Calcutta` (AA 360 legacy identifier).

---

## Bot-B — QueueBot

Runs as multiple parallel instances. Each instance:

1. Claims one Work Item from the queue
2. Executes business logic
3. Appends `workitem_id,Successful` or `workitem_id,False` to `populated.csv`
4. Re-throws errors so the Control Room correctly marks the Work Item as `Failed`

> **Important:** The `Throw` at the end of the Catch block is critical — without it, failed items are marked as Successful in the Control Room.

---

## Bot-C — CheckerBot

Triggered by the schedule created by Bot-A. Receives `nCount` (total work items) as an input variable.

- Removes the header row from `populated.csv`
- Compares row count directly against `nCount` (no +1 adjustment needed)
- Waits 6 seconds between each poll
- Triggers Bot-D once counts match

---

## Bot-D — ReporterBot (tesBotD)

Triggered by Bot-C. Replace the default message box with production logic such as:

- Summarise `populated.csv` — total, success count, failure count
- Send an email with the summary and CSV as an attachment
- Call a REST API to update a dashboard or ticketing system
- Archive `populated.csv` with a timestamp and create a fresh file for the next run

---

## APIs Used (Bot-A)

| Endpoint                 | Method | Purpose                                 |
| ------------------------ | ------ | --------------------------------------- |
| `/v1/authentication`     | POST   | Obtain Bearer token                     |
| `/v3/automations/deploy` | POST   | Schedule Bot-C as a one-time automation |

---

## Troubleshooting

| Symptom                                     | Likely Cause                                               | Fix                                                                                                                 |
| ------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Bot-C never triggers                        | Schedule API failed or `$time$` is in the past             | Check API response in `$dic$`. Ensure `$time$` is a future timestamp.                                               |
| Bot-C loops indefinitely                    | A Bot-B instance crashed before writing to `populated.csv` | Verify the Catch block in Bot-B writes the failure row before `Throw`. Check network drive access from all runners. |
| Work Items show Successful despite failures | Missing `Throw` in Bot-B outer Catch                       | Ensure `Error handler: Throw AllErrors` is active at the end of Bot-B's Catch block.                                |
| `populated.csv` grows across runs           | Bot-D doesn't archive the CSV                              | Add archive step to Bot-D: rename with timestamp, create fresh `populated.csv`.                                     |
| `nCount` is 0 or empty                      | Source CSV was empty or `$rows$` not set                   | Add a row count check in Bot-A — if 0 rows, log error and stop.                                                     |
| REST 401 Unauthorized                       | Wrong credentials or locked service account                | Verify credentials. Reset the account in the Control Room web UI.                                                   |
| REST 403 Forbidden                          | Service account lacks schedule permission                  | Ask an AA admin to grant `Schedule Create` permission.                                                              |
