# Bot Package Inventory - Retrieve the list of packages and their versions used within the bots located under a specified Control Room folder ID.

## Purpose and when to use it

Control Room reports show which bots exist, but not which package versions each bot is built on. This package reads the content of every taskbot in a folder (including all subfolders, at any depth) and lists, for each bot, every package it declares, the version, and how many commands from that package the bot actually uses.

Use it when you need an inventory of package usage, for example before a package upgrade, or as the input to a vulnerability review.

The package is `BotPackageInventory-1.0.1.jar` and provides the **Trace Packages and Commands** action.

## Before you run it

- An Automation Anywhere Control Room with rights to import a package, and a Windows bot runner for the Bot Agent.
- The package `BotPackageInventory-1.0.1.jar` imported into the Control Room (once per Control Room).
- A Control Room user and API key that can see the folders and bots in scope and can view packages. Bots the account cannot read show up in the report with the status "Content not retrieved".
- Network access to the Control Room URL from the machine that runs the bot (the Bot Agent machine).
- The ID of the folder to scan. The package scans that folder and everything below it.
- An output folder path. The folder is created if it does not exist, and must be writable by the Bot Agent user.
- Close `PackageReport.xlsx` in Excel before the run ends. If it is open, the report is saved under a new name with the time added.

## How to run it

1. Import `BotPackageInventory-1.0.1.jar` as a package in your Control Room.
2. In your bot, add the **Trace Packages and Commands** action (group *BotPackageInventory*).
3. Fill in the five inputs, in this order:

| Position | Value | Example |
|---|---|---|
| 1 | Control Room URL, without a trailing slash | `https://your-control-room.example.com` |
| 2 | User name | `'user-name'` |
| 3 | API key (select a stored credential, or type the value) | `'api-key'` |
| 4 | ID of the folder to scan | `11982` |
| 5 | Output folder | `C:\Reports\PackageScan` |

The action returns a string, so assign it to a variable to read the run summary. When the run ends, the summary is a short JSON document:

```json
{
    "bots_read": 1250,
    "bots_failed": 3,
    "bots_skipped_from_earlier_run": 0,
    "folders_failed": 0,
    "report": "C:\\Reports\\PackageScan\\PackageReport.xlsx"
}
```

Start with a small folder and a low `MAX_WORKERS` the first time, and watch the log. The package has not been validated against a live Control Room, so the first run on a real tenant is the real test.

## How it works

- Authenticate with `POST /v2/authentication`. The token is shared by all worker threads and renewed once when the Control Room answers `401`.
- Load artifact names with `POST /v3/packages/package/list`, 100 packages per page. This maps each package name used inside bots (for example `CsvTxt`) to its artifact name (`bot-command-csvtxt`). If the call fails, the scan still runs and the **Artifact Name** column stays blank.
- Walk the folders with `POST /v2/repository/folders/{id}/list`, 100 items per page, depth first. Only items of type taskbot are read; subfolders are queued. Other file types are ignored.
- Read each bot with `GET /v2/repository/files/{id}/content`. The bots of one page are read at the same time by `MAX_WORKERS` threads (default 5).
- Extract packages. The versions come from the bot's packages list. The command count comes from walking the whole bot and counting every command that names a package, including commands nested inside If, Loop and Try blocks.
- Save every batch to the progress files straight away, then continue with the next page.
- Build the Excel report from the progress files in streaming mode, so memory stays low however many rows there are.

## What the report contains

The file is `PackageReport.xlsx` in the output folder. A sheet that reaches 1,000,000 rows continues on a numbered sheet, for example "Full Report 2". Rows keep the order in which the folders were walked and are not sorted.

### Full Report sheet

One row for each package in each bot.

| Column | Meaning |
|---|---|
| Bot Id, Bot Name, Bot Path | The bot, as in the Control Room repository. |
| Package | The package name used inside the bot, for example `CsvTxt`. |
| Artifact Name | The package's artifact name, for example `bot-command-csvtxt`. Blank if the package was not in the package list. |
| Version | The package version the bot is built with, for example `2.15.1`. Blank if the bot lists no version for it. |
| Commands Used | How many commands of this package the bot uses. A package the bot declares but never calls shows 0. |

### Bot Level Report sheet

One row for each bot, including bots that could not be read.

| Column | Meaning |
|---|---|
| Bot Id, Bot Name, Bot Path | The bot. |
| Status | `OK`, `Content not retrieved` (the call failed, was refused or returned nothing) or `Parse error` (the content was not valid bot JSON). |
| Package Count | Number of packages found in the bot. |
| Total Commands | Commands counted across all its packages. |
| Packages | One line per package, in the form `CsvTxt (2.15.1)`. |

## Stopping, resuming and errors

### Progress files

While the scan runs, three files are kept in the output folder: `PackageReport_progress_full.csv`, `PackageReport_progress_bots.csv` and `PackageReport_progress_meta.json`. They are removed after a scan finishes and the report is written.

### If the run stops

Any error still produces `PackageReport.xlsx` from everything read so far. The log says the scan was not complete and the progress files stay in place. Run the action again with the same inputs. Bots that were read fine are skipped, and the final report contains the earlier and the new bots. The meta file records the folder ID and Control Room URL, and progress from a different folder or Control Room is discarded. Bots that failed in the earlier run are tried again on the resumed run.

If the bot run is killed outright, no report is written at that moment. The progress files remain, and the next run builds the report.

### Retries and timeouts

- Every call waits at most 10 seconds to connect and 120 seconds for an answer.
- Timeouts and the status codes 429, 500, 502, 503 and 504 are retried up to 3 times, waiting 2, 4 and 8 seconds, or the time the Control Room gives in its `Retry-After` header (never more than 60 seconds).
- If a folder listing still fails, that folder is logged as failed and its content is missing from the report. The scan continues with the other folders.
- If a bot still cannot be read, it is recorded with the status "Content not retrieved".
- A bot that failed in a run that finished is not retried automatically. Run the action again to rescan everything.

## Settings you can change

These are constants inside the package. They are fixed for a given build; change them in the source and rebuild the package if a different value is needed.

| Setting | Default | What it does |
|---|---|---|
| `MAX_WORKERS` | 5 | Bots read at the same time. Lower it if the Control Room slows down or returns 429. |
| `PAGE_SIZE` | 100 | Items requested per folder listing page. |
| Connect / read timeout | 10 s / 120 s | Seconds to connect and to wait for an answer. |
| `MAX_RETRIES` | 3 | Attempts for each call. |
| `RETRY_STATUS` | 429, 500, 502, 503, 504 | Answers that are retried. |
| `MAX_ROWS_PER_SHEET` | 1,000,000 | Rows after which a sheet continues on a new sheet. Excel's limit is 1,048,576. |
| `LOG_EVERY` | 500 | A progress line is logged after this many bots. |
