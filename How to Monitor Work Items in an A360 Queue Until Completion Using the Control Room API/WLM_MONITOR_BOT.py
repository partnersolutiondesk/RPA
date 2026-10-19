#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 Automation Anywhere (A360) - WLM Queue Monitor + Excel Report
==============================================================================
What this script does (in plain English):
  1. Logs in to the Control Room.
  2. Waits until today's work items actually exist in the queue.
  3. Polls the queue every ~45 sec (30-60 sec) until nothing is still waiting, i.e. no item
     is New, Ready to run, Active or On hold (up to MAX_RUNTIME_HOURS, default 8).
  4. Warns (in the log) every hour while items are still waiting, and lists Active
     items that have not changed for a long time (possible stuck bot / dead runner).
  5. Downloads every work item for the day in pages of PAGE_SIZE (400): 1300 items =
     400 + 400 + 400 + 100. The last page is recognised from the total the server reports.
  6. Builds a formatted Excel report (Summary and Work Items sheets).
     The Failures & Attention, Error Summary and Run Log sheets are still in the code but
     commented out (search for "COMMENTED OUT" to switch them back on).
  7. Sending the report by e-mail is NOT part of this script - add that on your side
     (see the comment at the end of run_report()).

How each Control Room status is treated:
  New, Ready to run, Active  -> still waiting (keep polling)
  On hold                    -> still waiting (WAIT_FOR_ON_HOLD = True, default)
  Completed                  -> Success
  Failed                     -> Failed (bot / processing error)
  Data error                 -> Data error (bad input data - shown separately)
  Unknown                    -> Needs attention (flagged in report, NOT waited for)

Key protections built in:
  * PROACTIVE TOKEN REFRESH - A360 tokens die after ~20 min. We log in
    again every 15 min BEFORE the token dies. A 401 also triggers a re-login.
  * RETRIES with back-off for network errors, timeouts, HTTP 408 / 423 / 425 / 429 / 5xx and
    non-JSON (maintenance page) replies.
  * "Done" must be confirmed more than once AND stay true for a QUIET PERIOD (default 5 min)
    with an unchanged total - so a producer that loads items in waves cannot cause a false "finished".
  * Wait for queue ingestion (queue empty at start is NOT "finished"; default wait 15 min;
    a failed call is never counted as "empty").
  * Max run-time guard (default 8 h) -> partial report (file name ends _PARTIAL) + alert.
  * Control Room outage tolerance is TIME based (default 30 min), not a few failed polls.
  * Final safety check: before saying "all done" the whole queue is re-read and every
    item's status is checked on our side, so a wrongly spelled status filter can never
    produce a false "all done".
  * Safe paging (follows the real page size the server returns, falls back 400 -> 200 -> 100).
  * Large queues are polled less often so the rate limit is not wasted.
  * Excel-safe output (no formula injection, no illegal characters), times follow daylight saving.
  * Ctrl+C / SIGTERM still produces a partial report - even if it arrives during the download.
  * The report folder is next to this script (not the scheduler's working folder).

Exit codes (useful for Jenkins / schedulers):
  0 = all items finished, report created
  1 = fatal error (bad login, no permission, queue not found, CR down...)
  2 = timed out, PARTIAL report created (some items still waiting)
  3 = no work items appeared in the queue within the wait time
  130 = stopped by user / scheduler (partial report created)

HOW TO CALL IT FROM AUTOMATION ANYWHERE (A360 "Python Script" package):
  Function to run : generateReport
  Argument (list) : [date, cr_url, username, password, api_key, queue_id, output_path]
      args[0] date      text   report date, e.g. "2026-10-19" or "19/10/2026" ("" = today)
      args[1] cr_url    text   Control Room URL, e.g. "https://xxx.my.automationanywhere.digital"
      args[2] username  text   Control Room user name
      args[3] password  text   password   ("" allowed IF an API key is given)
      args[4] api_key   text   API key     (optional when a password is given)
      args[5] queue_id  text   numeric queue ID, e.g. "100025611"
      args[6] output    text   where to save the Excel report + log file. A FOLDER, e.g. "C:\\WLM_Reports"
                               (created if missing), or a full file name ending in .xlsx.
                               "" = a "wlm_reports" folder next to this script.
  Return value    : a text summary (outcome, exit code, report path, totals).
  Login rule      : password given -> password is used (API key is only a fallback);
                    no password     -> API key is used;  neither -> clear error message.

Setup:
  pip install requests openpyxl     (into the Python that A360 uses)
==============================================================================
"""

import getpass
import json
import logging
import os
import random
import re
import signal
import sys
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter



# ==============================================================================
# 1. CONFIGURATION  -  *** THE ONLY PART YOU NEED TO EDIT ***
#    Everything is set right here. No environment variables or other files needed.
# ==============================================================================

# --- A. RUN INPUTS  (filled in automatically from the args list of generateReport) ----------
# Do NOT type credentials here. They arrive from Automation Anywhere through args[...].
CR_URL = ""                         # args[1]  Control Room URL
USERNAME = ""                       # args[2]  user name
PASSWORD = ""                       # args[3]  password (may be empty if API_KEY is given)
API_KEY = ""                        # args[4]  API key (optional when a password is given)
QUEUE_ID = ""                       # args[5]  numeric queue ID

# --- B. WHICH ITEMS BELONG TO THE REPORT? ----------------------------------------------
# By default: everything created TODAY. "Today" and all times in the Excel come from the
# date, time and time zone of THIS computer (so no time zone setting is needed).
# NOTE: if a run crosses midnight, the window still starts at the midnight of the day the run
#       STARTED, so items created after midnight are included in the same report.
LOOKBACK_DAYS = 0                   # 1 = also include yesterday, 2 = last 2 days ... (0 = today only)
WINDOW_START = ""                   # set from args[0] (the date). Empty = today.
                                    #   Items created from the start of that date (computer time) onwards are included.

# Control Room work item statuses (8): New, Ready to run, Active, On hold, Completed,
# Failed, Data error, Unknown.   See the table in the header of this file.
WAIT_FOR_READY_TO_RUN = True        # True  = Ready to run items count as "still waiting" (recommended)
                                    # (set this AND WAIT_FOR_ON_HOLD to False = exactly New + Active)
WAIT_FOR_ON_HOLD = True             # True  = keep polling until On hold items are released/finished
                                    # False = do not wait for On hold items (they are listed in the
                                    #         report and in a warning)

# --- C. POLLING BEHAVIOUR (defaults are good for most cases) --------------------------------
POLL_INTERVAL_SEC = 45              # wait between checks (never below MIN_POLL_INTERVAL_SEC)
MIN_POLL_INTERVAL_SEC = 30          # protects the Control Room from over-polling
MAX_RUNTIME_HOURS = 8               # give up after 8 hours -> partial report (0 = no limit at all)
WARN_EVERY_SEC = 3600               # while items are still waiting: warn every hour (in the log)
INGESTION_WAIT_MAX_SEC = 900        # if NO items from today exist at start, wait this long (15 min) for them
                                    # to appear, then stop (0 = do not wait at all, stop immediately)
CONFIRMATIONS = 2                   # "0 pending" must be seen this many times in a row
CONFIRM_GAP_SEC = 20                # seconds between those confirmation checks
QUIET_PERIOD_SEC = 300              # "0 waiting" and an UNCHANGED total must hold this long (5 min) before the
                                    # run is called finished. Protects against a producer that loads items in
                                    # waves. 0 = switch this extra protection off.
MAX_POLL_OUTAGE_MIN = 30            # stop only if the Control Room has been failing for this many minutes in a row
STUCK_ACTIVE_MIN = 60               # an Active item unchanged this long is listed as "possibly stuck" (0 = off)
LARGE_QUEUE_ITEMS = 2000            # above this many items ...
LARGE_QUEUE_POLL_SEC = 120          # ... poll only every 2 minutes (saves API requests)

# --- D. TOKEN, NETWORK, PAGING (rarely changed) --------------------------------------------------
TOKEN_REFRESH_SEC = 900             # log in again after 15 min (A360 tokens die at ~20 min)

# --- D2. API RATE LIMIT (Control Room allows only so many requests) ----------------------
# The script counts every request it sends (logins included) and pauses by itself BEFORE it can
# cross the limit. Your limit is about 350-400 requests; stay clearly below it.
RATE_LIMIT_CALLS = 300              # max requests per period (0 = no self-limit)
RATE_LIMIT_PERIOD_SEC = 60          # the period. If your limit is per hour, use 3600 (and e.g. 300 calls)
RATE_LIMIT_MAX_429_WAIT_SEC = 900   # if the Control Room still answers "429 too many requests", wait up to
                                    #   this long (it obeys the Retry-After time) before giving up
AUTH_PATHS = ["/v2/authentication", "/v1/authentication"]   # login endpoints, tried in order
MULTIPLE_LOGIN = True               # True = re-login without logging out other sessions of the user
                                    #   (if the Control Room rejects this field, the script drops it by itself)
CONNECT_TIMEOUT_SEC = 15
READ_TIMEOUT_SEC = 90
HTTP_MAX_RETRIES = 5                # retries for network errors / 429 / 5xx
BACKOFF_BASE_SEC = 2.0
BACKOFF_MAX_SEC = 60
PAGE_SIZE = 400                     # items per page (the Control Room maximum). If it is refused: 200, then 100
VERIFY_SSL = True                   # keep True (only set False if your company proxy forces it)

# --- E. OUTPUT ------------------------------------------------------------------------------------------
# Reports + logs go to OUTPUT_DIR_OVERRIDE if set (e.g. r"C:\\WLM_Reports" - recommended for A360),
# otherwise next to this script, otherwise into the current working folder.
OUTPUT_DIR_OVERRIDE = ""            # fixed fallback folder (used only when args[6] is empty)
OUTPUT_FILE_NAME = ""               # set from args[6] when it ends in .xlsx (else the name is generated)


def _default_output_dir():
    if OUTPUT_DIR_OVERRIDE:
        return Path(OUTPUT_DIR_OVERRIDE)
    try:
        return Path(__file__).resolve().parent / "wlm_reports"
    except NameError:                       # some hosts run the script without __file__
        return Path.cwd() / "wlm_reports"


OUTPUT_DIR = _default_output_dir()
INCLUDE_ITEM_DATA = True            # add the work item's own data fields as Data_* columns
MAX_DATA_COLUMNS = 25

# ==============================================================================
# 1b. STATUS TABLES  (normally no need to edit)
#     Status spellings are normalised (upper-case, spaces/underscores ignored), so
#     "Ready to run", "READY_TO_RUN" and "ready-to-run" are all the same status.
# ==============================================================================
STATUS_ALIASES = {                  # normalised spelling -> canonical API name
    "NEW": "NEW", "READYTORUN": "READY_TO_RUN", "ACTIVE": "ACTIVE", "ONHOLD": "ON_HOLD",
    "COMPLETED": "COMPLETED", "SUCCESS": "COMPLETED",
    "FAILED": "FAILED", "UNSUCCESSFUL": "FAILED",
    "DATAERROR": "DATA_ERROR", "UNKNOWN": "UNKNOWN",
}
STATUS_LABELS = {                   # how a status is shown in the report
    "NEW": "New", "READY_TO_RUN": "Ready to run", "ACTIVE": "Active", "ON_HOLD": "On hold",
    "COMPLETED": "Completed", "FAILED": "Failed", "DATA_ERROR": "Data error", "UNKNOWN": "Unknown",
}
STATUS_SPELLINGS = {                # spellings tried (in order) for the status filter at start-up
    "NEW": ["NEW", "New"],
    "ACTIVE": ["ACTIVE", "Active"],
    "READY_TO_RUN": ["READY_TO_RUN", "READYTORUN", "READY", "Ready to run", "Ready_to_run"],
    "ON_HOLD": ["ON_HOLD", "ONHOLD", "On hold", "On_hold"],
}
WAITING_STATUSES = (["NEW", "ACTIVE"] + (["READY_TO_RUN"] if WAIT_FOR_READY_TO_RUN else [])
                    + (["ON_HOLD"] if WAIT_FOR_ON_HOLD else []))
MAX_RUNTIME_SEC = MAX_RUNTIME_HOURS * 3600 if MAX_RUNTIME_HOURS else 0   # 0 = no limit


def status_key(value):
    """'Ready to run' / 'READY_TO_RUN' / 'ready-to-run' -> 'READYTORUN'."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def canon_status(value):
    """Canonical status name, e.g. 'Ready to run' -> 'READY_TO_RUN'.
    A spelling we do not know is returned as plain upper-case text."""
    key = status_key(value)
    return STATUS_ALIASES.get(key, str(value or "").strip().upper())


def status_label(value):
    c = canon_status(value)
    return STATUS_LABELS.get(c, c or "(blank)")


# ==============================================================================
# 2. LOGGING  (console + file + in-memory copy of warnings for the Excel report)
# ==============================================================================
log = logging.getLogger("wlm")


class _MemoryHandler(logging.Handler):
    """Keeps WARNING/ERROR lines (used by the 'Run Log' sheet, which is currently commented out)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append((datetime.now(), record.levelname, record.getMessage()))


MEMORY_LOG = _MemoryHandler()


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for h in list(log.handlers):            # generateReport may run many times in one A360 session
        log.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    MEMORY_LOG.records = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logfile = logging.FileHandler(OUTPUT_DIR / f"WLM_Monitor_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    logfile.setFormatter(fmt)
    for h in (console, logfile, MEMORY_LOG):
        log.addHandler(h)


# ==============================================================================
# 3. ERRORS
# ==============================================================================
class FatalError(Exception):
    """Retrying will not help (bad credentials, no permission, wrong queue...)."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class TransientError(Exception):
    """Temporary problem (network, 5xx, 429...) that still failed after all retries."""


# ==============================================================================
# 4. CONTROL ROOM CLIENT  (login, proactive token refresh, retries)
# ==============================================================================
RETRYABLE_CODES = (408, 423, 425)       # besides 429 (handled separately) and every 5xx


class ControlRoomClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.http = requests.Session()
        self.http.verify = VERIFY_SSL
        self.token = None
        self._token_time = 0.0       # monotonic clock time of last login
        self.login_count = 0
        self.request_count = 0       # every HTTP request sent (for the report)
        self.pause_count = 0         # times we paused ourselves to stay under the rate limit
        self.rate_429_count = 0      # times the Control Room said 429
        self._sent = deque()         # send times of the most recent requests
        self._rate_calls = RATE_LIMIT_CALLS
        self._last_pause_log = 0.0
        self.use_multiple_login = MULTIPLE_LOGIN

    # ---- token handling ------------------------------------------------------
    def _token_is_fresh(self):
        return self.token is not None and (time.monotonic() - self._token_time) < TOKEN_REFRESH_SEC

    def _ensure_token(self):
        """Called before EVERY request. Re-login when the token is older than 15 min."""
        if not self._token_is_fresh():
            if self.token is not None:
                age = int(time.monotonic() - self._token_time)
                log.info("Token is %ss old (limit %ss) - refreshing proactively.", age, TOKEN_REFRESH_SEC)
            self.login()

    def _login_on_paths(self, body):
        """Try the login path(s) in order. Some Control Rooms only expose /v2."""
        data, last_404 = None, None
        for path in AUTH_PATHS:
            try:
                data = self._request(path, body, authenticated=False)
                break
            except FatalError as exc:
                if exc.status != 404:
                    raise
                last_404 = exc
                log.warning("Login path %s returned 404 - trying next path.", path)
        if data is None:
            raise last_404
        return data

    def _login_once(self, secret_field, secret_value):
        """One login attempt with either {"password": ...} or {"apiKey": ...}."""
        body = {"username": USERNAME, secret_field: secret_value}
        # multipleLogin = True lets this script log in WITHOUT logging out other sessions of the
        # same account (a person, or a second copy of this script).
        if self.use_multiple_login:
            body["multipleLogin"] = True
        try:
            return self._login_on_paths(body)
        except FatalError as exc:
            if self.use_multiple_login and exc.status == 400:
                log.warning("Control Room rejected the multipleLogin field - logging in without it.")
                self.use_multiple_login = False
                body.pop("multipleLogin", None)
                return self._login_on_paths(body)
            raise

    def login(self):
        """Password given -> use the password (API key = fallback only if the password is refused).
        No password      -> use the API key."""
        if PASSWORD:
            try:
                data = self._login_once("password", PASSWORD)
            except FatalError as exc:
                if API_KEY and exc.status in (400, 401):
                    log.warning("Password login was refused - trying the API key instead.")
                    data = self._login_once("apiKey", API_KEY)
                else:
                    raise
        elif API_KEY:
            data = self._login_once("apiKey", API_KEY)
        else:
            raise FatalError("Give a password (args[3]) or an API key (args[4]).")
        token = data.get("token") if isinstance(data, dict) else None
        if not token:
            raise FatalError("Login reply did not contain a token (check user / API key).")
        self.token = token
        self._token_time = time.monotonic()
        self.login_count += 1
        log.info("Logged in to Control Room (login #%d).", self.login_count)

    def logout(self):
        """Best effort - never raises. A360 logout is a GET; a POST is tried if GET is not allowed."""
        if not self.token:
            return
        url = self.base_url + "/v1/authentication/logout"
        headers = {"X-Authorization": self.token}
        try:
            resp = self.http.get(url, headers=headers, timeout=(CONNECT_TIMEOUT_SEC, 20))
            if resp.status_code in (404, 405):
                self.http.post(url, headers=headers, timeout=(CONNECT_TIMEOUT_SEC, 20))
        except Exception:
            pass
        self.token = None

    # ---- generic POST with retries --------------------------------------------
    def post(self, path, payload, retries=None):
        return self._request(path, payload, authenticated=True, retries=retries)

    def _request(self, path, payload, authenticated, retries=None):
        retries = retries or HTTP_MAX_RETRIES
        url = self.base_url + path
        relogin_done = False
        last_problem = "unknown"
        attempt, waited_429, n_429 = 0, 0.0, 0
        while attempt < retries:
            attempt += 1
            if authenticated:
                self._ensure_token()
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["X-Authorization"] = self.token
            self._throttle()
            try:
                resp = self.http.post(url, json=payload, headers=headers,
                                      timeout=(CONNECT_TIMEOUT_SEC, READ_TIMEOUT_SEC))
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_problem = f"network error ({type(exc).__name__})"
                self._backoff(attempt, last_problem, total=retries)
                continue
            except requests.RequestException as exc:
                raise FatalError(f"Request failed: {exc}")

            code = resp.status_code
            if 200 <= code < 300:
                try:
                    return resp.json()
                except ValueError:
                    # e.g. HTML maintenance page returned with HTTP 200
                    last_problem = "reply was not valid JSON"
                    self._backoff(attempt, last_problem, total=retries)
                    continue

            if code == 401:
                if not authenticated:
                    raise FatalError("Login rejected (401): wrong username/password/API key, "
                                     "account locked, or SSO-only user (use an API key).", code)
                if relogin_done:
                    raise FatalError("Still 401 right after a fresh login - check user rights.", code)
                log.warning("HTTP 401 on %s - token expired/invalidated. Logging in again.", path)
                self.token = None          # forces a new login at the top of the loop
                relogin_done = True
                continue

            if code == 429:
                # Too many requests. This does NOT use up a normal retry: we obey Retry-After
                # (or wait 15 s, 30 s, 60 s ... up to 5 min) and also lower our own limit.
                attempt -= 1
                self.rate_429_count += 1
                wait = None
                try:
                    wait = float(resp.headers.get("Retry-After"))
                except (TypeError, ValueError):
                    pass
                if wait is None:
                    wait = min(300.0, 15.0 * (2 ** n_429))
                wait = max(1.0, min(wait, float(RATE_LIMIT_MAX_429_WAIT_SEC)))
                n_429 += 1
                if waited_429 + wait > RATE_LIMIT_MAX_429_WAIT_SEC:
                    raise TransientError(f"{path}: Control Room keeps answering 429 (too many requests) - "
                                         f"gave up after waiting {int(waited_429)} s.")
                now = time.monotonic()
                seen = sum(1 for t in self._sent if now - t < RATE_LIMIT_PERIOD_SEC)   # requests just sent
                base = min(seen, self._rate_calls) if self._rate_calls else seen
                self._rate_calls = max(20, int(base * 0.7))          # learn: stay clearly below what failed
                log.warning("HTTP 429 (too many requests) - pausing %d s and limiting ourselves to %d requests "
                            "per %d s from now on.", int(wait), self._rate_calls, RATE_LIMIT_PERIOD_SEC)
                waited_429 += wait
                self._sent.clear()
                time.sleep(wait)
                continue

            if code >= 500 or code in RETRYABLE_CODES:
                snippet = (resp.text or "")[:150].replace("\n", " ")
                last_problem = f"HTTP {code} {snippet}".strip()
                self._backoff(attempt, last_problem, resp.headers.get("Retry-After"), total=retries)
                continue

            raise FatalError(self._explain(code, resp), code)   # other 4xx: retry is pointless

        raise TransientError(f"{path}: gave up after {retries} attempts ({last_problem}).")

    def _throttle(self):
        """Self-imposed rate limit: never send more than _rate_calls requests in any
        RATE_LIMIT_PERIOD_SEC. If we are at the limit we simply wait until the oldest request
        is old enough - so the Control Room never has to say 429 in the first place."""
        self.request_count += 1
        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0] >= RATE_LIMIT_PERIOD_SEC:
                self._sent.popleft()
            if not self._rate_calls or len(self._sent) < self._rate_calls:
                break
            wait = RATE_LIMIT_PERIOD_SEC - (now - self._sent[0]) + 0.05
            self.pause_count += 1
            if now - self._last_pause_log >= 60:
                self._last_pause_log = now
                log.info("Rate-limit guard: %d requests in the last %d s - pausing %.1f s to stay under the limit.",
                         len(self._sent), RATE_LIMIT_PERIOD_SEC, wait)
            time.sleep(max(0.05, wait))
        self._sent.append(time.monotonic())

    @staticmethod
    def _backoff(attempt, reason, retry_after=None, total=None):
        delay = min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
        delay *= random.uniform(0.7, 1.3)                 # jitter
        if retry_after:                                   # server told us how long to wait
            try:
                delay = max(delay, min(300, float(retry_after)))
            except ValueError:
                pass
        log.warning("%s - retry in %.1fs (attempt %d/%d).", reason, delay, attempt, total or HTTP_MAX_RETRIES)
        time.sleep(delay)

    @staticmethod
    def _explain(code, resp):
        snippet = (resp.text or "")[:300].replace("\n", " ")
        hints = {
            400: "Bad request - check queue id, filter fields or page size.",
            403: "No permission - the user needs a role with WLM/queue access "
                 "(view work items) and must be a participant/owner of the queue.",
            404: "Not found - wrong queue id or wrong Control Room URL.",
        }
        sent = ""
        if "/wlm/" in resp.url:                    # never print login bodies (they hold the password)
            body = resp.request.body
            body = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
            sent = f"\n  We sent: {body[:400]}"
        return (f"HTTP {code} on {resp.request.method} {resp.url}\n"
                f"  {hints.get(code, 'Unexpected response.')}\n  Server said: {snippet}{sent}")


# ==============================================================================
# 5. SMALL HELPERS (time window, filters, counting)
# ==============================================================================
def clean_base_url(url):
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        raise FatalError("CR_URL must start with https://")
    m = re.match(r"^(https?://[^/#?]+)", url)      # drop trailing '/', '/#/...' etc.
    return m.group(1)


def compute_window_start():
    """Returns (utc_string_for_api, utc_datetime, local_tz)."""
    tz = datetime.now().astimezone().tzinfo          # this computer's own time zone
    if WINDOW_START:
        dt = parse_ts(WINDOW_START, tz=timezone.utc)
        if not dt:
            raise FatalError("WINDOW_START is not a valid ISO date-time.")
    else:
        # midnight of today on this computer's clock (naive local time -> UTC; daylight saving safe)
        local_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dt = (local_midnight - timedelta(days=LOOKBACK_DAYS)).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dt, tz


def parse_ts(value, tz=None):
    """Parse many ISO styles. Returns tz-aware datetime or None."""
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    s = re.sub(r"\.(\d+)", lambda m: "." + (m.group(1) + "000000")[:6], s)   # any fraction -> 6 digits
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)                          # +0530 -> +05:30
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz or timezone.utc)
    return dt


class Window:
    """Builds the request bodies. Control Rooms differ in what filter formats they
    accept, so negotiate_api_format() (below) tests a few variants once at start
    and switches these flags to whatever your Control Room accepts."""

    def __init__(self, start_utc_str):
        self.start = start_utc_str       # e.g. "2026-10-15T18:30:00Z"
        self.date_ms = False             # True -> send "...:00.000Z"
        self.use_fields = False          # True -> add "fields": [] to the body
        self.local_date = False          # True -> no date filter sent; we filter locally
        self.status_mode = "or"          # "single" (one status per call) | "or" | "each" | "in" | "local"
        self.raw_names = {}              # status_key -> exact spelling seen in YOUR Control Room data
        self.last_breakdown = None       # last status breakdown logged (log it again only when it changes)
        self.last_waiting_by = None      # {"New": n, "Active": n, ...} from the latest per-status count
        self.last_total = None           # items in the report window (latest full read)
        self.last_counts = None          # {"Completed": n, ...} from the latest full read
        self.page_len = PAGE_SIZE        # page size that the Control Room really accepts (400 -> 200 -> 100)
        self.size_hint = 0               # latest known number of items in the window (for the poll interval)

    def start_dt(self):
        return parse_ts(self.start)

    def api_status(self, name):
        """Use the exact spelling the Control Room itself returned (if we have seen it)."""
        return self.raw_names.get(status_key(name), name)

    def date_value(self):
        return self.start.replace("Z", ".000Z") if self.date_ms else self.start

    def filter(self, statuses=None):
        parts = []
        statuses = [self.api_status(x) for x in statuses] if statuses else statuses
        if statuses and self.status_mode == "single":
            # EXACT shape the Control Room's own screen sends: ONE status, nothing else combined with it
            return {"field": "status", "operator": "eq", "value": statuses[0]}
        if not self.local_date:
            parts.append({"field": "createdOn", "operator": "gte", "value": self.date_value()})
        if statuses and self.status_mode == "in":
            parts.append({"field": "status", "operator": "in", "value": list(statuses)})
        elif statuses and self.status_mode in ("or", "each"):
            ops = [{"field": "status", "operator": "eq", "value": st} for st in statuses]
            parts.append(ops[0] if len(ops) == 1 else {"operator": "or", "operands": ops})
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else {"operator": "and", "operands": parts}

    def body(self, offset, length, statuses=None, sort=None):
        body = {"page": {"offset": offset, "length": length}}
        flt = self.filter(statuses)
        if flt:
            body["filter"] = flt
        if self.use_fields:
            body["fields"] = []
        if sort:
            body["sort"] = sort
        return body


def note_breakdown(window, items):
    """Log the status breakdown, but only when it changed (keeps the log readable)."""
    br = dict(Counter(status_label(i.get("status")) for i in items))
    if br != window.last_breakdown:
        window.last_breakdown = br
        log.info("Status breakdown (report window): %s", br)


def count_items(client, queue_id, window, statuses=None, retries=None):
    """Counts items. Normal path = cheap call with page length 1 (reads totalFilter).
    A reply without totalFilter is an error - it must NEVER be read as '0 pending'.
    Local path (only if the Control Room rejects our filters) = download and count here."""
    if statuses and window.status_mode == "single":
        return count_single(client, queue_id, window, statuses, retries)
    if window.local_date or (statuses and window.status_mode == "local"):
        items, _ = fetch_all_items(client, queue_id, window, quiet=True)
        note_breakdown(window, items)
        if statuses:
            wanted = {canon_status(x) for x in statuses}
            items = [i for i in items if canon_status(i.get("status")) in wanted]
        return len(items)
    if statuses and window.status_mode == "each":      # one small call per status, then add up
        return sum(_count_call(client, queue_id, window, [st], retries) for st in statuses)
    return _count_call(client, queue_id, window, statuses, retries)


def count_single(client, queue_id, window, statuses, retries):
    """One tiny call per status, exactly like the Control Room screen does. A status with 0 items
    costs nothing more. If some exist, we read ONLY those items and keep the ones created inside
    the report window (a single-status filter cannot carry a date, so the date is checked here)."""
    start = window.start_dt()
    total, by = 0, {}
    for st in statuses:
        by[status_label(st)] = 0
        if _count_call(client, queue_id, window, [st], retries) == 0:
            continue
        offset, inside = 0, set()
        while True:
            data = client.post(f"/v3/wlm/queues/{queue_id}/workitems/list",
                               window.body(offset, window.page_len, [st]), retries=retries)
            batch = data.get("list") or []
            if not batch:
                break
            for it in batch:
                if (parse_ts(it.get("createdOn")) or start) >= start:
                    inside.add(it.get("id") if it.get("id") is not None else ("noid", len(inside)))
            offset += len(batch)
            reported = (data.get("page") or {}).get("totalFilter")
            if isinstance(reported, int) and offset >= reported:
                break
        total += len(inside)
        by[status_label(st)] = len(inside)
    window.last_waiting_by = by
    return total


def probe_single_status_filter(client, queue_id, window):
    """Try the request shape your Control Room's own screen uses: ONE status per call
    (filter = {field: status, operator: eq, value: ACTIVE}). For every waiting status we try the
    likely spellings; a 400/500 means 'wrong spelling' -> next one. True if all statuses work."""
    window.status_mode = "single"
    chosen = {}
    for canon in WAITING_STATUSES:
        for spelling in STATUS_SPELLINGS.get(canon, [canon]):
            window.raw_names[status_key(canon)] = spelling
            try:
                _count_call(client, queue_id, window, [canon], 1)
                chosen[canon] = spelling
                break
            except FatalError as exc:
                if exc.status != 400:
                    raise
            except TransientError:
                pass
        else:
            log.warning("No accepted spelling found for status %s (tried %s).", canon,
                        STATUS_SPELLINGS.get(canon, [canon]))
            for c in WAITING_STATUSES:
                window.raw_names.pop(status_key(c), None)
            return False
    log.info("Status filter OK (one status per call). Spellings used: %s", chosen)
    return True


def _count_call(client, queue_id, window, statuses, retries):
    data = client.post(f"/v3/wlm/queues/{queue_id}/workitems/list",
                       window.body(0, 1, statuses), retries=retries)
    page = data.get("page") if isinstance(data, dict) else None
    if not isinstance(page, dict) or not isinstance(page.get("totalFilter"), int):
        raise TransientError("Reply had no page.totalFilter - not trusting it.")
    return page["totalFilter"]


def negotiate_api_format(client, queue_id, window):
    """Runs once. HTTP 400 'json.deserialization.exception' means the Control Room did
    not like our request shape. We try the likely variants and keep the first that works.
    Worst case we filter on our side (slower, but it still works)."""
    last_error = None
    for date_ms, use_fields in ((False, False), (True, False), (False, True), (True, True)):
        window.date_ms, window.use_fields = date_ms, use_fields
        try:
            count_items(client, queue_id, window)
            log.info("Date filter format accepted (milliseconds=%s, fields=%s).", date_ms, use_fields)
            break
        except FatalError as exc:
            if exc.status != 400:
                raise
            last_error = exc
            log.warning("Date filter variant (ms=%s, fields=%s) rejected with 400.", date_ms, use_fields)
        except TransientError as exc:
            last_error = exc
            log.warning("Date filter variant (ms=%s, fields=%s) gave an unusable reply (%s).", date_ms, use_fields, exc)
    else:
        window.date_ms = window.use_fields = False
        window.local_date = True
        log.warning("Control Room rejects the createdOn filter - filtering by date on our side instead.")
        try:
            count_items(client, queue_id, window)
        except FatalError:
            raise last_error          # even an unfiltered call fails: real problem, show the first error

    # Does the Control Room REALLY apply the date filter? (Some return old items anyway.)
    # If a sample page contains items created before our window start, we stop trusting
    # the server-side date filter and filter by date on our side instead.
    if not window.local_date:
        sample = client.post(f"/v3/wlm/queues/{queue_id}/workitems/list", window.body(0, 100), retries=2)
        start = window.start_dt()
        too_old = [i for i in (sample.get("list") or [])
                   if (parse_ts(i.get("createdOn")) or start) < start]
        if too_old:
            window.local_date = True
            log.warning("Control Room ignored the createdOn filter (got %d item(s) older than the window, "
                        "e.g. created %s) - filtering by date on our side.",
                        len(too_old), too_old[0].get("createdOn"))
    if probe_single_status_filter(client, queue_id, window):
        return
    if window.local_date:
        window.status_mode = "local"      # everything is counted on our side
        return

    # Status filter: try styles in order. A 400 (bad format) OR repeated 500 (server
    # bug on that filter) both mean "this style does not work here" -> next style.
    for mode in ("or", "each", "in"):
        window.status_mode = mode
        try:
            count_items(client, queue_id, window, WAITING_STATUSES, retries=2)
            log.info("Status filter style accepted: %s.", mode)
            return
        except FatalError as exc:
            if exc.status != 400:
                raise
            log.warning("Status filter style '%s' rejected with 400.", mode)
        except TransientError as exc:
            log.warning("Status filter style '%s' fails on the server (%s).", mode, exc)
    window.status_mode = "local"
    log.warning("Control Room cannot filter by status - counting pending items on our side "
                "(downloads the queue each poll; use a longer poll interval for big queues).")


def discover_statuses(client, queue_id, window):
    """Runs once at start. Reads the queue and logs which statuses really exist, so you can
    confirm in the log that the script points at the right queue. It also remembers the exact
    spelling your Control Room uses for each status (used in server-side filters)."""
    try:
        items, _ = fetch_all_items(client, queue_id, window, quiet=True)
    except (FatalError, TransientError) as exc:
        log.warning("Could not read the queue for the start-up status check: %s", exc)
        return
    raw = Counter(to_text(i.get("status")).strip() or "(blank)" for i in items)
    ids = sorted(i.get("id") for i in items if isinstance(i.get("id"), int))
    window.size_hint = len(items)
    log.info("Queue %s at start: %d item(s) in the report window%s | statuses: %s",
             queue_id, len(items), f" | item IDs {ids[0]}..{ids[-1]}" if ids else "", dict(raw))
    for name in raw:
        key = status_key(name)
        if key in STATUS_ALIASES:
            window.raw_names.setdefault(key, name)
            window.raw_names.setdefault(status_key(STATUS_ALIASES[key]), name)
        else:
            log.warning("Unrecognised status '%s' in the queue - it will be reported as 'needs attention' "
                        "and NOT waited for. Add it to STATUS_ALIASES if it means something else.", name)
    verify_status_filter(client, queue_id, window)


def verify_status_filter(client, queue_id, window):
    """If the server-side status filter is used, prove it gives the same 'waiting' number as our own
    count. If not (filter ignored / wrong spelling), switch to counting on our side. Two tries,
    because items may change status between the two calls."""
    if window.status_mode == "local":
        return
    # Test 1: a status that cannot exist must give 0. If the count is NOT 0, the Control Room is
    # ignoring the status filter completely (and would report 'waiting' forever).
    try:
        bogus = count_items(client, queue_id, window, ["NO_SUCH_STATUS_XYZ"], retries=1)
    except (FatalError, TransientError):
        bogus = 0                       # server refused the nonsense value = it does check the filter
    if bogus > 0:
        window.status_mode = "local"
        log.warning("Control Room ignores the status filter (a non-existent status matched %d items) - "
                    "counting on our side from now on.", bogus)
        return
    # Test 2: the real 'waiting' number must equal our own count.
    for _ in (1, 2):
        try:
            server = count_items(client, queue_id, window, WAITING_STATUSES, retries=2)
            items, _w = fetch_all_items(client, queue_id, window, quiet=True)
        except (FatalError, TransientError) as exc:
            log.warning("Could not verify the status filter (%s) - counting on our side.", exc)
            window.status_mode = "local"
            return
        mine = sum(1 for i in items if category_of(i.get("status")) == "Waiting")
        if server == mine:
            log.info("Server-side status filter verified (%d waiting).", mine)
            return
        log.warning("Status filter check: Control Room says %d waiting, our count says %d.", server, mine)
    window.status_mode = "local"
    log.warning("Server-side status filter is not reliable here - counting on our side from now on.")


def resolve_queue_id(client):
    """The queue is given by its numeric ID (QUEUE_ID at the top of the script)."""
    qid = str(QUEUE_ID).strip()
    if not qid.isdigit():
        raise FatalError(f"QUEUE_ID must be the numeric queue ID (e.g. \"40\"), got '{QUEUE_ID}'.")
    log.info("Using queue ID %s.", qid)
    return qid


# ==============================================================================
# 6. MONITORING  (ingestion wait -> poll loop -> quiet period -> confirmation)
# ==============================================================================
class RunState:
    """Shared run information (used by the report)."""

    def __init__(self):
        self.started = datetime.now()
        self.finished = None
        self.outcome = "NOT_STARTED"   # COMPLETE / TIMEOUT / NO_ITEMS / INTERRUPTED / ERROR
        self.error_text = ""
        self.poll_count = 0
        self.consecutive_errors = 0
        self.first_error_at = None     # monotonic time when the current streak of failed polls began
        self.max_pending_seen = 0


def safe_call(state, func, *args):
    """Run a count call; on temporary failure return None and count the error.
    The Control Room must be failing for MAX_POLL_OUTAGE_MIN minutes in a row (time based, so a
    maintenance window does not kill a good run) before we give up with a FatalError."""
    try:
        value = func(*args)
        state.consecutive_errors = 0
        state.first_error_at = None
        return value
    except TransientError as exc:
        state.consecutive_errors += 1
        if state.first_error_at is None:
            state.first_error_at = time.monotonic()
        down_min = (time.monotonic() - state.first_error_at) / 60.0
        log.warning("Poll problem (%d in a row, failing for %.1f min, giving up after %d min): %s",
                    state.consecutive_errors, down_min, MAX_POLL_OUTAGE_MIN, exc)
        if down_min >= MAX_POLL_OUTAGE_MIN:
            raise FatalError(f"Control Room has been unavailable for {down_min:.0f} minutes - giving up.")
        return None


def sleep_capped(seconds, deadline):
    time.sleep(max(0.0, min(seconds, deadline - time.monotonic())))


def poll_interval(window):
    """Normal queues: POLL_INTERVAL_SEC. Big queues cost more requests per check, so poll less often."""
    base = max(POLL_INTERVAL_SEC, MIN_POLL_INTERVAL_SEC)
    if LARGE_QUEUE_ITEMS and window.size_hint >= LARGE_QUEUE_ITEMS:
        return max(base, LARGE_QUEUE_POLL_SEC)
    return base


def wait_for_ingestion(client, queue_id, window, state, deadline):
    """Edge case: bot starts before the producer has loaded today's items.
    Zero items must NOT be treated as 'all done'. Only a SUCCESSFUL reply that says 0 counts as
    'empty' - a failed call is never counted (it is retried until the outage limit is reached).
    Returns the number of items found, or 0 if none appeared in time."""
    started = time.monotonic()
    while True:
        total = safe_call(state, count_items, client, queue_id, window)
        if total:
            window.size_hint = total
            log.info("Queue has %d work item(s) in the report window.", total)
            return total
        if total == 0:
            log.info("No work items yet - producer may not have loaded the queue.")
            if time.monotonic() - started >= INGESTION_WAIT_MAX_SEC:
                return 0
        if time.monotonic() >= deadline:
            return 0
        sleep_capped(max(POLL_INTERVAL_SEC, MIN_POLL_INTERVAL_SEC), deadline)


def local_mode(window):
    return window.status_mode == "local"


def read_queue(client, queue_id, window, state):
    """Counting on our side: ONE download gives total and waiting. Returns (total, waiting) or None."""
    got = safe_call(state, fetch_all_items, client, queue_id, window, True)
    if got is None:
        return None
    items = got[0]
    note_breakdown(window, items)
    window.last_total = len(items)
    window.size_hint = len(items)
    window.last_counts = Counter(status_label(i.get("status")) for i in items)
    return len(items), sum(1 for i in items if category_of(i.get("status")) == "Waiting")


def poll_waiting(client, queue_id, window, state):
    """Number of items still waiting, or None if this check failed temporarily."""
    if local_mode(window):
        result = read_queue(client, queue_id, window, state)
        return None if result is None else result[1]
    return safe_call(state, count_items, client, queue_id, window, WAITING_STATUSES)


def snapshot(client, queue_id, window, state):
    """(total, waiting) or None if a check failed temporarily."""
    if local_mode(window):
        return read_queue(client, queue_id, window, state)
    total = safe_call(state, count_items, client, queue_id, window)
    waiting = safe_call(state, count_items, client, queue_id, window, WAITING_STATUSES)
    if total is None or waiting is None:
        return None
    window.size_hint = total
    return total, waiting


def confirm_finished(client, queue_id, window, state, deadline):
    """'0 waiting' must hold on several checks with an UNCHANGED total.
    Guards against: producer still adding items, item bouncing between
    statuses (retry), and a one-off bad reply.
    LAST STEP (when the Control Room's own filter is used): the whole queue is re-read and
    every status is checked on OUR side. If the filter missed something (e.g. a different
    spelling of 'Ready to run'), this catches it and all counting moves to our side.
    If the filter did NOT miss it (new work simply arrived), we just go back to polling."""
    first = snapshot(client, queue_id, window, state)
    if first is None:
        return False
    first_total, first_waiting = first
    if first_waiting > 0:
        return False
    for i in range(1, CONFIRMATIONS):
        sleep_capped(CONFIRM_GAP_SEC, deadline)
        snap = snapshot(client, queue_id, window, state)
        if snap is None:
            return False
        total, waiting = snap
        if waiting > 0 or total != first_total:
            log.info("Confirmation %d failed (waiting=%d, total %d -> %d). Back to polling.",
                     i, waiting, first_total, total)
            return False
        log.info("Confirmation %d/%d passed (waiting=0, total=%d).", i, CONFIRMATIONS - 1, total)

    if not local_mode(window):
        got = safe_call(state, fetch_all_items, client, queue_id, window, True)
        if got is None:
            return False
        waiting_items = [it for it in got[0] if category_of(it.get("status")) == "Waiting"]
        if waiting_items:
            by = Counter(status_label(it.get("status")) for it in waiting_items)
            server_now = safe_call(state, count_items, client, queue_id, window, WAITING_STATUSES)
            if server_now == 0:
                log.warning("Control Room filter said 0 waiting, but %d item(s) ARE still waiting (%s). "
                            "Counting on our side from now on.", len(waiting_items), dict(by))
                window.status_mode = "local"
            else:
                log.info("New work arrived during the final check (%d waiting: %s) - back to polling.",
                         len(waiting_items), dict(by))
            return False
    log.info("Final check: no item is waiting. Queue is finished.")
    return True


def find_stuck_active(items):
    """Active items that have not changed for STUCK_ACTIVE_MIN minutes: [(id, minutes), ...] oldest first."""
    if not STUCK_ACTIVE_MIN:
        return []
    now = datetime.now(timezone.utc)
    out = []
    for it in items:
        if canon_status(it.get("status")) != "ACTIVE":
            continue
        changed = parse_ts(it.get("updatedOn")) or parse_ts(it.get("createdOn"))
        if changed and (now - changed) >= timedelta(minutes=STUCK_ACTIVE_MIN):
            out.append((it.get("id"), int((now - changed).total_seconds() / 60)))
    out.sort(key=lambda x: -x[1])
    return out


def stuck_text(stuck):
    shown = ", ".join(f"ID {i} ({m} min)" for i, m in stuck[:10])
    more = f" and {len(stuck) - 10} more" if len(stuck) > 10 else ""
    return (f"{len(stuck)} Active item(s) have not changed for over {STUCK_ACTIVE_MIN} min "
            f"(possible stuck bot / dead runner): {shown}{more}.")


def hourly_warning(client, queue_id, window, state, t0, last_change, deadline, pending):
    """Called about once an hour while items are still waiting (log only)."""
    now = time.monotonic()
    hours = (now - t0) / 3600.0
    count, detail, hold_note, stuck_note = pending, "", "", ""
    waiting_items = None
    # Cross-check the server-side count against our own. The server count is taken right before
    # the download, and a difference is re-checked once, because items change status all the time.
    for attempt in (1, 2):
        server = None
        if not local_mode(window):
            server = safe_call(state, count_items, client, queue_id, window, WAITING_STATUSES)
        got = safe_call(state, fetch_all_items, client, queue_id, window, True)
        if got is None:
            break
        waiting_items = [it for it in got[0] if category_of(it.get("status")) == "Waiting"]
        if server is None or server == len(waiting_items):
            break
        if attempt == 1:
            log.info("Cross-check differs (Control Room %d, ours %d) - items may have just changed, "
                     "checking once more.", server, len(waiting_items))
        else:
            log.warning("Cross-check still differs (Control Room %d, ours %d) - "
                        "counting on our side from now on.", server, len(waiting_items))
            window.status_mode = "local"
    if waiting_items is not None:
        count = len(waiting_items)
        by = Counter(status_label(it.get("status")) for it in waiting_items)
        detail = " (" + ", ".join(f"{n} x {name}" for name, n in by.most_common()) + ")"
        held = [parse_ts(it.get("updatedOn")) for it in waiting_items if canon_status(it.get("status")) == "ON_HOLD"]
        held = [h for h in held if h]
        if held:
            age = (datetime.now(timezone.utc) - min(held)).total_seconds() / 3600.0
            hold_note = f" The longest-waiting On hold item last changed {age:.1f} h ago - someone must release it."
        stuck = find_stuck_active(waiting_items)
        if stuck:
            stuck_note = " " + stuck_text(stuck)
    idle_min = int((now - last_change) / 60)
    idle_note = (f" No change in the waiting count for {idle_min} min - check bot runners / devices / pool."
                 if idle_min >= 30 else "")
    if deadline == float("inf"):
        limit_note = " No time limit is set."
    else:
        limit_note = (f" Monitoring continues for up to {max(0.0, (deadline - now) / 3600.0):.1f} h more, "
                      f"then a PARTIAL report is created.")
    msg = f"{count} work item(s) still waiting after {hours:.1f} h{detail}.{hold_note}{stuck_note}{idle_note}{limit_note}"
    log.warning(msg)


def log_poll(client, queue_id, window, state, pending, t0, interval):
    """One clear progress line after every check."""
    names = [STATUS_LABELS.get(c, c) for c in WAITING_STATUSES]
    total, done = None, {}
    if local_mode(window) and window.last_counts is not None:
        by = {n: window.last_counts.get(n, 0) for n in names}
        total = window.last_total
        done = {k: v for k, v in window.last_counts.items() if k not in names and v}
    else:
        by = window.last_waiting_by or {}
        if not window.local_date:                       # cheap one-number call
            total = safe_call(state, count_items, client, queue_id, window)
            if total:
                window.size_hint = total
    text = f"Poll #{state.poll_count} | waiting {pending}"
    if by:
        text += " (" + ", ".join(f"{k} {v}" for k, v in by.items()) + ")"
    if total:
        finished = max(0, total - pending)
        text += f" | finished {finished}/{total} ({100.0 * finished / total:.0f}%)"
    if done:
        text += " [" + ", ".join(f"{k} {v}" for k, v in sorted(done.items())) + "]"
    text += f" | elapsed {timedelta(seconds=int(time.monotonic() - t0))} | next check in {interval} s"
    log.info(text)


def monitor_queue(client, queue_id, window, state, deadline):
    """Main poll loop. Returns 'COMPLETE', 'TIMEOUT' or 'NO_ITEMS'."""
    t0 = time.monotonic()
    if not wait_for_ingestion(client, queue_id, window, state, deadline):
        log.warning("No work items found in the report window (since %s UTC) after %d sec - nothing to "
                    "monitor, stopping. (Old items in the queue do not count. To test on old data set "
                    "WINDOW_START, e.g. \"2025-06-01T00:00:00Z\", or LOOKBACK_DAYS.)",
                    window.start, INGESTION_WAIT_MAX_SEC)
        return "NO_ITEMS" if time.monotonic() < deadline else "TIMEOUT"

    last_pending, last_change = None, time.monotonic()
    next_warn = t0 + WARN_EVERY_SEC
    quiet_since, quiet_total = None, None
    log.info("Monitoring statuses %s (time limit: %s, warning every %d min, quiet period %d s).",
             [STATUS_LABELS.get(x, x) for x in WAITING_STATUSES],
             f"{MAX_RUNTIME_HOURS} h" if MAX_RUNTIME_SEC else "none", WARN_EVERY_SEC // 60, QUIET_PERIOD_SEC)

    while time.monotonic() < deadline:
        interval = poll_interval(window)
        pending = poll_waiting(client, queue_id, window, state)
        state.poll_count += 1
        if pending is not None:
            state.max_pending_seen = max(state.max_pending_seen, pending)
            log_poll(client, queue_id, window, state, pending, t0, interval)
            now = time.monotonic()
            if pending != last_pending:
                last_pending, last_change = pending, now

            if pending == 0:
                if QUIET_PERIOD_SEC <= 0:
                    if confirm_finished(client, queue_id, window, state, deadline):
                        return "COMPLETE"
                else:
                    snap = snapshot(client, queue_id, window, state)
                    if snap is not None and snap[1] == 0:
                        total = snap[0]
                        if quiet_since is None or total != quiet_total:
                            if quiet_since is not None:
                                log.info("Total changed (%s -> %d) while idle - quiet period restarts.",
                                         quiet_total, total)
                            quiet_since, quiet_total = now, total
                            log.info("Nothing waiting (total %d). Waiting %d s to make sure no more items arrive.",
                                     total, QUIET_PERIOD_SEC)
                        elif now - quiet_since >= QUIET_PERIOD_SEC:
                            log.info("Quiet period of %d s passed with an unchanged total - final checks.",
                                     QUIET_PERIOD_SEC)
                            if confirm_finished(client, queue_id, window, state, deadline):
                                return "COMPLETE"
                            quiet_since = None
                    else:
                        quiet_since = None
            else:
                if quiet_since is not None:
                    log.info("New work arrived (%d waiting) - quiet period cancelled, polling continues.", pending)
                quiet_since = None

            if pending > 0 and now >= next_warn:          # hourly warning
                hourly_warning(client, queue_id, window, state, t0, last_change, deadline, pending)
                next_warn += WARN_EVERY_SEC
                if next_warn <= now:
                    next_warn = now + WARN_EVERY_SEC
        sleep_capped(interval, deadline)

    return "TIMEOUT"


# ==============================================================================
# 7. FETCH ALL ITEMS (safe paging)
# ==============================================================================
def fetch_all_items(client, queue_id, window, quiet=False):
    """Downloads every item in the window. Returns (items_list, warnings_list).
    Paging example with PAGE_SIZE 400 and 1300 items: items 1-400, 401-800, 801-1200 and
    1201-1300. After each page we add the real number of items received to the offset; when the offset
    reaches the total the server reports (page.totalFilter), that was the LAST page. An empty page
    also ends the loop (so it can never run forever).
    Newest first (createdOn never changes, so bots updating items cannot make us skip any).
    If the Control Room ignores our date filter, we stop paging as soon as we reach items older
    than the window - but only after PROVING the newest-first order on 50+ items; otherwise we
    simply read everything."""
    warnings, by_id = [], {}
    start = window.start_dt()
    offset, use_sort, page_len, reported_total = 0, True, window.page_len, None
    sort = [{"field": "createdOn", "direction": "desc"}]
    order_ok, prev_created, seen, page_no = True, None, 0, 0
    while True:
        payload = window.body(offset, page_len, None, sort if use_sort else None)
        try:
            data = client.post(f"/v3/wlm/queues/{queue_id}/workitems/list", payload)
        except FatalError as exc:
            if exc.status == 400 and use_sort:          # some tenants reject this sort field
                log.warning("Sort not accepted - continuing without sort.")
                use_sort, order_ok = False, False
                continue
            if exc.status == 400 and page_len > 100:    # some tenants cap the page size lower
                smaller = 200 if page_len > 200 else 100
                log.warning("Page size %d not accepted - trying %d.", page_len, smaller)
                page_len = window.page_len = smaller
                continue
            raise
        batch = data.get("list") or []
        page = data.get("page") or {}
        reported_total = page.get("totalFilter", reported_total)
        for item in batch:
            key = item.get("id")
            by_id[key if key is not None else ("noid", len(by_id))] = item   # de-duplicate
            created = parse_ts(item.get("createdOn"))
            if created is None or (prev_created is not None and created > prev_created):
                order_ok = False                        # not newest-first (or unreadable) -> read everything
            else:
                prev_created = created
            seen += 1
        if not batch:
            break                                       # empty page = end (no infinite loop)
        first = offset + 1
        offset += len(batch)                            # use REAL size: server may cap page length
        page_no += 1
        is_last = reported_total is not None and offset >= reported_total
        if not quiet:
            log.info("Page %d: items %d-%d of %s%s", page_no, first, offset,
                     reported_total if reported_total is not None else "?", "  <- last page" if is_last else "")
        if is_last:
            break
        if window.local_date and use_sort and order_ok and seen >= 50 and prev_created is not None \
                and prev_created < start:
            break                                       # everything after this is older than the window
    items = list(by_id.values())
    inside = [i for i in items if (parse_ts(i.get("createdOn")) or start) >= start]   # always double-check
    if len(inside) != len(items) and not window.local_date:
        log.warning("%d item(s) older than the window were returned by the Control Room - ignored.",
                    len(items) - len(inside))
        warnings.append(f"{len(items) - len(inside)} item(s) older than the report window were ignored.")
    if len(inside) != len(items):
        items = inside
    elif not window.local_date and reported_total is not None and len(by_id) != reported_total:
        msg = f"Fetched {len(by_id)} items but Control Room reported {reported_total} (data changed while paging)."
        log.warning(msg)
        warnings.append(msg)
    return items, warnings


# ==============================================================================
# 8. TURN RAW ITEMS INTO REPORT ROWS
# ==============================================================================
def to_text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def category_of(status):
    c = canon_status(status)
    if c == "COMPLETED":
        return "Success"
    if c == "FAILED":
        return "Failed"
    if c == "DATA_ERROR":
        return "Data error"
    if c in WAITING_STATUSES:
        return "Waiting"
    return "Attention"      # Unknown, On hold (if WAIT_FOR_ON_HOLD=False) or an unrecognised status


def build_rows(items, tz=None):
    """tz is kept only for compatibility: every time is converted with the computer's local time zone
    rule for THAT moment (astimezone() without argument), so daylight saving changes are handled."""
    rows, data_keys = [], []
    for it in items:
        created, updated = parse_ts(it.get("createdOn")), parse_ts(it.get("updatedOn"))
        minutes = None
        if created and updated:
            minutes = round((updated - created).total_seconds() / 60.0, 2)
        status = status_label(it.get("status"))
        # Error text: field name differs between versions, so try a few.
        error = next((to_text(it.get(k)) for k in ("error", "errorMessage", "errorDetails") if it.get(k)), "")
        payload = it.get("json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {"raw": payload}
        payload = payload if isinstance(payload, dict) else {}
        for k in payload:
            if k not in data_keys:
                data_keys.append(k)
        rows.append({
            "WorkItem_ID": it.get("id"),
            "Status": status,
            "Category": category_of(it.get("status")),
            "Result": to_text(it.get("result")),
            "Error_Message": error,
            "Created_On": created.astimezone().replace(tzinfo=None) if created else None,
            "Updated_On": updated.astimezone().replace(tzinfo=None) if updated else None,
            # created -> last update. This INCLUDES the time the item waited in the queue, so it is
            # not the bot's processing time.
            "Created_To_Updated_Min": minutes,
            "Created_By": to_text(it.get("createdBy")),
            "Updated_By": to_text(it.get("updatedBy")),
            "Comment": to_text(it.get("comment")),
            "_data": payload,
        })
    return rows, data_keys[:MAX_DATA_COLUMNS]


def summarize(rows):
    cats = Counter(r["Category"] for r in rows)
    statuses = Counter(r["Status"] or "(blank)" for r in rows)
    finished = cats["Success"] + cats["Failed"] + cats["Data error"]
    elapsed = [r["Created_To_Updated_Min"] for r in rows
               if r["Created_To_Updated_Min"] is not None and r["Category"] in ("Success", "Failed", "Data error")]
    return {
        "total": len(rows), "success": cats["Success"], "failed": cats["Failed"],
        "data_error": cats["Data error"], "waiting": cats["Waiting"], "attention": cats["Attention"],
        "success_rate": (cats["Success"] / finished) if finished else None,
        "avg_min": (sum(elapsed) / len(elapsed)) if elapsed else None,
        "max_min": max(elapsed) if elapsed else None,
        "statuses": statuses,
    }


def normalise_error(text):
    """Group similar errors: drop numbers/ids so 'Row 5 failed' == 'Row 9 failed'."""
    t = re.sub(r"\s+", " ", text or "").strip()
    t = re.sub(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}", "<id>", t)
    t = re.sub(r"\d+", "#", t)
    return t[:200] or "(no error text)"


# ==============================================================================
# 9. EXCEL REPORT
# ==============================================================================
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CAT_FILL = {"Success": PatternFill("solid", fgColor="E2F0D9"),
            "Failed": PatternFill("solid", fgColor="F8CBAD"),
            "Data error": PatternFill("solid", fgColor="FCE4D6"),
            "Waiting": PatternFill("solid", fgColor="FFF2CC"),
            "Attention": PatternFill("solid", fgColor="D9D2E9")}
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def put(ws, row, col, value, bold=False, fill=None, wrap=False, number_format=None):
    """Write a cell safely: strips illegal chars, trims to Excel limit and
    forces text so values starting with = + - @ are NEVER run as formulas."""
    if isinstance(value, str):
        value = _ILLEGAL.sub("", value)[:32000]
    cell = ws.cell(row=row, column=col, value=value)
    if isinstance(value, str):
        cell.data_type = "s"
    if bold:
        cell.font = Font(bold=True)
    if fill:
        cell.fill = fill
    if wrap:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    if number_format:
        cell.number_format = number_format
    return cell


def write_table(ws, headers, records, widths=None, wrap_cols=(), date_cols=(), color_by_category_col=None):
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill, cell.font, cell.border = HEADER_FILL, HEADER_FONT, BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for r, rec in enumerate(records, 2):
        for c, val in enumerate(rec, 1):
            cell = put(ws, r, c, val, wrap=(c in wrap_cols),
                       number_format="yyyy-mm-dd hh:mm:ss" if c in date_cols else None)
            cell.border = BORDER
        if color_by_category_col:
            fill = CAT_FILL.get(rec[color_by_category_col - 1])
            if fill:
                ws.cell(row=r, column=color_by_category_col).fill = fill
    ws.freeze_panes = "A2"
    if records:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(records) + 1}"
    for c, w in enumerate(widths or [], 1):
        ws.column_dimensions[get_column_letter(c)].width = w


def build_workbook(rows, data_keys, state, ctx):
    wb = Workbook()
    s = summarize(rows)
    outcome_text = {"COMPLETE": "COMPLETE - all items finished",
                    "TIMEOUT": f"PARTIAL - time limit reached{f' ({MAX_RUNTIME_HOURS} h)' if MAX_RUNTIME_SEC else ''}, some items still waiting",
                    "NO_ITEMS": "NO ITEMS - queue stayed empty",
                    "INTERRUPTED": "PARTIAL - run was stopped",
                    "ERROR": "PARTIAL - run stopped by an error"}.get(state.outcome, state.outcome)

    # ---- Sheet 1: Summary ---------------------------------------------------
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Automation Anywhere - WLM Queue Daily Report"
    ws["A1"].font = Font(bold=True, size=16, color="1F3864")
    info = [
        ("Run outcome", outcome_text),
        ("Control Room", ctx["cr_url"]),
        ("Queue", f"ID {ctx['queue_id']}"),
        ("Report window start", f"{ctx['window_local']:%Y-%m-%d %H:%M} computer time, {ctx['window_local']:%Z} ({ctx['window_utc']} UTC)"),
        ("Run started", f"{state.started:%Y-%m-%d %H:%M:%S}"),
        ("Run finished", f"{state.finished:%Y-%m-%d %H:%M:%S}" if state.finished else ""),
        ("Monitoring duration", str(timedelta(seconds=int((state.finished - state.started).total_seconds()))) if state.finished else ""),
        ("Polls / re-logins", f"{state.poll_count} polls / {ctx['logins']} logins (token refresh)"),
        ("API requests", f"{ctx.get('api_calls', 0)} requests, {ctx.get('rate_pauses', 0)} self-imposed rate-limit pauses, "
                         f"{ctx.get('rate_429', 0)} HTTP 429 replies"),
    ]
    r = 3
    for k, v in info:
        put(ws, r, 1, k, bold=True)
        put(ws, r, 2, v)
        r += 1
    if state.error_text:
        put(ws, r, 1, "Error", bold=True)
        put(ws, r, 2, state.error_text, wrap=True)
        r += 1

    r += 1
    put(ws, r, 1, "KEY NUMBERS", bold=True)
    r += 1
    waiting_names = "New / Ready to run / Active" + (" / On hold" if "ON_HOLD" in WAITING_STATUSES else "")
    kpis = [("Total work items", s["total"], None),
            ("Completed", s["success"], CAT_FILL["Success"]),
            ("Failed (bot / processing error)", s["failed"], CAT_FILL["Failed"]),
            ("Data error (bad input data)", s["data_error"], CAT_FILL["Data error"]),
            (f"Still waiting ({waiting_names})", s["waiting"], CAT_FILL["Waiting"]),
            ("Needs attention (Unknown / other)", s["attention"], CAT_FILL["Attention"])]
    for k, v, fill in kpis:
        put(ws, r, 1, k, bold=True, fill=fill)
        put(ws, r, 2, v, fill=fill)
        r += 1
    put(ws, r, 1, "Success rate (Completed / finished)", bold=True)
    c = put(ws, r, 2, s["success_rate"] if s["success_rate"] is not None else "n/a", number_format="0.0%")
    c.alignment = Alignment(horizontal="left")
    r += 1
    put(ws, r, 1, "Avg minutes created -> last update (incl. queue wait)", bold=True)
    put(ws, r, 2, round(s["avg_min"], 2) if s["avg_min"] is not None else "n/a")
    ws.cell(row=r, column=2).alignment = Alignment(horizontal="left")
    r += 1
    put(ws, r, 1, "Max minutes created -> last update (incl. queue wait)", bold=True)
    put(ws, r, 2, s["max_min"] if s["max_min"] is not None else "n/a")
    ws.cell(row=r, column=2).alignment = Alignment(horizontal="left")
    r += 2

    # status breakdown table (also feeds the chart)
    put(ws, r, 1, "STATUS BREAKDOWN", bold=True)
    r += 1
    head_row = r
    for c_i, h in enumerate(("Status", "Count"), 1):
        cell = ws.cell(row=r, column=c_i, value=h)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    for status, count in s["statuses"].most_common():
        r += 1
        put(ws, r, 1, status)
        put(ws, r, 2, count)
    if s["statuses"]:
        chart = BarChart()
        chart.title, chart.style, chart.legend = "Work items by status", 10, None
        chart.add_data(Reference(ws, min_col=2, min_row=head_row, max_row=r), titles_from_data=True)
        chart.set_categories(Reference(ws, min_col=1, min_row=head_row + 1, max_row=r))
        chart.height, chart.width = 7, 13
        ws.add_chart(chart, "D3")
    r += 2

    if ctx["warnings"]:
        put(ws, r, 1, "DATA WARNINGS", bold=True)
        for w in ctx["warnings"]:
            r += 1
            put(ws, r, 1, w, wrap=False)
    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["B"].width = 60

    # ---- Sheet 2: Work Items ------------------------------------------------
    data_headers = [f"Data_{k}" for k in data_keys] if INCLUDE_ITEM_DATA else []
    base_headers = ["WorkItem_ID", "Status", "Category", "Result", "Error_Message",
                    "Created_On", "Updated_On", "Created_To_Updated_Min", "Created_By", "Updated_By", "Comment"]
    records = []
    for row in rows:
        rec = [row[h] for h in base_headers]
        if INCLUDE_ITEM_DATA:
            rec += [to_text(row["_data"].get(k)) for k in data_keys]
        records.append(rec)
    ws2 = wb.create_sheet("Work Items")
    write_table(ws2, base_headers + data_headers, records,
                widths=[14, 15, 11, 18, 60, 20, 20, 12, 18, 18, 30] + [20] * len(data_headers),
                wrap_cols=(5,), date_cols=(6, 7), color_by_category_col=3)

    # ==========================================================================================
    # COMMENTED OUT (not removed): the three sheets below are switched off on request.
    # To switch them back on, remove the leading "# " from every line of the block.
    # ==========================================================================================

    # ---- Sheet 3: Failures / attention items ----------------------------------
    # bad = [rec for rec in records if rec[2] in ("Failed", "Data error", "Waiting", "Attention")]
    # ws3 = wb.create_sheet("Failures & Attention")
    # write_table(ws3, base_headers + data_headers, bad,
    #             widths=[14, 15, 11, 18, 60, 20, 20, 12, 18, 18, 30] + [20] * len(data_headers),
    #             wrap_cols=(5,), date_cols=(6, 7), color_by_category_col=3)

    # ---- Sheet 4: Error summary (Failed and Data error shown separately) ----------------
    # groups = Counter((rec[2], normalise_error(rec[4])) for rec in bad if rec[2] in ("Failed", "Data error"))
    # cat_totals = Counter()
    # for (cat, _), n in groups.items():
    #     cat_totals[cat] += n
    # ordered = sorted(groups.items(), key=lambda kv: (kv[0][0] != "Failed", -kv[1]))
    # ws4 = wb.create_sheet("Error Summary")
    # write_table(ws4, ["Category", "Error (numbers/ids masked)", "Count", "% of that category"],
    #             [[cat, msg, n, n / cat_totals[cat]] for (cat, msg), n in ordered],
    #             widths=[14, 100, 10, 18], wrap_cols=(2,), color_by_category_col=1)
    # for rr in range(2, len(ordered) + 2):
    #     ws4.cell(row=rr, column=4).number_format = "0.0%"

    # ---- Sheet 5: Run log (warnings & errors captured during the run) ------------------
    # ws5 = wb.create_sheet("Run Log")
    # write_table(ws5, ["Time", "Level", "Message"],
    #             [[t, lvl, msg] for t, lvl, msg in MEMORY_LOG.records] or [[datetime.now(), "INFO", "No warnings during this run."]],
    #             widths=[20, 10, 120], wrap_cols=(3,), date_cols=(1,))
    return wb, s


def save_workbook(wb, stamp, partial):
    """Never lose the report: if the file is open in Excel, save under a new name."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE_NAME:                    # exact file name given in args[6]
        base = Path(OUTPUT_FILE_NAME).stem
        name = f"{base}{'_PARTIAL' if partial else ''}.xlsx"
    else:
        name = f"WLM_Consolidated_Report_{stamp}{'_PARTIAL' if partial else ''}.xlsx"
    path = OUTPUT_DIR / name
    try:
        wb.save(path)
    except PermissionError:
        path = OUTPUT_DIR / f"{path.stem}_{datetime.now():%H%M%S}.xlsx"
        wb.save(path)
    return path


# ==============================================================================
# 10. MAIN
# ==============================================================================
_STOP = {"seen": False}


def _stop_handler(signum, frame):          # SIGTERM (Jenkins abort, scheduler stop)
    if _STOP["seen"]:
        return                             # already stopping: let the partial report finish
    _STOP["seen"] = True
    raise KeyboardInterrupt()


def get_credentials():
    """Check the login inputs. Rule: password present -> API key optional;
    password missing -> API key is required. (No prompt: A360 runs without a keyboard.)"""
    if not USERNAME:
        raise FatalError("Username (args[2]) is empty.")
    if not PASSWORD and not API_KEY:
        raise FatalError("Password (args[3]) is empty and no API key (args[4]) was given - "
                         "provide a password or an API key.")
    log.info("Login method: %s.", "password" + (" (API key as fallback)" if API_KEY else "") if PASSWORD
             else "API key")


def download_with_retries(client, queue_id, window, state):
    """Download everything (best effort, even after timeout / error / stop request).
    A stop request (SIGTERM or the first Ctrl+C) during the download does NOT cancel it: the download
    is repeated so a partial report can still be written. A SECOND Ctrl+C quits at once."""
    attempt, interrupted_once = 0, False
    while attempt < 3:
        attempt += 1
        try:
            return fetch_all_items(client, queue_id, window)
        except KeyboardInterrupt:
            if interrupted_once:
                raise                       # second Ctrl+C = really quit
            interrupted_once = True
            attempt -= 1                    # a stop request does not use up a retry
            log.warning("Stop requested during the download - finishing the download first so the report "
                        "is not lost (press Ctrl+C again to quit at once).")
        except (FatalError, TransientError) as exc:
            log.error("Could not download work items (attempt %d/3): %s", attempt, exc)
            if attempt == 3 or isinstance(exc, FatalError):
                state.error_text = ((state.error_text + " | " if state.error_text else "")
                                    + f"Download failed: {exc}")
                if state.outcome == "COMPLETE":
                    state.outcome = "ERROR"
                return [], []
            time.sleep(30)
    return [], []


def run_report(early_note=""):
    setup_logging()
    _STOP["seen"] = False
    log.info("Script version: output path (args[6]) supported. Output folder in use: %s%s",
             OUTPUT_DIR.resolve(), f" | report file name: {OUTPUT_FILE_NAME}" if OUTPUT_FILE_NAME else "")
    if early_note:
        log.warning(early_note)
    try:
        signal.signal(signal.SIGTERM, _stop_handler)
    except (ValueError, OSError, AttributeError):   # not the main thread / not supported here - fine
        pass
    state = RunState()
    client, ctx, items, rows_warnings = None, {"warnings": [], "queue_id": QUEUE_ID, "logins": 0}, [], []
    exit_code = 1
    try:
        # ---- setup ---------------------------------------------------------------
        base = clean_base_url(CR_URL)
        get_credentials()
        window_str, window_dt, tz = compute_window_start()
        window = Window(window_str)
        ctx.update(cr_url=base, window_utc=window_str, window_local=window_dt.astimezone(tz))
        deadline = time.monotonic() + MAX_RUNTIME_SEC if MAX_RUNTIME_SEC else float("inf")
        log.info("Starting. Control Room: %s | window start (UTC): %s", base, window_str)

        client = ControlRoomClient(base)
        client.login()
        queue_id = resolve_queue_id(client)
        ctx.update(queue_id=queue_id)
        negotiate_api_format(client, queue_id, window)   # find a request format this Control Room accepts
        discover_statuses(client, queue_id, window)      # log real statuses + confirm it is the right queue

        # ---- wait until everything is processed --------------------------------------
        try:
            state.outcome = monitor_queue(client, queue_id, window, state, deadline)
        except KeyboardInterrupt:
            state.outcome = "INTERRUPTED"
            log.warning("Stopped by user/scheduler - building partial report.")
        except FatalError as exc:
            state.outcome, state.error_text = "ERROR", str(exc)
            log.error("%s", exc)

        # ---- download everything (best effort, even after timeout / error / stop) -------------
        if state.outcome != "NO_ITEMS":
            items, rows_warnings = download_with_retries(client, queue_id, window, state)
        exit_code = {"COMPLETE": 0, "TIMEOUT": 2, "NO_ITEMS": 3, "INTERRUPTED": 130, "ERROR": 1}[state.outcome]

    except (FatalError, TransientError) as exc:
        state.outcome, state.error_text = "ERROR", str(exc)
        log.error("FATAL: %s", exc)
    except KeyboardInterrupt:
        state.outcome, exit_code = "INTERRUPTED", 130
    except Exception as exc:               # last safety net - never die silently
        state.outcome, state.error_text = "ERROR", f"Unexpected error: {type(exc).__name__}: {exc}"
        log.exception("Unexpected error")
    finally:
        state.finished = datetime.now()
        if client:
            ctx["logins"] = client.login_count
            ctx.update(api_calls=getattr(client, "request_count", 0), rate_pauses=getattr(client, "pause_count", 0),
                       rate_429=getattr(client, "rate_429_count", 0))

    # ---- report ------------------------------------------------------------------------------
    report_path, summary = None, summarize([])
    if state.outcome not in ("NO_ITEMS",) and "window_local" in ctx and ctx["logins"] > 0 and not items:
        log.warning("No work items were downloaded - no Excel report created.")
    if "window_local" in ctx and ctx["logins"] > 0 and items:      # nothing downloaded = nothing to report
        try:
            if state.outcome == "COMPLETE" and any(category_of(i.get("status")) == "Waiting" for i in items):
                rows_warnings.append("Some items moved back to a waiting status after the final check.")
            stuck = find_stuck_active(items)
            if stuck:
                msg = stuck_text(stuck)
                log.warning(msg)
                rows_warnings.append(msg)
            ctx["warnings"] = rows_warnings
            rows, data_keys = build_rows(items)
            attention = Counter(r["Status"] for r in rows if r["Category"] == "Attention")
            if attention:
                msg = ("Items needing attention (not waited for): "
                       + ", ".join(f"{n} x {name}" for name, n in attention.most_common())
                       + ". Filter the Category column in the 'Work Items' sheet to see them.")
                log.warning(msg)
                rows_warnings.append(msg)
            odd = sorted({r["Status"] for r in rows if status_key(r["Status"]) not in STATUS_ALIASES
                          and r["Status"] and r["Category"] == "Attention"})
            if odd:
                msg = (f"Unrecognised status value(s): {', '.join(odd)}. They were NOT waited for. If any means "
                       f"'still waiting', tell the script owner so STATUS_ALIASES / WAITING_STATUSES can be updated.")
                log.warning(msg)
                rows_warnings.append(msg)
            wb, summary = build_workbook(rows, data_keys, state, ctx)
            report_path = save_workbook(wb, datetime.now().strftime("%Y%m%d_%H%M%S"), state.outcome != "COMPLETE")
            log.info("Report saved: %s", report_path.resolve())
        except Exception as exc:
            log.exception("Report could not be created")
            state.error_text += f" | Report failed: {exc}"
            exit_code = 1

    # E-mail is not part of this script. To send the report yourself, add your code here:
    #   report_path = path of the Excel file (None if no report was created)
    #   state.outcome = COMPLETE / TIMEOUT / NO_ITEMS / INTERRUPTED / ERROR
    #   summary = dict with totals (summary['success'], ['failed'], ['total'] ...)

    if client:
        client.logout()
    log.info("Finished with outcome %s (exit code %d).", state.outcome, exit_code)
    return exit_code, state, report_path, summary, ctx


# ==============================================================================
# 11. AUTOMATION ANYWHERE ENTRY POINT
# ==============================================================================
def _txt(value):
    """AA360 may hand over None, numbers, dates or text - turn anything into clean text."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)                  # 100025611.0 -> 100025611
    return str(value).strip()


def _date_to_window_start(value):
    """'' -> '' (today). Accepts 2026-10-19, 2026/10/19, 19/10/2026, 19-10-2026, 19.10.2026,
    19-Oct-2026, 19 Oct 2026, or a full ISO date-time. Returns 'YYYY-MM-DDT00:00:00Z' built from the
    LOCAL midnight of that date, so it matches how 'today' is computed when no date is given."""
    if isinstance(value, datetime):
        d = value.date()
    elif hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        d = value
    else:
        text = _txt(value)
        if not text:
            return ""
        d = None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y",
                    "%d %B %Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
            try:
                d = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                continue
        if d is None:
            dt = parse_ts(text)
            if dt is None:
                raise FatalError(f"Date (args[0]) '{text}' is not understood. Use e.g. 2026-10-19 or 19/10/2026.")
            d = dt.date()
    local_midnight = datetime(d.year, d.month, d.day)          # naive = this computer's clock
    return local_midnight.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generateReport(args):
    """Entry point for the A360 Python Script package.

    args = [date, cr_url, username, password, api_key, queue_id, output_path]
    Returns a text summary (also when something goes wrong - it never raises)."""
    global CR_URL, USERNAME, PASSWORD, API_KEY, QUEUE_ID, WINDOW_START, OUTPUT_DIR, OUTPUT_FILE_NAME
    try:
        args = list(args) if args is not None else []
        _received = len(args)
        args += [""] * (7 - len(args))                    # missing trailing values = empty
        CR_URL = _txt(args[1])
        USERNAME = _txt(args[2])
        PASSWORD = _txt(args[3]) if args[3] is not None else ""
        API_KEY = _txt(args[4])
        QUEUE_ID = _txt(args[5])
        WINDOW_START = _date_to_window_start(args[0])     # may raise FatalError (bad date)
        if not CR_URL:
            raise FatalError("Control Room URL (args[1]) is empty.")
        if not QUEUE_ID:
            raise FatalError("Queue ID (args[5]) is empty.")

        # ---- output path (args[6]) : a folder, or a full .xlsx file name ---------------------
        out = _txt(args[6]).strip('"').strip("'").strip()
        OUTPUT_FILE_NAME = ""
        if not out:
            OUTPUT_DIR = _default_output_dir()
        else:
            out = os.path.expanduser(os.path.expandvars(out))
            p = Path(out)
            if p.suffix.lower() == ".xlsx":
                OUTPUT_FILE_NAME, p = p.name, p.parent
            elif p.suffix:
                raise FatalError(f"Output path (args[6]) '{out}' must be a folder or end in .xlsx.")
            OUTPUT_DIR = p
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            probe = OUTPUT_DIR / ".write_test"            # prove we can really write there
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            raise FatalError(f"Cannot write to the output folder '{OUTPUT_DIR}': {exc}")
    except Exception as exc:                              # bad inputs: report, do not crash A360
        return f"ERROR | exit_code=1 | Input problem: {exc}"

    try:
        if _received < 7 or not _txt(args[6]):
            log_early = f"Only {_received} argument(s) received / output path empty - using the default folder."
        else:
            log_early = ""
        exit_code, state, report_path, summary, ctx = run_report(log_early)
    except Exception as exc:                              # absolute last safety net
        return f"ERROR | exit_code=1 | Unexpected error: {type(exc).__name__}: {exc}"

    parts = [f"{state.outcome} | exit_code={exit_code}"]
    parts.append(f"report={report_path.resolve() if report_path else 'none'}")
    parts.append(f"total={summary['total']} success={summary['success']} failed={summary['failed']} "
                 f"data_error={summary['data_error']} waiting={summary['waiting']} "
                 f"attention={summary['attention']}")
    if state.error_text:
        parts.append(f"error={state.error_text}")
    return " | ".join(parts)


if __name__ == "__main__":
    # Local test only (A360 calls generateReport directly). Fill in real values to try it:
    print(generateReport(["", "https://YOUR-CR.my.automationanywhere.digital",
                          "USER", "PASSWORD", "", "100025611", r"C:\WLM_Reports"]))