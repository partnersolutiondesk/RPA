import sys
import os
import time
import datetime
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from urllib.parse import urlparse
import requests
import msal
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File


_DEFAULTS = dict(
    PROXY=None,
)

# System / internal columns to skip in output
_SKIP_COLS_LOWER = {
    "edit", "linktitle", "linktitlenomenu", "itemchildcount",
    "folderchildcount", "contenttype", "authorlookupid",
    "editorlookupid", "_uiversionstring", "odata.type",
    "id", "title", "created", "modified", "author", "editor",
}

# OData filter operators we recognise
_TEXT_OPS = {"eq", "ne", "contains", "startswith", "endswith"}
_COMPARISON_OPS = {"eq", "ne", "gt", "ge", "lt", "le"}
_ALL_OPS = _TEXT_OPS | _COMPARISON_OPS


_STRING_OP_TOKENS = [
    (">=", "ge"),
    ("<=", "le"),
    ("!=", "ne"),
    ("==", "eq"),
    ("~=", "contains"),
    ("^=", "startswith"),
    ("$=", "endswith"),
    (">", "gt"),
    ("<", "lt"),
    ("=", "eq"),
]

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SEP = "-" * 60

# Public SharePoint Online client ID for username/password auth (NOT the Graph app CLIENT_ID).
SHAREPOINT_USERPASS_CLIENT_ID = "9bc3ab49-b65d-410a-85ad-de819febfddc"


# ==============================================================================
#  SIMPLE STRING FILTER PARSER  (for AA360 users who don't want dicts)
# ==============================================================================
def _parse_filter_string(s: str):
    if not isinstance(s, str):
        return None
    raw = s.strip()
    # JSON-dict short form ('{"country":"india"}') is handled by _json_dict_to_filters;
    # bail here so we don't try to parse { ... } as Col=Value.
    if _looks_like_json_dict(raw):
        return None
    if not raw or "=" not in raw and ">" not in raw and "<" not in raw and "~" not in raw and "^" not in raw and "$" not in raw:
        return None

    for token, op in _STRING_OP_TOKENS:
        idx = raw.find(token)
        if idx > 0:
            col = raw[:idx].strip()
            val = raw[idx + len(token):].strip()
            if not col or not val:
                continue
            # Try to cast numeric values
            try:
                if "." in val:
                    val_num = float(val)
                else:
                    val_num = int(val)
                val = val_num
            except (ValueError, TypeError):
                pass
            return {"column": col, "operator": op, "value": val}
    return None


# ==============================================================================
#  INPUT UNPACKING  (dynamic, defensive, index-based for AA360)
# ==============================================================================
_EMPTY_STRINGS = {"", "none", "null", "n/a", "na", "default", "use default", "use_default", "-"}
_BAD_PROXY_HOSTS = {"none", "null", "localhost", "", "local", "hostname", "host", "example", "test", "-", "0"}


def _looks_like_json_dict(raw) -> bool:
    """Heuristic: return True if raw looks like '{...}' JSON dict (string)."""
    if not isinstance(raw, str):
        return False
    s = raw.strip()
    return len(s) >= 2 and s[0] == "{" and s[-1] == "}"


def _json_dict_to_filters(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, dict):
        d = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if not (s.startswith("{") and s.endswith("}")):
            return []
        try:
            d = json.loads(s)
        except Exception:
            return []
        if not isinstance(d, dict):
            return []
    else:
        return []

    out = []
    for k, v in d.items():
        if not isinstance(k, str) or not k:
            continue
        if v is None:
            continue
        out.append({"column": k.strip(), "operator": "eq", "value": v})
    return out


def _diagnose_json_dict_issue(raw, slot_index=None) -> str:
    """
    Best-effort diagnostic for a malformed JSON-dict-looking filter string.
    FIX: now takes the actual slot index it's inspecting (caller loops over
    every filter slot >= 13, not just slot 13) so a syntax error at, say,
    slot 15 is reported correctly instead of silently ignored.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return ""

    try:
        json.loads(s)
        return ""
    except json.JSONDecodeError as e:
        idx_label = f"index {slot_index}" if slot_index is not None else "one of your filter slots"
        hint_lines = [
            f"  [WARN] Your filter at {idx_label} looks like a JSON dict but has a SYNTAX ERROR.",
            f"         Error near position {e.pos}: {e.msg}",
            f"         Your input (trimmed): {s[:120]}{'...' if len(s) > 120 else ''}",
            f"",
            f"         Common fixes:",
            f"           1. Use DOUBLE QUOTES around keys and string values: \"country\" not 'country'",
            f"           2. No trailing comma after the last value",
            f"           3. Correct example: {{ \"country\": \"india\", \"name\": \"dev\" }}",
        ]
        return "\n".join(hint_lines)
    except Exception as e:
        return f"  [WARN] JSON filter could not be parsed: {e}"


def _sanitize_proxies(proxies):
    if not proxies:
        return None
    if not isinstance(proxies, dict):
        return None
    clean = {}
    for key, raw in proxies.items():
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        if "://" not in s:
            s = f"http://{s}"
        try:
            p = urlparse(s)
        except Exception:
            continue
        host = (p.hostname or "").strip().lower()
        if host in _BAD_PROXY_HOSTS or "." not in host and ":" not in host and host not in {"localhost", "127.0.0.1", "::1"}:
            continue
        clean[key] = s
    return clean or None


def _sharepoint_site_url(cfg: dict) -> str:
    """Build full site URL, e.g. https://contoso.sharepoint.com/sites/mysite."""
    host = (cfg.get("SHAREPOINT_HOST") or "").strip().rstrip("/")
    site = (cfg.get("SITE_NAME") or "").strip().strip("/")
    if not host:
        return ""
    if host.startswith(("http://", "https://")):
        base = host.rstrip("/")
    else:
        base = f"https://{host}"
    return f"{base}/{site}" if site else base


def _bare_host(cfg: dict) -> str:
    """
    FIX: Graph's /sites/{hostname}:/{path} syntax needs a bare hostname
    (e.g. 'contoso.sharepoint.com'), not a full URL. Previously get_site_id()
    used SHAREPOINT_HOST directly, which broke if the user supplied it with
    an https:// scheme. Centralize the stripping here.
    """
    host = (cfg.get("SHAREPOINT_HOST") or "").strip()
    if host.startswith(("http://", "https://")):
        host = urlparse(host).netloc
    return host.rstrip("/")


def _tenant_domain_for_userpass(cfg: dict) -> str:
    """Office365 user/pass auth expects tenant.onmicrosoft.com, not a tenant GUID."""
    host = _bare_host(cfg)
    prefix = host.split(".")[0] if host else ""
    if prefix:
        return f"{prefix}.onmicrosoft.com"
    tenant = (cfg.get("TENANT_ID") or "").strip()
    if tenant and ".onmicrosoft.com" in tenant.lower():
        return tenant
    return tenant or ""


def _val(lst, idx, fallback):
    try:
        v = lst[idx]
    except (IndexError, TypeError):
        return fallback
    if v is None:
        return fallback
    if isinstance(v, str) and v.strip().lower() in _EMPTY_STRINGS:
        return fallback
    return v


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off", ""):
            return False
    return None


def _looks_like_date(s: str) -> bool:
    """Heuristic: 'YYYY-MM-DD'."""
    if not isinstance(s, str):
        return False
    s2 = s.strip()
    if len(s2) < 10:
        return False
    try:
        datetime.date.fromisoformat(s2[:10])
        return True
    except (ValueError, TypeError):
        return False


def load_config(input_list):
    """
    Unpack the mixed-type AA360 input list into a clean config dict.
    Everything defaulted -> fully dynamic yet backward-safe.
    """
    if input_list is None:
        input_list = []

    cfg = {}

    cfg["TENANT_ID"] = _val(input_list, 0, _DEFAULTS.get("TENANT_ID"))
    cfg["CLIENT_ID"] = _val(input_list, 1, _DEFAULTS.get("CLIENT_ID"))
    cfg["CLIENT_SECRET"] = _val(input_list, 2, _DEFAULTS.get("CLIENT_SECRET"))
    cfg["PROXY"] = _val(input_list, 3, _DEFAULTS.get("PROXY"))
    cfg["SHAREPOINT_HOST"] = _val(input_list, 4, _DEFAULTS.get("SHAREPOINT_HOST"))
    cfg["SITE_NAME"] = _val(input_list, 5, _DEFAULTS.get("SITE_NAME"))
    cfg["LIST_NAME"] = _val(input_list, 6, _DEFAULTS.get("LIST_NAME"))

    def_perf_page = _DEFAULTS.get("PAGE_SIZE", 200)
    def_perf_timeout = _DEFAULTS.get("REQUEST_TIMEOUT", 30)
    def_perf_retries = _DEFAULTS.get("MAX_RETRIES", 3)

    try:
        cfg["PAGE_SIZE"] = int(_val(input_list, 7, def_perf_page))
    except (TypeError, ValueError):
        cfg["PAGE_SIZE"] = def_perf_page

    try:
        cfg["REQUEST_TIMEOUT"] = int(_val(input_list, 8, def_perf_timeout))
    except (TypeError, ValueError):
        cfg["REQUEST_TIMEOUT"] = def_perf_timeout

    try:
        cfg["MAX_RETRIES"] = int(_val(input_list, 9, def_perf_retries))
    except (TypeError, ValueError):
        cfg["MAX_RETRIES"] = def_perf_retries

    # Slots 10-12 — SharePoint user credentials + output folder (needed for attachment download)
    raw_email = _val(input_list, 10, None)
    raw_pass = _val(input_list, 11, None)
    raw_out = _val(input_list, 12, None)
    cfg["SHAREPOINT_EMAIL"] = raw_email.strip() if isinstance(raw_email, str) else raw_email
    cfg["SHAREPOINT_PASSWORD"] = raw_pass if raw_pass is not None else None
    cfg["OUTPUT_FOLDER_PATH"] = raw_out.strip() if isinstance(raw_out, str) else None

    # ======================================================================
    #  CLIENT-FRIENDLY VALIDATION
    # ======================================================================
    _REQ = [
        ("TENANT_ID", 0, "Azure AD Tenant ID (GUID). Find in Azure Portal -> App Registrations -> Endpoints."),
        ("CLIENT_ID", 1, "Azure AD Application (Client) ID (GUID)."),
        ("CLIENT_SECRET", 2, "Azure AD Client Secret value (create in Certificates & secrets)."),
        ("SHAREPOINT_HOST", 4, 'SharePoint host, e.g. "contoso.sharepoint.com" (no https:// needed).'),
        ("SITE_NAME", 5, 'SharePoint site path, e.g. "sites/mysite" or just "mysite".'),
        ("LIST_NAME", 6, 'List display name exactly as shown in SharePoint, e.g. "SupportTickets".'),
    ]
    missing = []
    for key, idx, hint in _REQ:
        v = cfg.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append((key, idx, hint))
    if missing:
        print()
        print("=" * 72)
        print("  CONFIG ERROR — missing required input values")
        print("=" * 72)
        print("  Fill the following slots in your input list then try again.")
        print()
        for key, idx, hint in missing:
            print("  • args[%d]  %s" % (idx, key))
            print("       %s" % hint)
            print()
        print("  Example correct input list skeleton:")
        print('    [')
        print('      "tenant-guid",         # args[0]  TENANT_ID')
        print('      "app-guid",            # args[1]  CLIENT_ID')
        print('      "secret-value",        # args[2]  CLIENT_SECRET')
        print('      "",                    # args[3]  PROXY_URL (or "")')
        print('      "contoso.sharepoint.com",  # args[4]  SHAREPOINT_HOST')
        print('      "my-site",             # args[5]  SITE_NAME')
        print('      "MyList",              # args[6]  LIST_NAME')
        print('      0, 0, 0,                # args[7-9] perf (0=default)')
        print('      "user@domain.com",     # args[10] SHAREPOINT_EMAIL (required for download)')
        print('      "<password>",          # args[11] SHAREPOINT_PASSWORD (required for download)')
        print('      r"C:\\\\Output",         # args[12] OUTPUT_FOLDER_PATH (required for download)')
        print('      \'{ "column" : "value" }\',  # args[13+] FILTER')
        print('    ]')
        print("=" * 72)
        print()
        sys.exit(1)

    # ======================================================================
    #  FILTER PARSING  — every slot from index 13 onward is read and
    #  combined with AND logic.
    # ======================================================================
    parsed_filters = []
    used_json_dict_mode = False

    if isinstance(input_list, (list, tuple)) and len(input_list) > 13:

        # FIX: diagnose EVERY filter slot for a malformed JSON dict, not just
        # index 13. Previously a typo at index 14+ was silently swallowed.
        for slot_idx in range(13, len(input_list)):
            warn = _diagnose_json_dict_issue(input_list[slot_idx], slot_idx)
            if warn:
                print()
                print(warn)
                print()

        for raw in input_list[13:]:
            # ---- skip empty / placeholder slots ----
            if raw is None:
                continue
            if isinstance(raw, str) and raw.strip().lower() in _EMPTY_STRINGS:
                continue

            f = None

            # ---- Method 1 / short-form dict: JSON dict string, e.g. '{"Status":"Active"}' ----
            if isinstance(raw, str) and _looks_like_json_dict(raw):
                extra = _json_dict_to_filters(raw)
                for ef in extra:
                    if ef and ef["column"] and ef["value"] is not None:
                        op = ef["operator"].strip().lower() if isinstance(ef["operator"], str) else "eq"
                        if op not in _ALL_OPS:
                            op = "eq"
                        ef["operator"] = op
                        parsed_filters.append(ef)
                        used_json_dict_mode = True
                continue

            # ---- Method 2: plain string, e.g. "Status=Pending", "amount>=12500" ----
            if isinstance(raw, str):
                f = _parse_filter_string(raw)

            # ---- Method 4 (or short-form dict): dict, e.g. {"column":..,"operator":..,"value":..} ----
            elif isinstance(raw, dict):
                col = (raw.get("column") or raw.get("Column")
                       or raw.get("name") or raw.get("field"))
                op = (raw.get("operator") or raw.get("op") or "eq")
                if isinstance(op, str):
                    op = op.strip().lower()
                val = raw.get("value") if "value" in raw else raw.get("Value")
                if col is None and val is None and len(raw) > 0:
                    extra = _json_dict_to_filters(raw)
                    for ef in extra:
                        op2 = (ef["operator"].strip().lower()
                               if isinstance(ef["operator"], str) else "eq")
                        if op2 not in _ALL_OPS:
                            op2 = "eq"
                        ef["operator"] = op2
                        parsed_filters.append(ef)
                        used_json_dict_mode = True
                    continue
                if col is not None and val is not None:
                    f = {"column": str(col).strip(), "operator": op, "value": val}

            # ---- Method 3: list/tuple, e.g. ["amount","ge",12500] or ["Country","India"] ----
            elif isinstance(raw, (list, tuple)):
                if len(raw) >= 3:
                    col = raw[0]
                    op = raw[1] if raw[1] is not None else "eq"
                    if isinstance(op, str):
                        op = op.strip().lower()
                    val = raw[2]
                    if col is not None and val is not None:
                        f = {"column": str(col).strip(), "operator": op, "value": val}
                elif len(raw) == 2:
                    col = raw[0]
                    val = raw[1]
                    if col is not None and val is not None:
                        f = {"column": str(col).strip(), "operator": "eq", "value": val}

            if f is None:
                continue
            if f["operator"] not in _ALL_OPS:
                f["operator"] = "eq"
            parsed_filters.append(f)

    cfg["FILTERS"] = parsed_filters
    cfg["_USED_JSON_DICT_MODE"] = used_json_dict_mode

    # Derived
    proxy_raw = cfg.get("PROXY")
    if isinstance(proxy_raw, str):
        p = proxy_raw.strip()
        if p:
            if "://" not in p:
                p = f"http://{p}"
            cfg["PROXIES"] = {"http": p, "https": p}
        else:
            cfg["PROXIES"] = None
    else:
        cfg["PROXIES"] = {"http": proxy_raw, "https": proxy_raw} if proxy_raw else None

    cfg["PROXIES"] = _sanitize_proxies(cfg["PROXIES"])

    return cfg


# ==============================================================================
#  TOKEN
# ==============================================================================
def get_token(cfg: dict) -> str:
    try:
        proxies = _sanitize_proxies(cfg.get("PROXIES")) or None
        app = msal.ConfidentialClientApplication(
            cfg["CLIENT_ID"],
            authority=f"https://login.microsoftonline.com/{cfg['TENANT_ID']}",
            client_credential=cfg["CLIENT_SECRET"],
            proxies=proxies,
            timeout=cfg.get("REQUEST_TIMEOUT", 30),
        )
        result = app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )
        if "access_token" in result:
            print("  [OK] Graph token acquired")
            return result["access_token"]

        err = result.get("error", "")
        desc = result.get("error_description", "")
        print(f"\n  [FAIL] Token failed: {err}")
        if desc:
            print(f"         {desc[:200]}")
        if "700016" in desc:
            print("  Hint: Check CLIENT_ID")
        if "7000215" in desc:
            print("  Hint: Check CLIENT_SECRET")
        if "90002" in desc:
            print("  Hint: Check TENANT_ID")
        if "unauthorized" in err:
            print("  Hint: Check API permissions + admin consent")
        sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("\n  [FAIL] Cannot reach login.microsoftonline.com")
        print("  Hint: Set PROXY in input list if on a corporate network")
        print("  Hint: Run in CMD -> netsh winhttp show proxy")
        sys.exit(1)


# ==============================================================================
#  HTTP GET WITH RETRY
# ==============================================================================
def get_with_retry(url: str, headers: dict, cfg: dict, params: dict = None) -> requests.Response:
    max_retries = cfg["MAX_RETRIES"]
    proxies = _sanitize_proxies(cfg.get("PROXIES"))
    timeout = cfg["REQUEST_TIMEOUT"]

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                url, headers=headers, params=params,
                proxies=proxies, timeout=timeout,
            )
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  [WAIT] Throttled - waiting {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504) and attempt < max_retries:
                print(f"  [WARN] Server error {r.status_code}, retrying ({attempt}/{max_retries})...")
                time.sleep(2 ** attempt)
                continue
            return r

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"  [WARN] Timeout, retrying ({attempt}/{max_retries})...")
                time.sleep(2)
                continue
            print("  [FAIL] Request timed out after all retries")
            sys.exit(1)

    print(f"  [FAIL] Failed after {max_retries} attempts")
    sys.exit(1)


# ==============================================================================
#  SITE ID
# ==============================================================================
def get_site_id(headers: dict, cfg: dict) -> str:
    # FIX: use the bare hostname (scheme stripped) — Graph's sites/{host}:/{path}
    # syntax 404s if given a full URL with https://.
    host = _bare_host(cfg)
    name = cfg["SITE_NAME"]
    r = get_with_retry(f"{GRAPH_BASE}/sites/{host}:/{name}", headers, cfg)
    if r.status_code == 401:
        print("  [FAIL] 401 - Sites.Read.All Application permission missing or not consented")
        sys.exit(1)
    if r.status_code == 404:
        print(f"  [FAIL] 404 - Site '{name}' not found. Check SITE_NAME in input list (index 5).")
        sys.exit(1)
    r.raise_for_status()
    site_id = r.json().get("id")
    if not site_id:
        print("  [FAIL] Site ID missing in response")
        sys.exit(1)
    print(f"  [OK] Site    : {name}")
    return site_id


# ==============================================================================
#  LIST ID
# ==============================================================================
def get_list_id(site_id: str, headers: dict, cfg: dict) -> str:
    r = get_with_retry(f"{GRAPH_BASE}/sites/{site_id}/lists", headers, cfg)
    r.raise_for_status()
    lists = r.json().get("value", [])

    if not lists:
        print("  [FAIL] No lists found on this site")
        sys.exit(1)

    target = cfg["LIST_NAME"].lower()
    available = []
    for lst in lists:
        display = lst.get("displayName", "")
        available.append(display)
        if display.lower() == target:
            print(f"  [OK] List    : {display}")
            return lst["id"]

    print(f"  [FAIL] List '{cfg['LIST_NAME']}' not found.")
    print(f"         Available lists: {', '.join(available)}")
    print(f"         Update LIST_NAME in input list (index 6) to one of the above.")
    sys.exit(1)


# ==============================================================================
#  DISCOVER ALL COLUMN INTERNAL NAMES  (dynamic - any list)
# ==============================================================================
def discover_columns(site_id: str, list_id: str, headers: dict, cfg: dict):
    all_internal = set()
    display_lookup = {}

    try:
        r = get_with_retry(
            f"{GRAPH_BASE}/sites/{site_id}/lists/{list_id}/columns",
            headers, cfg,
            params={"$top": 500},
        )
        r.raise_for_status()
        cols_data = r.json().get("value", [])
        for c in cols_data:
            name = c.get("name")
            disp = c.get("displayName") or name
            hidden = bool(c.get("isHidden", False))
            if not name:
                continue
            if hidden and name.lower() not in {"title", "created", "modified", "author", "editor", "attachments"}:
                continue
            all_internal.add(name)
            if disp:
                display_lookup[name.lower()] = disp
    except Exception as _e_meta:
        print(f"  [WARN] Columns metadata call failed ({_e_meta}), falling back to item scan")

    try:
        r_items = get_with_retry(
            f"{GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items",
            headers, cfg,
            params={"$expand": "fields", "$top": 25},
        )
        r_items.raise_for_status()
        items = r_items.json().get("value", [])
        for it in items:
            for k in (it.get("fields") or {}).keys():
                if "@odata" not in k and not k.startswith("_"):
                    all_internal.add(k)
    except Exception:
        pass

    if not all_internal:
        print("  [WARN] Could not discover any columns, using minimal set")
        return {}, ["Title"]

    col_map = {k.lower(): k for k in all_internal}

    system_lower = set(_SKIP_COLS_LOWER)
    system_lower.update({k.lower() for k in all_internal if k.startswith("_")})
    system_lower.update({k.lower() for k in all_internal if "@odata" in k})

    ordered = []
    if "title" in col_map:
        ordered.append(col_map["title"])

    user_only = []
    for k in all_internal:
        kl = k.lower()
        if kl in system_lower:
            continue
        if kl == "title":
            continue
        user_only.append(k)

    user_only.sort(key=lambda x: x.lower())
    ordered.extend(user_only)

    # Make sure "Attachments" is tracked if the list exposes it, so the
    # downloader can use the flag to skip items with none.
    if "attachments" in col_map and col_map["attachments"] not in ordered:
        ordered.append(col_map["attachments"])

    if "created" in col_map and col_map["created"] not in ordered:
        ordered.append(col_map["created"])
    if "modified" in col_map and col_map["modified"] not in ordered:
        ordered.append(col_map["modified"])

    print(f"  [OK] Columns ({len(ordered)}): {ordered}")
    return col_map, ordered


def _format_odata_value(raw_value, operator: str) -> str:
    """Escape / cast a value into its OData literal form."""
    # FIX: previously this only emitted bare true/false when raw_value was
    # ALREADY a Python bool. A string like "true" coming from a plain-text
    # filter ("IsActive=true") fell through to the string branch and got
    # wrapped in quotes ('true'), which SharePoint rejects/mis-filters on a
    # real Boolean column. Now any value that _as_bool() can confidently
    # interpret as boolean is emitted as a bare OData true/false.
    if isinstance(raw_value, bool):
        return "true" if raw_value else "false"
    if isinstance(raw_value, str) and raw_value.strip().lower() in (
        "true", "false", "yes", "no", "y", "n", "on", "off",
    ):
        b = _as_bool(raw_value)
        if b is not None:
            return "true" if b else "false"

    # Number (and not a date string)
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        return repr(raw_value)

    # String / date -> treat as quoted string with OData escaping
    s = str(raw_value)
    is_date = _looks_like_date(s)

    if is_date and operator in _COMPARISON_OPS:
        base = s.strip()[:10]
        if len(s.strip()) <= 10:
            s_out = f"{base}T00:00:00Z"
        else:
            s_out = s.strip()
    else:
        s_out = s

    escaped = s_out.replace("'", "''")
    return f"'{escaped}'"


def build_filter(col_map: dict, filters: list) -> str:
    def resolve(name: str) -> str:
        return col_map.get(str(name).strip().lower(), str(name).strip())

    clauses = []
    for f in filters:
        col_internal = resolve(f["column"])
        op = f["operator"]
        val_raw = f["value"]

        val_str = _format_odata_value(val_raw, op)
        field_accessor = f"fields/{col_internal}"

        if op in ("contains", "startswith", "endswith"):
            clauses.append(f"{op}({field_accessor}, {val_str})")
        else:
            clauses.append(f"{field_accessor} {op} {val_str}")

    return " and ".join(clauses)


# ==============================================================================
#  FETCH FILTERED ITEMS  (server-side, paginated)
# ==============================================================================
def fetch_filtered_items(site_id: str, list_id: str, headers: dict, cfg: dict, filter_str: str) -> list:
    base_url = f"{GRAPH_BASE}/sites/{site_id}/lists/{list_id}/items"
    params = {
        "$expand": "fields",
        "$top": cfg["PAGE_SIZE"],
    }
    if filter_str:
        params["$filter"] = filter_str
        preview = filter_str if len(filter_str) <= 180 else filter_str[:177] + "..."
        print(f"  [OK] Filter  : {preview}")
    else:
        print("  [INFO] No filters set - fetching all items")

    all_items = []
    url = base_url
    page = 1

    while url:
        r = get_with_retry(url, headers, cfg, params)

        if r.status_code == 400:
            print()
            print("  [FAIL] Server rejected the filter (400 Bad Request).")
            print()
            print("  Most likely cause: a filter column is not indexed in SharePoint.")
            print()
            print("  How to fix:")
            print("  1. Go to your SharePoint list")
            print("  2. Click Settings (gear) -> List Settings")
            print("  3. Click 'Indexed columns'")
            print("  4. Click 'Create a new index'")
            print("  5. Select each column you filter on -> Create")
            print("  6. Run this script again")
            print()
            print("  Direct link:")
            print(f"  https://{_bare_host(cfg)}/{cfg['SITE_NAME']}/_layouts/15/listindex.aspx")
            print()
            print("  Also possible causes:")
            print("  - Typo in column name (case-insensitive matching is used but internal names differ)")
            print("  - Wrong operator for column type (e.g. contains on a Number column)")
            sys.exit(1)

        r.raise_for_status()
        data = r.json()
        batch = data.get("value", [])
        all_items.extend(batch)
        print(f"  [OK] Page {page:<3}: {len(batch):>4} items  |  Running total: {len(all_items):>5}")
        url = data.get("@odata.nextLink")
        params = None
        page += 1

    print(f"  [OK] Done    : {len(all_items)} items fetched in total")
    return all_items


# ==============================================================================
#  PARSE / FLATTEN ITEMS  (dynamic column list)
# ==============================================================================
def parse_items(items: list, user_columns: list) -> list:
    result = []
    for item in items:
        fields = item.get("fields", {}) or {}
        item_id = fields.get("id") or item.get("id")
        if item_id is None:
            continue

        row = {"id": item_id}
        for col in user_columns:
            v = fields.get(col, "")
            if isinstance(v, str) and _looks_like_date(v) and "T" in v:
                v = v[:10]
            row[col.lower()] = v

        result.append(row)
    return result


# ==============================================================================
#  USER/PASS ATTACHMENT DOWNLOAD  (office365)
# ==============================================================================
class SharePointUserPassClient:
    """Download list-item attachments using SharePoint username + password."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.site_url = _sharepoint_site_url(cfg)
        self.site_path = urlparse(self.site_url).path.strip("/")
        self.username = cfg.get("SHAREPOINT_EMAIL") or ""
        self.password = cfg.get("SHAREPOINT_PASSWORD") or ""
        self.tenant = _tenant_domain_for_userpass(cfg)
        self.client_id = SHAREPOINT_USERPASS_CLIENT_ID
        self.list_name = cfg.get("LIST_NAME") or ""
        self._ctx = None

    def _auth(self):
        if self._ctx is None:
            self._ctx = ClientContext(self.site_url).with_username_and_password(
                tenant=self.tenant,
                client_id=self.client_id,
                username=self.username,
                password=self.password,
            )
        return self._ctx

    def attachment_folder_path(self, item_id):
        return f"/{self.site_path}/Lists/{self.list_name}/Attachments/{item_id}"

    def _list_attachments_from_folder(self, item_id):
        try:
            conn = self._auth()
            folder_url = self.attachment_folder_path(item_id)
            folder = conn.web.get_folder_by_server_relative_url(folder_url)
            folder.expand(["Files"]).get().execute_query()
        except Exception:
            return []
        attachments = []
        for file in folder.files:
            attachments.append(
                {
                    "file_name": file.name,
                    "server_relative_url": file.server_relative_url,
                }
            )
        return attachments

    def get_item_attachments(self, item_id):
        conn = self._auth()
        attachments = []
        try:
            item = conn.web.lists.get_by_title(self.list_name).get_item_by_id(int(item_id))
            item.attachment_files.get().execute_query()
            for attachment in item.attachment_files:
                attachments.append(
                    {
                        "file_name": attachment.file_name,
                        "server_relative_url": attachment.server_relative_url,
                    }
                )
        except Exception:
            attachments = []
        if attachments:
            return attachments
        return self._list_attachments_from_folder(item_id)

    @staticmethod
    def _safe_local_path(output_folder, file_name):
        """Build a safe local path; strip characters invalid on Windows."""
        name = (file_name or "attachment").strip()
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        name = name.rstrip(". ")
        if not name:
            name = "attachment"
        return os.path.join(output_folder, name)

    def download_all_attachments(self, item_id, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        attachments = self.get_item_attachments(item_id)
        if not attachments:
            return []

        conn = self._auth()
        saved_paths = []
        errors = []

        for att in attachments:
            local_path = self._safe_local_path(output_folder, att["file_name"])
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                content = File.open_binary(conn, att["server_relative_url"])
                with open(local_path, "wb") as handle:
                    handle.write(content.content)
                saved_paths.append(local_path)
            except Exception as e:
                errors.append(f"{att.get('file_name', '?')}: {e}")

        if errors and not saved_paths:
            raise RuntimeError("; ".join(errors))
        return saved_paths


def download_all_attachments_userpass(result, cfg) -> dict:
    """Download attachments for filtered items only, using SharePoint email/password."""
    output_folder = cfg.get("OUTPUT_FOLDER_PATH")
    email = cfg.get("SHAREPOINT_EMAIL")
    password = cfg.get("SHAREPOINT_PASSWORD")

    if not output_folder:
        print("  [INFO] No OUTPUT_FOLDER_PATH provided — skipping attachment download")
        print("  [INFO] Add your output folder path to args[12].")
        return {"downloaded": 0, "items_with_attachments": 0, "failed_items": 0}

    if not email or not password:
        print("  [INFO] SHAREPOINT_EMAIL / SHAREPOINT_PASSWORD not set — skipping download")
        print("  [INFO] Provide args[10] (email) and args[11] (password) to download attachments.")
        return {"downloaded": 0, "items_with_attachments": 0, "failed_items": 0}

    os.makedirs(output_folder, exist_ok=True)
    print(f"\n  Total matched items : {len(result)}")
    print(f"  Output folder       : {output_folder}")
    print("  Auth method         : SharePoint username/password (office365)")
    print(f"  Site                : {_sharepoint_site_url(cfg)}")
    print(f"  List                : {cfg['LIST_NAME']}")
    print()

    sp_client = SharePointUserPassClient(cfg)
    total_files = 0
    items_had = 0
    failed = 0

    for i, row in enumerate(result, 1):
        item_id = row["id"]

        # FIX: only skip an item when we have an explicit, confident "no
        # attachments" signal. Previously `row.get("attachments")` returning
        # None (field missing / not discovered) was NOT treated as False, so
        # this branch never actually skipped anything reliably — it always
        # fell through and issued a download attempt. We now use _as_bool()
        # so real Boolean columns (bool or "false"/"0"/etc. strings) are
        # honored, while an unknown/missing flag still attempts a download
        # (safer default: don't silently miss real attachments).
        has_attachments_flag = _as_bool(row.get("attachments"))
        if has_attachments_flag is False:
            print(f"  [{i}/{len(result)}] Item {item_id} — no attachments flag, skipping")
            continue

        item_folder = os.path.join(output_folder, f"Item_{item_id}")
        print(f"  [{i}/{len(result)}] Item {item_id} — downloading attachments...")
        try:
            saved = sp_client.download_all_attachments(item_id, item_folder)
        except Exception as e:
            failed += 1
            print(f"    [WARN] Item {item_id} failed: {e}")
            continue

        if saved:
            total_files += len(saved)
            items_had += 1
            print(f"    -> {len(saved)} file(s) saved:")
            for path in saved:
                print(f"       {path}")
        else:
            print(f"    -> no attachments found for item {item_id}")

    return {
        "downloaded": total_files,
        "items_with_attachments": items_had,
        "failed_items": failed,
    }


# ==============================================================================
#  DYNAMIC RESULT TABLE  (auto column widths)
# ==============================================================================
def print_results(result: list, user_columns: list, item_ids: list):
    print(SEP)
    if not result:
        print("  No items matched the filters.")
        print()
        print("  Tips:")
        print("  - Text values are case-sensitive - check exact spelling")
        print("  - Remove all filters (omit indices 10+) to see all items and verify data")
        print()
        print(SEP)
        return

    display_cols = ["id"]
    lower_cols = [c.lower() for c in user_columns]
    if "title" in lower_cols:
        title_key = user_columns[lower_cols.index("title")]
        display_cols.append(title_key.lower())

    for col in user_columns:
        cl = col.lower()
        if cl in display_cols:
            continue
        display_cols.append(cl)

    widths = {}
    for dc in display_cols:
        header = "ID" if dc == "id" else dc
        widths[dc] = len(header)

    for row in result:
        for dc in display_cols:
            cell = str(row.get(dc, ""))
            widths[dc] = max(widths[dc], len(cell))

    CAP = 36
    for dc in display_cols:
        if widths[dc] > CAP:
            widths[dc] = CAP

    header_parts = []
    sep_parts = []
    for dc in display_cols:
        label = "ID" if dc == "id" else dc
        w = widths[dc]
        header_parts.append(f"  {label[:w]:<{w}}")
        sep_parts.append(f"  {'-' * w}")
    print(f"  RESULTS: {len(result)} item(s) matched\n")
    print("".join(header_parts))
    print("".join(sep_parts))

    for row in result:
        line_parts = []
        for dc in display_cols:
            w = widths[dc]
            cell = str(row.get(dc, ""))
            if len(cell) > w:
                cell = cell[:w - 1] + "…"
            line_parts.append(f"  {cell:<{w}}")
        print("".join(line_parts))

    print()
    preview_ids = item_ids if len(item_ids) <= 25 else item_ids[:25] + [f"...+{len(item_ids) - 25} more"]
    print(f"  Item IDs : {preview_ids}")
    print(SEP)


# ==============================================================================
#  MAIN  (single entry point, callable from AA360 with mixed-type list)
# ==============================================================================
def get_sharepoint_item_ids(input_list=None) -> dict:
    if input_list is None:
        input_list = []

    print()
    print(SEP)
    print("  SharePoint List Filter + Attachment Downloader")
    print(SEP)

    cfg = load_config(input_list)

    n_filters = len(cfg.get("FILTERS") or [])
    if cfg.get("_USED_JSON_DICT_MODE"):
        print(f"\n  [OK] JSON dict filter detected ({n_filters} condition{'s' if n_filters != 1 else ''}).")
        if n_filters:
            names = ", ".join(f"'{f['column']}'" for f in cfg["FILTERS"])
            print(f"       Columns to match: {names}")
    elif n_filters:
        print(f"\n  [OK] {n_filters} filter condition(s): {', '.join(repr(f['column']) for f in cfg['FILTERS'])}")
    else:
        print("\n  [INFO] No filters specified — will fetch ALL list items.")
    if cfg.get("OUTPUT_FOLDER_PATH"):
        print(f"  [OK] Output folder: {cfg['OUTPUT_FOLDER_PATH']}")
    else:
        print("  [INFO] No output folder set — attachment download will be skipped")
    if cfg.get("SHAREPOINT_EMAIL"):
        print(f"  [OK] SharePoint user: {cfg['SHAREPOINT_EMAIL']}")
    else:
        print("  [INFO] No SharePoint email (args[10]) — attachment download will be skipped")

    start = time.time()

    print("\n[1/6] Authenticating to Microsoft Graph...")
    token = get_token(cfg)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "ConsistencyLevel": "eventual",
    }

    print("\n[2/6] Locating SharePoint site...")
    site_id = get_site_id(headers, cfg)

    print("\n[3/6] Locating list...")
    list_id = get_list_id(site_id, headers, cfg)

    print("\n[4/6] Discovering columns...")
    col_map, user_columns = discover_columns(site_id, list_id, headers, cfg)

    print("\n[5/6] Fetching filtered items...")
    filter_str = build_filter(col_map, cfg["FILTERS"])
    filtered_items = fetch_filtered_items(site_id, list_id, headers, cfg, filter_str)
    result = parse_items(filtered_items, user_columns)

    elapsed = round(time.time() - start, 1)
    item_ids = [str(row["id"]) for row in result]

    print(f"\n  [OK] Completed filter in {elapsed}s")
    print()
    print_results(result, user_columns, item_ids)

    dl_summary = {"downloaded": 0, "items_with_attachments": 0, "failed_items": 0}
    if cfg.get("OUTPUT_FOLDER_PATH") and result:
        email = cfg.get("SHAREPOINT_EMAIL")
        password = cfg.get("SHAREPOINT_PASSWORD")
        if email and password:
            print(f"\n[6/6] Downloading attachments for {len(result)} filtered item(s) (user/pass)...")
            dl_summary = download_all_attachments_userpass(result, cfg)
            print(f"\n  [OK] Attachments done — {dl_summary['downloaded']} file(s) downloaded")
            if dl_summary["failed_items"]:
                print(f"  [WARN] {dl_summary['failed_items']} item(s) had errors — check logs above")
        else:
            print("\n[6/6] Attachment download — skipped (email/password not provided)")
    else:
        print("\n[6/6] Attachment download — skipped (no output folder or no matched items)")

    total_elapsed = round(time.time() - start, 1)
    print(f"\n  Total time: {total_elapsed}s")
    print()
    print(SEP)
    print("  QUICK SUMMARY")
    print(SEP)
    print(f"  Matched items    : {len(item_ids)}")
    print(f"  Item IDs         : {item_ids}")
    print(f"  Files downloaded : {dl_summary['downloaded']}")
    print(f"  Filter applied   : {filter_str or '(none)'}")
    print(f"  Time taken       : {total_elapsed}s")
    print(SEP)

    return {
        "item_ids": item_ids,
        "items": result,
        "columns": list(user_columns),
        "filter": filter_str,
        "matched": len(item_ids),
        "elapsed_s": total_elapsed,
        "downloaded": dl_summary["downloaded"],
        "download_summary": dl_summary,
    }


# ==============================================================================
#  AA360 SIMPLE ENTRY POINTS
# ==============================================================================
def get_sharepoint_item_ids_aa360(args):
    result_dict = get_sharepoint_item_ids(args)
    return result_dict["item_ids"]


def get_sharepoint_items_aa360(args):
    result_dict = get_sharepoint_item_ids(args)
    return result_dict["items"]


def get_sharepoint_download_summary_aa360(args):
    """
    Convenience entry point when the download itself (not just item IDs) is
    what you care about. Returns just the download summary dict:
    {"downloaded": int, "items_with_attachments": int, "failed_items": int}
    """
    result_dict = get_sharepoint_item_ids(args)
    return result_dict["download_summary"]


def _run_silently(fn, *a, **kw):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            return fn(*a, **kw)
    except SystemExit:
        raise
    except Exception:
        raise


def get_sharepoint_ids_only_list(args):
    return _run_silently(get_sharepoint_item_ids_aa360, args)


def get_sharepoint_ids_only_text(args):
    ids = _run_silently(get_sharepoint_item_ids_aa360, args)
    if not ids:
        return ""
    return "\n".join(str(i) for i in ids)


def get_sharepoint_download_verbose(args):
    """Run full flow and return all console output as a string (good for AA360 Message box)."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            get_sharepoint_item_ids(args)
    except SystemExit:
        pass
    return buf.getvalue()


# ==============================================================================
#  CLI DRIVER  (run standalone for testing / demo)
# ==============================================================================
if __name__ == "__main__":
    ids_only = any(a in ("--ids-only", "--ids", "--quiet", "--silent", "-q") for a in sys.argv[1:])
    ids_text = "--ids-text" in sys.argv[1:]

    demo_input = [
    "your-tenant-id-guid",              # 0  TENANT_ID       (Azure AD tenant GUID)
    "your-client-id",  # 1  CLIENT_ID       (the app ID you already have)
    "your-secret-value",   # 2  CLIENT_SECRET   (from Certificates & secrets — Value column, not Secret ID)
    None,                                # 3  PROXY           (leave None unless on corporate network)
    "yourtenant.sharepoint.com",        # 4  SHAREPOINT_HOST
    "your-site-name",                   # 5  SITE_NAME       (e.g. "sites/mysite" or "mysite")
    "YourListName",                     # 6  LIST_NAME       (exact display name in SharePoint)
    None, None, None,                   # 7-9  perf settings  (leave None for defaults)
    "your-sharepoint-email",  # 10  SHAREPOINT_EMAIL
    "your-sharepoint-password",         # 11  SHAREPOINT_PASSWORD
    r"your-outputfolder-path",  # 12  OUTPUT_FOLDER_PATH
    'your-filter-condition',           # 13  FILTER
]

    if ids_only or ids_text:
        if ids_text:
            txt = get_sharepoint_ids_only_text(demo_input)
            if txt:
                sys.stdout.write(txt + "\n")
        else:
            ids = get_sharepoint_ids_only_list(demo_input)
            if ids:
                sys.stdout.write("\n".join(str(i) for i in ids) + "\n")
        sys.exit(0)

    out = get_sharepoint_item_ids(demo_input)

    print()
    print(f"  Returned -> item_ids = {out['item_ids']}")
    print(f"  Count   : {out['matched']}")
    print(f"  Columns : {out['columns']}")
    print(f"  Filter  : {out['filter'] or '(none)'}")
    print(f"  Took    : {out['elapsed_s']}s")