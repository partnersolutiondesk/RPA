# Updating SharePoint “Person or Group” Columns in Automation 360 Using the SharePoint Person Group Custom Package

An Automation Anywhere 360 custom package that writes SharePoint **Person or Group** columns through
Microsoft Graph.

**Why this package exists.** The built-in SharePoint **Update list item** action cannot write
`Person or Group` columns (together with Multiple lines of text (rich text), Lookup, Hyperlink,
Image, Location, Currency with calculation value and Managed metadata). The reason is that Microsoft
Graph does not accept a name or an email address in these columns — it needs the numeric **lookup
ID** that the site keeps in a hidden list called the _User Information List_. This package resolves
every email address, display name or group name to that ID and writes the column.

The package is self-contained: everything it needs is inside the jar.

---

## Requirements

| Item                              | Requirement                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Automation Anywhere               | Automation 360, Control Room **8750** or newer                                                                           |
| Bot agent                         | **20.11** or newer, **Windows only**                                                                                     |
| Package file                      | `SharePointPersonGroup-1.0.0.jar`                                                                                        |
| Network access from the Bot Agent | `login.microsoftonline.com` and `graph.microsoft.com` over HTTPS (port 443)                                              |
| Microsoft Entra ID                | An app registration with the Microsoft Graph **application** permission `Sites.ReadWrite.All`, and admin consent granted |

### Microsoft Entra ID app registration

Create this once per tenant, before the first bot run. An administrator is needed.

1. **Create an app registration.** Microsoft Entra ID → _App registrations_ → **New registration**.
   A single-tenant registration is enough. Note the **Application (client) ID** and the
   **Directory (tenant) ID**.
2. **Grant the Graph permission.** _API permissions_ → **Add a permission** → **Microsoft Graph** →
   **Application permissions** → add `Sites.ReadWrite.All`.
   _Application_ permissions are required, not _Delegated_: the bot runs unattended, without a
   signed-in user.
3. **Grant admin consent.** Click **Grant admin consent** for the tenant. Without it the first Graph
   call fails with HTTP 403.
4. **Create a client secret.** _Certificates & secrets_ → **New client secret**. Copy the secret
   **Value** immediately — not the _Secret ID_. Note the expiry date; an expired secret makes
   `Authenticate` fail.

No redirect URI and no other configuration is needed: the package uses the client credentials flow
(`https://graph.microsoft.com/.default`). Store the secret value in a **Credential** and select that
credential in `Authenticate`. The secret and the access token are never written to the log.

---

## Installation

1. Log in to the Control Room as an administrator.
2. Go to **Manage > Packages**.
3. Click **Upload packages**, click **Browse** and select `SharePointPersonGroup-1.0.0.jar`.
4. Click **Upload** and wait for the confirmation message.
5. If the package is not visible in the bot editor afterwards, open it in the Packages list and set it
   as the default version.

---

## Before you run the bot: add the users and groups to the site

Every user or group you want to put in a Person or Group column must first be added to the SharePoint
site, so that it exists in the site's **User Information List**. A user or group that was never added
cannot be resolved and the update will not work. Add the user first, then the group, and only then run
the bot.

1. Open the SharePoint site (the same site used in the bot).
2. Click the **Gear** icon (Settings) at the top right.
3. Click **Site permissions**.
4. Click **Advanced permissions settings** (shown as **Additional permission settings** in some
   tenants).
5. Select the required role / permission group and click **Grant Permissions**.
6. Enter the user or group (create the user) and click **Share / Grant** to save.
7. Repeat for every user and group that the bot will use.

Use the same user or group names (email address, display name or group name) in the bot as in
SharePoint.

---

## Actions

The package contains three actions:

| Action                           | Purpose                                                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Authenticate                  | Starts a SharePoint session. Gets a Microsoft Graph token (client credentials flow) and resolves the site. The token is refreshed automatically before it expires. |
| 2. Update Person or Group column | Updates one or more Person or Group columns of a single list item.                                                                                                 |
| 3. End session                   | Closes the SharePoint session so the next Authenticate starts from a fresh token.                                                                                  |

A bot always uses them in this order:

```
Authenticate  →  Update Person or Group column  →  End session
```

All actions after `Authenticate` receive the session through their **Session name** input.

### 1. Authenticate

Gets the Graph token with the app registration credentials and resolves the site, so that a wrong
secret, a missing admin consent or a wrong site name fails in this action instead of in the middle of
an update.

| #   | Field           | What to enter                                                                                                                                                                                        | Required |
| --- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1   | Tenant ID       | Directory (tenant) ID of the app registration in Microsoft Entra ID                                                                                                                                  | Yes      |
| 2   | Client ID       | Application (client) ID of the app registration in Microsoft Entra ID                                                                                                                                | Yes      |
| 3   | Client secret   | Select the credential holding the secret (recommended). You can also use a variable or type the value directly. Use the secret **VALUE**, not the secret ID. The secret is never written to the log. | Yes      |
| 4   | SharePoint host | Host name only, for example `yourcompany.sharepoint.com`. A pasted `https://` address is also accepted.                                                                                              | Yes      |
| 5   | Site name       | Leave empty to use the root site. Otherwise enter the part of the site URL after `/sites/`, for example `Finance`.                                                                                   | Optional |
| 6   | Session name    | Local session (default name `Default`), Global session or a variable. Use the same name in the following actions.                                                                                    | Yes      |

### 2. Update Person or Group column

Writes one or more Person or Group columns of a single list item.

| #   | Field                     | What to enter                                                                                                                                                                                          | Required |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| 1   | Session name              | The same session name given in Authenticate (default: `Default`)                                                                                                                                       | Yes      |
| 2   | List name                 | Display name or internal name of the SharePoint list                                                                                                                                                   | Yes      |
| 3   | Item selection            | **By item ID**: enter the ID of the list item. Or **By column match**: enter the Match column (display or internal name) and the Match value. Exactly one item must match, otherwise the action fails. | Yes      |
| 4   | Person or Group updates   | Dictionary. Key = column display name (or internal name). Value = one email address, display name or group name, or several values passed as a List (see _Passing multiple users_).                    | Yes      |
| 5   | Assign output to variable | Variable that receives the result of the action                                                                                                                                                        | Optional |

**How the value is interpreted**

| Value               | Behaviour                                                                                                       |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| Contains `@`        | Looked up as an **email address** in the site's User Information List.                                          |
| Anything else       | Looked up as a **name** — the display name of a user, or the name of a SharePoint group such as `Support Team`. |
| A **List**          | Every item of the list is resolved and written. Use this for columns that allow multiple selections.            |
| A single text value | Exactly **one** person. Text is not split on `;` or `,`.                                                        |

**Rules**

- A column that allows **multiple** selections is written as a collection of lookup IDs.
- A column that allows **only one** selection must receive exactly one value, otherwise the action
  fails with _"Column '…' allows only one person or group, but N values were provided."_
- The column must be a real Person or Group column.
- Entries with an empty column name or an empty value are skipped. If every entry is skipped the
  action fails with _"No Person or Group updates were provided."_
- The action returns the **ID of the updated item** as text.

### 3. End session

| #   | Field                         | What to enter                                              | Required |
| --- | ----------------------------- | ---------------------------------------------------------- | -------- |
| 1   | Session name                  | The session name used in Authenticate (default: `Default`) | Optional |
| 2   | Assign the result to variable | Variable that receives the result                          | Optional |

Closes the session and removes it, so the next `Authenticate` starts from a fresh token. Always close
the session at the end of the bot.

---

## How the item is found

| Selection           | Behaviour                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **By item ID**      | Uses the ID you enter. A wrong ID fails while writing.                                                             |
| **By column match** | Reads the column you name and looks for items whose value equals the Match value. **Exactly one** item must match. |

- **No match** → _"No item was found where \<column\> = '\<value\>'."_
- **More than one match** → the action fails and **nothing is updated**, listing the matching IDs so
  you can make the value more specific or switch to _By item ID_.

List names and column names are matched case-insensitively against both the display name and the
internal name; the first match wins.

---

## Passing multiple users – using a List

For a column that allows multiple people (for example **Members**), pass the values as a list. There
are two ways to do this.

**Option A — enter the list values in the action**

1. In **Person or Group updates**, select **Dictionary**.
2. Add a Field, set **Type = List** and **Key =** the column name (for example `Members`).
3. Under **Value**, keep **List** selected and add one **Item** (Type = String) for each user, for
   example Arun, Vani and Vaibhav.

**Option B — use a List variable (recommended for dynamic data)**

1. Create a **List** variable (for example `MembersList`) and fill it in the bot, or read it from
   Excel, CSV or a database.
2. In **Person or Group updates**, select **Dictionary**, add a Field with **Type = List** and
   **Key =** the column name.
3. Under **Value**, select **Variable** and enter the list variable, for example `$MembersList$`.

Both options give the same result. Option B keeps the bot clean when the number of users changes at
run time.

---

## Recommended bot flow

| Order | Action                                                 | Notes                                                |
| ----- | ------------------------------------------------------ | ---------------------------------------------------- |
| 1     | SharePoint Person Group: Authenticate                  | Opens the session (for example `Default`)            |
| 2     | SharePoint Person Group: Update Person or Group column | Add one action for each list item you need to update |
| 3     | SharePoint Person Group: End session                   | Always close the session at the end                  |

**Example**

```
1) SharePoint Person Group: Authenticate
     Tenant ID        : <your-tenant-id>
     Client ID        : <your-client-id>
     Client secret    : credential "SP-APP-SECRET"
     SharePoint host  : contoso.sharepoint.com
     Site name        : Finance            (empty = root site)
   -> Session name = Default

2) SharePoint Person Group: Update Person or Group column
     Session name     : Default
     List name        : Requests
     Item selection   : By column match
     Match column     : Title
     Match value      : REQ-1042
     Person or Group updates:
         Assigned To  -> deva.kumar@contoso.com
         Approvers    -> [priya.sharma@contoso.com, Support Team]   (Type = List)
   -> updated item ID = 42

3) SharePoint Person Group: End session
     Session name     : Default
```

---

## Troubleshooting

| Symptom                                            | What to check                                                                                                                                 |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Authentication fails                               | Confirm the Tenant ID and the Client ID. Make sure the secret **VALUE** (not the secret ID) is used and has not expired.                      |
| _"Microsoft Graph denied the request (HTTP 403)…"_ | The Graph **application** permission is missing, or admin consent was not granted.                                                            |
| _"SharePoint site '…' was not found…"_             | Wrong host or wrong site name.                                                                                                                |
| User or group is not found / not updated           | The user or group is not yet added to the site. Repeat _Before you run the bot_. Check the spelling of the email, display name or group name. |
| _"Exactly one item must match" type failure_       | With **By column match**, the Match value must identify exactly one item. Use **By item ID** or a more unique column.                         |
| Multiple values rejected                           | The column allows only a single selection. Pass exactly one value, or allow multiple selections in the column settings.                       |
| _"Column '…' was not found in list '…'."_          | Check the spelling of the column name (display or internal name).                                                                             |
| _"This session is closed…"_                        | `End session` ran before this action, or the bot used a session variable that `Authenticate` never set.                                       |
| Action not visible in the bot editor               | Check that the package was uploaded successfully and is the default version. Confirm bot agent version 20.11 or above.                        |
| Wrong column name                                  | Use the column display name or internal name exactly as in SharePoint.                                                                        |

---

## Notes and limitations

- **Site layout.** The site must be addressed as `/sites/<name>` (the standard layout). A site under a
  different path, for example `/site/<name>`, is **not** supported by this version.
- **People must exist in the site's User Information List** (see _Before you run the bot_).
- Prefer **email addresses** over display names: they are unambiguous.
- The token is refreshed automatically shortly before it expires, so long bots do not fail mid-run.
- Graph paging is followed automatically; requests time out after 120 seconds; an HTTP `429` or `503`
  answer is retried once after the `Retry-After` delay.

---

A step-by-step walkthrough with screenshots is available in the companion knowledge base article
`KB_SharePoint_Person_or_Group_Column_Update.docx`.
