# CredToString - Automation Anywhere 360 Custom Package

A custom bot command package for Automation Anywhere 360 that converts a Credential (SecureString) variable to a plain String value.

## Overview

This package provides a single action that allows bot developers to extract the underlying string value from a secured credential variable in Automation Anywhere Control Room. This is useful when you need to pass credential values to external systems, APIs, or scripts that expect plain text inputs.

## Project Structure

```
CredToString/
├── Src/
│   └── CredToString.java       # Source code for the bot command
└── Jar/
    └── CredToString-2.7.0.jar  # Compiled package ready for upload
```

## Installation

1. Open Automation Anywhere Control Room
2. Navigate to **Packages** > **Add Package**
3. Upload the `CredToString-2.7.0.jar` file from the `Jar/` directory
4. Once uploaded, the package will be available in the Bot Editor action palette

## Usage

### Input Parameters

| Parameter | Type       | Required | Description                                  |
| --------- | ---------- | -------- | -------------------------------------------- |
| Input     | Credential | Yes      | The credential variable to convert to string |

### Output

| Output | Type   | Description                            |
| ------ | ------ | -------------------------------------- |
| Result | String | The plain text value of the credential |

### Example Workflow

1. Create a credential in Control Room Credential Vault (e.g., `API_Token`)
2. In your bot, drag the **CredToString** action from the package
3. Select your credential variable as the input
4. Assign the output to a String variable (e.g., `$tokenString$`)
5. Use the string variable in subsequent actions (HTTP requests, scripts, etc.)

## Building from Source

The source code is located at `Src/CredToString.java`. To rebuild:

1. Set up the Automation Anywhere SDK development environment
2. Compile using the AA SDK build tools
3. The resulting JAR can be uploaded to Control Room

## Requirements

- Automation Anywhere 360 Control Room
- Java SDK for custom package development (if building from source)
- Appropriate permissions to upload packages in Control Room
