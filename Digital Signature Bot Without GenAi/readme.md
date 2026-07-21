# Digital Signature Detection Bot (Without GenAI)

This bot determines whether a digital signature is present in a document processed through Document Extraction without using any generative AI.

It works by inspecting the JSON output produced by the Document Extraction action and checking for a SIGNATURE block in the extracted feature objects.

## What this bot does

- Processes documents through Document Extraction
- Reads the generated JSON output
- Checks whether a SIGNATURE block is present in the extracted feature objects
- Returns a result indicating whether a digital signature was detected

## Prerequisites

Before deploying this bot, ensure the following are in place:

- Bot creator license
- The Document Extraction action configured to download extracted contents in JSON format
- A local or shared file path accessible to the bot runner for JSON storage

## Bot runner prerequisites

The bot runner must have:

- Access to the Automation Anywhere environment
- Permission to read the input document and write to the configured JSON storage path
- The required bot package imported and available in the runtime environment

## How to run this bot

1. Open Automation Anywhere Control Room.
2. Import the bot package file: signatureDetection_withoutGENAI_20260720_034627.dw.
3. Ensure the Document Extraction action is configured to output extracted contents in JSON format.
4. Provide a valid local or shared folder path where the JSON output will be stored.
5. Place the document to be processed in the input location configured for the bot.
6. Run the bot.
7. Review the output result to confirm whether a digital signature was detected.




## Notes

- This solution uses rule-based logic and JSON inspection rather than Generative AI.
- The detection depends on the presence of the SIGNATURE block in the extracted feature objects from Document Extraction.
- Ensure the JSON output path is accessible to the bot runner during execution.
