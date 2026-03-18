# Utility to Trace Packages and Commands Usage Across A360 TaskBots


This repository provides a utility for tracing Packages and Commands Usage Across A360 TaskBots within a specified folder. Follow the steps below to utilize this Utility effectively.

## Steps to Follow

1. **IMPORT THE BOT UTILITY TO YOUR CONTROL ROOM**
   - Download the exported copy of the utility (Its an A360 Bot - `Export.TracePackagesnCommands.sikha_creator.zip`) into your control room.
   - You can find the Utility bot and dependent files in "**TracePackagesnCommands**" folder
   - - This utility consists of :
       - **TracePackagesnCommands** task bot – Responsible for generating a report based on input data.
       - and a dependent Python script file  **TracePackagesnCommandsScript.py** as well.

2. **MODIFY THE INPUT FILE**
   - Download the **input.xlsx** file from this repository. 
   - Open the `input.xlsx` file and navigate to the "Inputs" sheet.
   - Fill in the required Packages and Commands.
   - You should keep this updated Input.xlsx file in the InputOutputFolder specified while running the Bots.
   - If you're only interested in knowing **which automations use certain packages**:
      - Just include a single column in your input sheet named **"Package Name"**. This will help the utility scan and report all automations that reference those packages.
      - <img width="299" height="128" alt="image" src="https://github.com/user-attachments/assets/c828b6ff-6f0d-442e-9525-dcc16261f44d" />


   - If you want to find out **only some specific commands within those packages** are being used:
        - Then your input sheet needs two columns:
             - One for **"Package Name"**
             - Another for **"Command Name"**
             - <img width="300" height="125" alt="image" src="https://github.com/user-attachments/assets/8b4b8472-8750-4d40-9983-038339d60e33" />

        - This allows the utility to trace command-level usage across automations.
    
          
 

4. **RETRIEVE COMMAND INFORMATION**
   - To obtain the Command name, Package name, Attribute name, and Attribute value:
     1. Create a new bot and drag and drop the command listed in the input Excel file.
     2. Fill in the required details with placeholder data.
     3. Inspect the page by pressing `F12` and navigate to the Network tab.
     4. Search for the API endpoint:
        ```
        <CR_URL>/v2/repository/files/<Bot_id>/content?hasErrors=false
        ```
     5. Check the request payload and the `nodes` object; you will find the command name, package name there. Use it as per your requirement in the input.xlsx file
     6.   ![image](https://github.com/user-attachments/assets/8190475f-10bc-4b82-b779-0b52532e30a0)


5. **GENERATE PRE-UPDATE REPORT/ RUN THE BulkBotCommandsUpdater_V2_REPORT BOT**

   - Please install the python modules required to run this bot to your environment using the following commmand
     ```
     pip install openpyxl pandas requests requests-toolbelt
     ```
   - Execute the **`TracePackagesnCommands`** bot to generate a report of the bots in the specified folder, which includes the given packages and commands.
   - Bot Input details :
   -    ![image](https://github.com/user-attachments/assets/472687da-e17e-418a-b2a6-b25bfbeafd33)

  
   - ![image](https://github.com/user-attachments/assets/37b2ce20-12b2-4b93-a1c7-5754b271b576)


   - This Utility will work on the Public folder.Please provide your Public folder ID as the input.  So no need to check out the bots before running this Utility bots. Keep all your bots in the public folder and give the public folder ID. Refer the image given below for the public folder ID sample
   -    ![image](https://github.com/user-attachments/assets/873bd31b-f348-47ec-839e-f2b2509216fb)


   - This will add two additional sheets to the `Input.xlsx` file:  "Report" and "Bot_IDs".
   -    <img width="959" height="423" alt="image" src="https://github.com/user-attachments/assets/d6f01a1e-8ad4-4f42-b552-a77b33a868c4" />

    - <img width="262" height="424" alt="image" src="https://github.com/user-attachments/assets/9438d12c-b0d4-4e25-ab1f-4dc745768377" />


   - 

   - A log file will be generated in the `inputOutputFolder`, named `packagescanner.log`.

## Note

Please make sure that the best practices are followed, and changes are done on development server before promoting to the production environment.




