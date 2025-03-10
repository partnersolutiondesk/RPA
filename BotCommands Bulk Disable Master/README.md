# BotCommands Bulk Disable Master

This repository provides a utility for disabling multiple bot commands in bots within a specified public folder. Follow the steps below to utilize the BotCommands Bulk Disable Master effectively.

## Steps to Follow

1. **Import the ZIP File**
   - Import the file `Export.BotCommandsBulkDisableMaster.sikha_creator.zip` into your control room.

2. **Modify Input File**
   - Open the `Input.xlsx` file and navigate to the "updates" sheet.
   - Fill in the required commands and attributes.

   **Example:**
   ![image](https://github.com/user-attachments/assets/88f49039-f688-4463-b7b1-fb59ad4b3dc3)



3. **Retrieve Command Information**
   - To obtain the Command name, Package name value:
     1. Create a new bot and drag and drop the command listed in the input Excel file.
     2. Fill in the required details with placeholder data.
     3. Inspect the page by pressing `F12` and navigate to the Network tab.
     4. Search for the API endpoint:
        ```
        <CR_URL>/v2/repository/files/<Bot_id>/content?hasErrors=false
        ```
     5. Check the request payload and the `nodes` object; you will find the command name, package name there.
     6.   ![image](https://github.com/user-attachments/assets/8190475f-10bc-4b82-b779-0b52532e30a0)
     7.   Note:  same API can be found if you refresh an existing bot (which has all the required commands) edit page. 


4. **Generate Pre-Update Report**
   - Execute the `BulkDisableCommands-Report` bot to generate a report of the bots in the specified public folder, which includes the given commands.

   - This Utility will work on the Public folder.Please provide your Public folder ID as the input.  So no need to check out the bots before running this Utility bots. Keep all your bots in the public folder and give the public folder ID. Refer the image given below for the public folder ID sample
   -    ![image](https://github.com/user-attachments/assets/873bd31b-f348-47ec-839e-f2b2509216fb)


   - This will add two additional sheets to the `Input.xlsx` file: "Bot_IDs" and "Pre-update-report".These details are necessary for executing the Bot that disables these commands. BulkBotCommandDisableMaster bot. 


5. **Run the BulkBotCommandDisableMaster Bot**
   - Next, run the `BulkBotCommandDisableMaster` bot to disable the commands the bots identified via the 'BulkDisableCommands-Report' bot..
   - Ensure you specify the folder containing the updated `Input.xlsx` file (which now has three sheets. "Updates", "Bot_IDs" and "Pre-update-report").
   - **Before**
   -    ![image](https://github.com/user-attachments/assets/2063599d-803f-4701-811e-6981015d95e0)

   - **After**
   -    
   -    ![image](https://github.com/user-attachments/assets/02a0fefe-b6ad-49dc-b2b1-2e0c4d1efe6b)


6. **Review Update Status**
   - The bot will disable the commands within the bots and create a `Post_update_report` sheet for you to review the update status.
   - A log file will be generated in the `inputOutputFolder`, named `botCommandsBulkDisable.log`.

## Note

Please make sure that the best practices are followed, and changes are done on development server before promoting to the production environment.




