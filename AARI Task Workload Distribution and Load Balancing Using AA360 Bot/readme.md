# Task Bot - Load-Balanced Request Assignment
 
## Overview
 
The Task Bot is an automation bot that intelligently distributes incoming requests among team members based on their current workload. It implements a **least-loaded assignment** strategy to ensure balanced work distribution across the team.
 
## Features
 
- **Automated Request Authentication**: Validates credentials and obtains session tokens
- **Process-Based Request Retrieval**: Fetches open requests by process name
- **Workload Analysis**: Tracks active request counts for each team member
- **Intelligent Load Balancing**: Assigns unassigned tasks to the team member with the fewest active requests
- **Existing Assignment Preservation**: Skips reassignment for tasks already owned by a member
- **Batch Processing**: Handles multiple process IDs in a single execution
## How It Works
 
### Process Flow
 
The bot executes the following steps in sequence:
 
1. **Authenticate & Fetch Process Details**
   - Accepts Process ID and authentication credentials from customer request
   - Exchanges credentials for an Auth Token
   - Retrieves Process Name and Default Team ID using the token
2. **Query Open Requests**
   - Fetches all requests with status `OPENED` using the Process Name
   - Collects Ref-IDs for each open request
3. **Check Task Assignment Status**
   - For each open request, queries human tasks with status `INPROGRESS`
   - Determines whether the task already has an assigned owner
4. **Retrieve Team Members & Workload**
   - Fetches team members belonging to the Default Team ID
   - Gets current request count for each team member
   - Builds a workload map (member → request count)
5. **Assign to Least-Loaded Member**
   - Identifies the team member with the minimum request count
   - Assigns the unassigned request to that member
   - Increments that member's request count
   - Repeats for all open requests
### Load-Balancing Rule
 
**Least Active Requests Wins**: The bot assigns each task to the team member currently handling the fewest requests.
 
**Tie-Breaker**: When two or more members have equal workloads, use:
- Lowest member ID, OR
- First in the list
Apply your chosen tie-breaker consistently throughout execution.
 
## Implementation Details
 
### Key Variables
 
| Variable | Purpose | Example |
|----------|---------|---------|
| `$authToken$` | Session token for API calls | JWT or session ID |
| `$processName$` | Name of the process | "CustomerSupport" |
| `$defaultTeamId$` | Team ID for assignment | "TEAM-12345" |
| `$openRequests$` | List of open request Ref-IDs | ["REQ-001", "REQ-002"] |
| `$memberCounts$` | Dict of member IDs → request counts | {"MEM-01": 2, "MEM-02": 1} |
 
### Core Logic Steps
 
#### Step 1: Authentication
```
Input: Process-ID, Username, Password
├─ Call authentication service
├─ Capture Auth Token
└─ Validate token before proceeding
```
 
#### Step 2: Fetch Process Details
```
Input: Auth Token, Process-ID
├─ Query process details
├─ Extract Process Name → $processName$
└─ Extract Team ID → $defaultTeamId$
```
 
#### Step 3: Query Open Requests
```
Input: Process Name, Status = "OPENED"
├─ Retrieve all matching requests
├─ Collect Ref-IDs → $openRequests$
└─ Exit if list is empty
```
 
#### Step 4: Loop Through Open Requests
```
For each Ref-ID in $openRequests$:
├─ Query human task with Status = "INPROGRESS"
├─ Check if task is assigned:
│  ├─ If YES → Skip to next request (keep existing owner)
│  └─ If NO → Continue to assignment logic
└─ Repeat until all requests processed
```
 
#### Step 5: Build Member Workload Map
```
For each team member in $defaultTeamId$:
├─ Create unique user key
├─ Query current active request count
└─ Store in dictionary: $memberCounts$ = {MemberID: Count, ...}
```
 
#### Step 6: Assign to Least-Loaded Member
```
1. Sort $memberCounts$ by request count (ascending)
2. Select member with minimum count
3. Assign current request to that member
4. Increment member's count by 1 in $memberCounts$
5. Continue to next unassigned request
```
 
## Setup Instructions
 
### Prerequisites
 
- Access to your automation platform (e.g., Automation 360)
- REST API credentials for your system
- Process ID and authentication details
- Read/Write permissions for requests and human tasks
### Creating the Bot
 
1. **Open Automation 360 Control Room**
   - Navigate to **Automation** → **Create**
   - Select **Task Bot**
   - Enter bot name: "Task Bot - Load Balanced Request Assignment"
2. **Add Actions in Sequence**
   - Step 1: REST Web Services call for authentication
   - Step 2: REST call to fetch process details
   - Step 3: Query/REST call for open requests
   - Step 4: Loop action for each Ref-ID
   - Step 5: Query for team members and counts
   - Step 6: Assignment logic with load-balancer selection
3. **Wire Variables**
   - Connect output of each step to input of next step
   - Use naming convention: `$stepName_Output$`
4. **Add Error Handling**
   - Wrap authentication action in Try-Catch
   - Add validation for token and Ref-ID lists
   - Log errors without stopping entire run
5. **Test Execution**
   - Use process with mixed assigned/unassigned tasks
   - Verify new work assigns to least-loaded member
   - Confirm existing assignments are preserved
## Input & Output
 
### Input Parameters
 
| Parameter | Type | Description |
|-----------|------|-------------|
| Process-ID | String | Identifier for the process |
| Username | String | Authentication username |
| Password | String | Authentication password |
 
### Output Results
 
- **Assignment Status**: Success or failure for each request
- **Updated Member Counts**: Final workload distribution
- **Audit Log**: Record of which member received each request
