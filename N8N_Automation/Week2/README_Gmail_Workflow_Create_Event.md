# Gmail Workflow & Create Event - Google Services Automation

## Purpose
This n8n workflow demonstrates a comprehensive automation that integrates multiple Google services. It automatically sends a Gmail message, creates a Google Drive folder, generates a Google Docs document, creates a Google Calendar event, and optionally creates a Google Sheet. This workflow is ideal for automating multi-step Google Workspace operations.

## Key Features
- Automated Gmail email sending
- Google Drive folder creation
- Google Docs document generation
- Google Calendar event scheduling with attendees
- Optional Google Sheets creation with timestamped naming

## Prerequisites
1. **n8n Instance**: Active n8n installation
2. **Google Cloud Console Setup**:
   - Gmail API enabled
   - Google Drive API enabled
   - Google Docs API enabled
   - Google Sheets API enabled
   - Google Calendar API enabled
3. **OAuth2 Credentials**: Separate OAuth2 credentials for each service:
   - Gmail OAuth2
   - Google Drive OAuth2
   - Google Docs OAuth2
   - Google Sheets OAuth2
   - Google Calendar OAuth2

## Step-by-Step Setup Instructions

### Step 1: Google Cloud Console Configuration

#### 1.1 Create/Select Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Sign in with your Google account
3. Create a **New Project** or select existing
4. Enter project name (e.g., "dripemaildemo")
5. Select organization (if applicable)

#### 1.2 Enable Required APIs
Navigate to **APIs & Services** → **Library** and enable:
1. **Gmail API**
2. **Google Drive API**
3. **Google Docs API**
4. **Google Sheets API**
5. **Google Calendar API**

#### 1.3 Configure OAuth Consent Screen
1. Go to **APIs & Services** → **OAuth consent screen**
2. Click **"Get Started"**
3. **App Information**:
   - App name: (e.g., "dripemaildemo")
   - User support email: Your organization email
4. **Audience**:
   - Choose **Internal** (for organizations) or **External** (for public/test users)
5. **Contact Information**:
   - Enter your email address
6. Click **"Finish"** and accept terms

#### 1.4 Create OAuth Client ID
1. Go to **Credentials** → **Create Credentials** → **OAuth client ID**
2. Application type: **Web application**
3. Name: (e.g., "dripemailschedulerdemo")
4. Click **"Create"**
5. **IMPORTANT**: Copy and save both **Client ID** and **Client Secret**
   - Note: After June 2025, these cannot be changed

### Step 2: Import Workflow
1. In n8n, click **"Workflows"** → **"+"** → **"Import from File"**
2. Select: `Gmail_Worklfow & Create Event.json`
3. Workflow nodes will appear

### Step 3: Configure Gmail Node

#### 3.1 Get Redirect URI
1. Click on **"rabdul585@gmail.com"** (Gmail node)
2. Go to **Credentials** → **Create New Credential**
3. Copy the **Redirect URL** displayed

#### 3.2 Configure Google Cloud Console
1. Return to Google Cloud Console
2. Click on your **OAuth client** name
3. Under **Authorized redirect URIs**, click **"+ Add URI"**
4. Paste the redirect URL from n8n
5. Click **"Save"**

#### 3.3 Complete n8n Configuration
1. In n8n Gmail node credentials:
   - Paste **Client ID**
   - Paste **Client Secret**
   - Click **"Connect with Google"**
   - Select your Gmail account
   - Grant all requested permissions
   - Click **"Continue"**
2. Verify **"Account connected"** appears (green checkmark)

#### 3.4 Configure Email Details
1. Click on the Gmail node
2. Set **"To"**: `rabdul585@gmail.com` (or your recipient)
3. Set **"Subject"**: `N8n_Demo`
4. Set **"Message"**: `Hi, this is your first automation`
5. Set **"Email Type"**: `text`
6. **Options** → Uncheck **"Append Attribution"** (optional)

### Step 4: Configure Google Drive Node
1. Click on **"Create folder"** node
2. **Credentials** → **Create New Credential**
3. Follow same OAuth2 setup as Gmail:
   - Copy redirect URI
   - Add to Google Cloud Console
   - Enter Client ID and Secret
   - Connect with Google
4. Configure folder settings:
   - **Name**: `N8n_Socialeagle`
   - **Drive**: `My Drive`
   - **Folder**: `/ (Root folder)`

### Step 5: Configure Google Docs Node
1. Click on **"Create a document"** node
2. **Credentials** → **Create New Credential**
3. Complete OAuth2 setup
4. Configure:
   - **Folder ID**: `1pNWqQCiDOyXapqGwEbOR2yhJ8A71m0WJ` (from Drive folder)
   - **Title**: `n8n_demo doc`

### Step 6: Configure Google Calendar Node
1. Click on **"Create an event"** node
2. **Credentials** → **Create New Credential**
3. Complete OAuth2 setup
4. Configure event:
   - **Calendar**: Select your email calendar
   - **Start**: `2025-12-06T16:17:24`
   - **End**: Set your desired end time
   - **Attendees**: 
     - `abdul.ai2807@gmail.com`
     - `abdul.ai2806@gmail.com`
     - `rabdul585@gmail.com`
   - **Description**: `meeting`
   - **Location**: `Chennai`

### Step 7: Configure Google Sheets Node (Optional)
1. Click on **"Create sheet"** node
2. **Credentials** → **Create New Credential**
3. Complete OAuth2 setup
4. The title uses a JavaScript expression for timestamped naming:
   ```javascript
   DDMMYYYY HHMMSS Testing time
   ```

### Step 8: Add Test Users (If External App)
1. In Google Cloud Console → **OAuth consent screen**
2. Go to **Test users** section
3. Click **"+ Add Users"**
4. Add email addresses that will access the app
5. Click **"Save"**

### Step 9: Verify Connections
- Check all nodes show **green connection indicators**
- All credentials should show **"Account connected"**

## How to Execute

### Method 1: Manual Trigger
1. The workflow starts with **"When clicking 'Execute workflow'"** node
2. Click the **"Execute Workflow"** button at the top
3. Workflow will execute in sequence:
   - Send Gmail → Create Drive Folder → Create Docs → Create Calendar Event

### Method 2: Schedule Trigger (Modification)
1. Replace Manual Trigger with **Schedule Trigger** node
2. Set desired schedule (daily, weekly, etc.)
3. Workflow will run automatically

### Method 3: Webhook Trigger (Modification)
1. Replace Manual Trigger with **Webhook** node
2. Copy webhook URL
3. Send POST request to trigger workflow

## Workflow Execution Flow
```
Manual Trigger
    ↓
Send Gmail Email
    ↓
Create Google Drive Folder
    ↓
Create Google Docs Document (in folder)
    ↓
Create Google Calendar Event
    ↓
[Optional] Create Google Sheet
```

## Configuration Details

### Node Details

| Node | Service | Purpose | Configuration |
|------|---------|---------|---------------|
| Manual Trigger | n8n | Starts workflow | None required |
| Gmail | Google | Send email | To, Subject, Message |
| Create folder | Drive | Create folder | Folder name, location |
| Create document | Docs | Create doc | Title, folder ID |
| Create event | Calendar | Schedule meeting | Date, attendees, location |
| Create sheet | Sheets | Create spreadsheet | Timestamped title |

### Email Configuration
- **Send To**: `rabdul585@gmail.com`
- **Subject**: `N8n_Demo`
- **Message**: `Hi, this is your first automation`

### Calendar Event Configuration
- **Start**: `2025-12-06T16:17:24`
- **Location**: Chennai
- **Attendees**: Multiple email addresses
- **Description**: Meeting details

## Expected Output
1. **Gmail**: Email sent to specified recipient
2. **Drive**: Folder "N8n_Socialeagle" created in root
3. **Docs**: Document "n8n_demo doc" created in the folder
4. **Calendar**: Event created with attendees and location
5. **Sheets**: (If connected) Spreadsheet with timestamped name

## Verification Steps
1. Check **Gmail inbox** for sent email
2. Verify **Google Drive** for new folder and document
3. Check **Google Calendar** for new event
4. Verify event invitations sent to attendees
5. Check **Google Sheets** (if enabled) for new spreadsheet

## Troubleshooting

### Issue: OAuth2 Connection Fails
- **Solution**: 
  - Verify redirect URI matches exactly (including trailing slash)
  - Ensure API is enabled in Google Cloud Console
  - Re-authenticate credentials
  - Check test users are added (for external apps)

### Issue: Folder/Document Not Created
- **Solution**:
  - Verify Drive API is enabled
  - Check folder permissions
  - Verify OAuth2 scope includes Drive write access

### Issue: Calendar Event Not Created
- **Solution**:
  - Ensure Calendar API is enabled
  - Verify date/time format is correct (ISO 8601)
  - Check calendar permissions
  - Verify attendee email addresses are valid

### Issue: Workflow Execution Fails
- **Solution**:
  - Check execution logs for specific error
  - Verify all credentials are connected
  - Ensure workflow is active (not paused)
  - Check node configurations match your requirements

## Important Notes
- **Credential Security**: Store OAuth2 credentials securely
- **Rate Limits**: Google APIs have rate limits; be mindful of execution frequency
- **Permissions**: Each service requires specific OAuth scopes
- **Time Zone**: Calendar events use workflow timezone setting (Asia/Kolkata)
- **File Paths**: Update folder IDs and file paths as needed
- **Workflow Status**: Currently set to `"active": false` - activate before use

## Customization Tips
1. **Modify Email Content**: Edit Gmail node message field
2. **Change Folder Structure**: Update folder IDs and names
3. **Adjust Calendar Settings**: Modify event times, attendees, location
4. **Add Conditional Logic**: Insert IF nodes for conditional execution
5. **Error Handling**: Add Error Trigger node for failure handling

