# AI Agent Work Gmail - Birthday Email Automation

## Purpose
This n8n workflow creates an AI-powered chatbot that automatically generates warm, personalized birthday emails and sends them via Gmail. Users can interact with the chatbot through a chat interface, providing recipient information, and the AI agent will craft and send appropriate birthday greetings.

## Key Features
- Interactive chat interface for birthday email requests
- AI-powered email content generation (warm, professional, 80-150 words)
- Automatic Gmail sending via OAuth2
- Context-aware personalization based on relationship and preferences
- Conversation memory for follow-up interactions

## Prerequisites
1. **n8n Instance**: Active n8n installation (cloud or self-hosted)
2. **OpenAI API Key**: Access to OpenAI API (model: gpt-4.1-nano)
3. **Gmail OAuth2 Credentials**:
   - Gmail API enabled in Google Cloud Console
   - OAuth2 Client ID and Client Secret
   - Authorized redirect URI configured

## Step-by-Step Setup Instructions

### Step 1: Import the Workflow
1. Open your n8n instance
2. Click on **"Workflows"** in the left sidebar
3. Click the **"+"** button to create a new workflow
4. Click on the three dots (⋮) and select **"Import from File"**
5. Select the file: `AI Agent work_Gmail.json`
6. The workflow will be imported with all nodes

### Step 2: Configure OpenAI Credentials
1. Click on the **"OpenAI Chat Model"** node
2. In the credentials section, click **"Create New Credential"** or select existing
3. Enter your **OpenAI API Key**
4. Test the connection
5. Ensure the model is set to **"gpt-4.1-nano"**

### Step 3: Configure Gmail OAuth2
1. **Set up Google Cloud Console** (if not already done):
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing
   - Enable **Gmail API**
   - Go to **Credentials** → **Create Credentials** → **OAuth client ID**
   - Configure consent screen (if first time)
   - Application type: **Web application**
   - Copy **Client ID** and **Client Secret**

2. **Configure in n8n**:
   - Click on **"Send a message in Gmail"** node
   - Click **"Create New Credential"** or select existing
   - Copy the **Redirect URL** displayed in n8n
   - Go back to Google Cloud Console → Your OAuth client → **Authorized redirect URIs**
   - Add the redirect URL from n8n
   - Paste **Client ID** and **Client Secret** in n8n credential form
   - Click **"Connect with Google"** and authorize access
   - Select your Gmail account and grant permissions

### Step 4: Configure AI Agent System Message
1. Click on the **"AI Agent"** node
2. Review the system message in **Options** → **System Message**
3. The system message instructs the AI to:
   - Create warm, friendly birthday emails
   - Use appropriate tone (warm, positive, professional)
   - Keep emails concise (80-150 words)
   - Use the Gmail tool to send emails

### Step 5: Activate the Workflow
1. Toggle the **"Active"** switch at the top right of the workflow
2. The workflow is now live and ready to receive chat messages

## How to Execute

### Method 1: Using Chat Interface
1. Once activated, click on the **"When chat message received"** node
2. Copy the **Chat URL** or use the inline chat interface
3. Open the chat interface in your browser
4. Start a conversation:
   - Example: *"Hi, I need to send a birthday email to john.doe@example.com. His name is John Doe, and he's a colleague."*
5. The AI agent will:
   - Generate a personalized birthday email
   - Use the Gmail tool to send it automatically
   - Confirm the email has been sent

### Method 2: Using API/Webhook
1. The workflow creates a webhook endpoint automatically
2. Use the webhook URL to send POST requests with message data
3. Format: Send JSON with message content

## Workflow Structure
```
Chat Trigger → AI Agent → Gmail Tool
                ↓
        OpenAI Chat Model
        Simple Memory
```

## Configuration Details

### AI Agent Node
- **Type**: LangChain Agent
- **Prompt**: Processes user input for birthday email requests
- **System Message**: Defines email creation rules and tone
- **Tools Available**: Gmail send tool

### OpenAI Chat Model Node
- **Model**: gpt-4.1-nano
- **Purpose**: Powers the AI agent's language understanding and generation

### Simple Memory Node
- **Purpose**: Maintains conversation context
- **Type**: Buffer Window Memory
- Allows the agent to remember previous messages in the conversation

### Gmail Tool Node
- **Operation**: Send message
- **Integration**: n8n Gmail Tool (AI-tool enabled)
- **OAuth2**: Required for authentication

## Expected Output
1. User provides birthday email details via chat
2. AI agent generates personalized email content
3. Email is automatically sent via Gmail
4. Confirmation message returned to user in chat

## Example Usage

**User Input:**
```
Send a birthday email to sarah@example.com. Her name is Sarah, 
she's a friend, and I'd like a warm, casual tone.
```

**AI Agent Response:**
```
Thought: I'll create a warm, friendly birthday email for Sarah 
with a casual tone since she's a friend.

Action: [Email sent via Gmail tool]

I've sent a warm birthday email to Sarah! The email has been 
delivered to sarah@example.com with an appropriate subject line.
```

## Troubleshooting

### Issue: Gmail authentication fails
- **Solution**: Re-authenticate OAuth2 credentials
- Ensure redirect URI matches exactly in Google Cloud Console
- Check that Gmail API is enabled

### Issue: AI agent doesn't send email
- **Solution**: Check system message includes Gmail tool instructions
- Verify Gmail tool is connected to AI Agent node
- Review execution logs for errors

### Issue: Email content is not appropriate
- **Solution**: Adjust system message in AI Agent node
- Modify tone and length requirements
- Test with different examples

## Notes
- The workflow uses the ReAct pattern for AI tool calling
- Email content is automatically generated; no templates are hardcoded
- Memory is session-based (conversation history maintained during chat)
- Workflow must be active to receive chat messages

