# My Workflow - Advanced Telegram AI Assistant

## Purpose
This n8n workflow creates an intelligent Telegram bot that handles both text and voice messages. It features AI-powered responses, Google search capabilities, Gmail integration, speech-to-text (STT), and text-to-speech (TTS) functionality. The bot maintains conversation context and can perform web searches and send emails on behalf of users.

## Key Features
- **Dual Input Support**: Handles both text and voice messages
- **AI-Powered Responses**: Uses OpenAI GPT-4.1-mini for intelligent conversations
- **Voice Processing**: 
  - Speech-to-Text (STT) for voice messages
  - Text-to-Speech (TTS) for audio responses
- **Web Search Integration**: Google search via SerpAPI
- **Gmail Integration**: Send emails through the bot
- **Conversation Memory**: Maintains context across messages
- **Smart Routing**: Automatically routes voice vs text messages

## Prerequisites
1. **n8n Instance**: Active n8n installation
2. **Telegram Bot**:
   - Bot created via [@BotFather](https://t.me/botfather)
   - Bot token obtained
3. **OpenAI API Key**: 
   - Access to OpenAI API
   - Models: gpt-4.1-mini (agent), gpt-4o-mini (TTS summarization)
4. **SerpAPI Key**: For Google search functionality
5. **Gmail OAuth2**: For email sending capabilities

## Step-by-Step Setup Instructions

### Step 1: Create Telegram Bot
1. Open Telegram and search for **[@BotFather](https://t.me/botfather)**
2. Send `/newbot` command
3. Follow instructions:
   - Choose a name for your bot (e.g., "SocialEagleTest Bot")
   - Choose a username (must end in "bot", e.g., "socialeagletest_bot")
4. **Save the Bot Token** provided by BotFather
5. Configure bot settings (optional):
   - `/setdescription` - Add bot description
   - `/setabouttext` - Add about text
   - `/setuserpic` - Set bot profile picture

### Step 2: Import Workflow
1. In n8n, go to **Workflows** → **"+"** → **Import from File**
2. Select: `My workflow.json`
3. Workflow will be imported with all nodes

### Step 3: Configure Telegram Trigger
1. Click on **"Telegram Trigger"** node
2. **Credentials** → **Create New Credential**
3. Enter your **Bot Token** from BotFather
4. Test connection
5. The webhook URL will be automatically registered

### Step 4: Configure OpenAI for AI Agent
1. Click on **"OpenAI Chat Model"** node
2. **Credentials** → **Create New Credential** or select existing
3. Enter your **OpenAI API Key**
4. Set model to **gpt-4.1-mini**
5. Verify connection

### Step 5: Configure OpenAI for Summarization
1. Click on **"OpenAI Chat Model1"** node
2. **Credentials** → Use same or different OpenAI credential
3. Set model to **gpt-4o-mini**
4. This model is used for audio response summarization

### Step 6: Configure Speech-to-Text (STT)
1. Click on **"STT_Transcribe"** node
2. **Credentials** → Select OpenAI credential
3. **Resource**: `audio`
4. **Operation**: `transcribe`
5. This will convert voice messages to text

### Step 7: Configure Text-to-Speech (TTS)
1. Click on **"TTS_Generate Audio"** node
2. **Credentials** → Select OpenAI credential
3. **Resource**: `audio`
4. **Input**: Text from summarization chain
5. This generates audio from text responses

### Step 8: Configure SerpAPI (Google Search)
1. Go to [SerpAPI](https://serpapi.com/) and create account
2. Get your **API Key** from dashboard
3. Click on **"Google search in SerpApi"** node
4. **Credentials** → **Create New Credential**
5. Enter **SerpAPI Key**
6. Test connection

### Step 9: Configure Gmail Integration
1. Click on **"Send a message in Gmail"** node
2. **Credentials** → **Create New Credential**
3. Follow Gmail OAuth2 setup:
   - Enable Gmail API in Google Cloud Console
   - Create OAuth2 Client ID
   - Add redirect URI from n8n
   - Connect with Google account
4. Grant email sending permissions

### Step 10: Configure Memory
1. Click on **"Simple Memory"** node
2. **Session ID Type**: `Custom Key`
3. **Session Key**: `Telegram_{{ $('Telegram Trigger').item.json.message.chat.id }}`
4. This ensures separate memory per Telegram chat

### Step 11: Configure AI Agent System Message
1. Click on **"AI Agent"** node
2. Review **System Message** in Options:
   - Defines bot personality and capabilities
   - Instructs on handling text vs audio
   - Specifies tool usage guidelines
3. Modify if needed for your use case

### Step 12: Configure Telegram Output Nodes
1. **"Send a text message"** node:
   - Automatically configured with chat ID
   - Sends text responses from AI agent

2. **"Send Audio Output to Telegram"** node:
   - Configured for binary audio data
   - Sends audio responses back to user

### Step 13: Activate Workflow
1. Toggle **"Active"** switch at top right
2. Workflow is now live and listening for Telegram messages

## How to Execute

### Method 1: Text Message
1. Open Telegram and find your bot
2. Send a text message: *"Hello, how are you?"*
3. Bot will:
   - Process text message
   - Route through Text Var node
   - AI Agent generates response
   - Send text reply back

### Method 2: Voice Message
1. In Telegram, tap and hold microphone icon
2. Record a voice message
3. Send to bot
4. Bot will:
   - Detect voice message
   - Fetch audio file from Telegram
   - Transcribe using STT
   - Process through AI Agent
   - Generate response
   - Summarize for audio format
   - Convert to speech (TTS)
   - Send both text and audio replies

### Method 3: Request Web Search
1. Send message: *"Search for latest AI trends"*
2. Bot will:
   - Use Google search tool via SerpAPI
   - Retrieve search results
   - Generate intelligent response
   - Reply with information

### Method 4: Request Email
1. Send message: *"Send an email to john@example.com with subject 'Meeting' and message 'Let's meet tomorrow'"*
2. Bot will:
   - Parse email request
   - Use Gmail tool
   - Send email automatically
   - Confirm sending

## Workflow Structure

### Main Flow
```
Telegram Trigger
    ↓
Switch: Voice or Text Message?
    ├─ Voice Path → Fetch Audio → STT → Audio Var → AI Agent
    └─ Text Path → Text Var → AI Agent
                        ↓
                    AI Agent
                    (with Memory, Search, Gmail tools)
                        ↓
            ┌───────────┴───────────┐
            ↓                       ↓
    Send Text Message      Summarize for Audio
                                ↓
                            TTS Generate Audio
                                ↓
                        Send Audio to Telegram
```

### AI Agent Tools
- **Google Search (SerpAPI)**: Web search capability
- **Gmail Tool**: Send emails
- **Memory**: Conversation context

## Configuration Details

### Switch Node Logic
- **Rule 1**: Voice Condition
  - Checks if `message.voice.file_id` exists
  - Routes to voice processing path
- **Rule 2**: Text Condition
  - Checks if `message.text` exists
  - Routes to text processing path

### AI Agent Configuration
- **Model**: gpt-4.1-mini
- **Memory**: Session-based per chat ID
- **Tools**: SerpAPI search, Gmail send
- **System Message**: Defines assistant behavior and capabilities

### Audio Processing
- **STT Model**: OpenAI Whisper (via OpenAI node)
- **TTS Model**: OpenAI TTS (via OpenAI node)
- **Summarization**: Optimizes text for speech synthesis

### Response Paths
1. **Text Only**: AI Agent → Send Text Message
2. **Audio Enhanced**: AI Agent → Summarize → TTS → Send Audio

## Expected Behavior

### Text Messages
- User sends text → Bot responds with text
- Maintains conversation context
- Can use tools (search, email) as needed

### Voice Messages
- User sends voice → Bot transcribes
- Processes through AI agent
- Responds with both:
  - Text message (full response)
  - Audio message (summarized, optimized for speech)

### Tool Usage Examples
- **Search**: *"What's the weather in New York?"*
- **Email**: *"Email sarah@example.com about the project update"*
- **Conversation**: *"Tell me a joke"* or *"How do I cook pasta?"*

## Troubleshooting

### Issue: Bot Doesn't Respond
- **Solution**:
  - Verify Telegram Trigger webhook is active
  - Check workflow is active (not paused)
  - Review execution logs for errors
  - Verify bot token is correct

### Issue: Voice Messages Not Processed
- **Solution**:
  - Check Switch node routing logic
  - Verify STT node has OpenAI credentials
  - Ensure audio file is being fetched from Telegram
  - Check audio format compatibility

### Issue: Audio Generation Fails
- **Solution**:
  - Verify TTS node configuration
  - Check OpenAI API quota/limits
  - Ensure text input is valid
  - Review summarization chain output

### Issue: Search Doesn't Work
- **Solution**:
  - Verify SerpAPI key is valid
  - Check API quota/credits
  - Ensure tool is connected to AI Agent
  - Review agent system message for tool usage

### Issue: Gmail Sending Fails
- **Solution**:
  - Re-authenticate Gmail OAuth2
  - Check email format in AI agent output
  - Verify Gmail API permissions
  - Review execution logs

### Issue: Memory Not Working
- **Solution**:
  - Check Session Key expression
  - Verify Memory node is connected to AI Agent
  - Ensure chat ID is being captured correctly

## Customization

### Modify Bot Personality
1. Edit **AI Agent** node → **System Message**
2. Adjust tone, style, and capabilities
3. Update tool usage instructions

### Add New Tools
1. Create new tool node (e.g., Calendar, Drive)
2. Connect to AI Agent as `ai_tool`
3. Update system message to describe tool

### Adjust Audio Settings
1. Modify **Summarization Chain** prompt for different audio styles
2. Change TTS voice/model settings in OpenAI node
3. Adjust audio format/quality parameters

### Change Response Behavior
1. Modify Switch node to add more routing logic
2. Add conditional nodes for specific responses
3. Customize error handling paths

## Important Notes
- **Session Memory**: Each Telegram chat has separate memory
- **Rate Limits**: Be aware of OpenAI and SerpAPI rate limits
- **Privacy**: Voice messages are processed by OpenAI; review privacy policy
- **Costs**: Monitor API usage (OpenAI, SerpAPI)
- **Webhook**: Telegram webhook must be accessible from internet
- **Workflow Status**: Currently `"active": false` - activate before use

## Security Considerations
- Store API keys securely in n8n credentials
- Regularly rotate bot tokens
- Review OAuth2 permissions granted
- Monitor bot usage and access logs
- Implement rate limiting if needed

