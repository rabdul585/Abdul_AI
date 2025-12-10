# Telegram Jarvis Text Agent - AI-Powered Telegram Assistant

## Purpose
This n8n workflow creates an intelligent Telegram bot named "Jarvis" that processes both text and voice messages. It provides AI-powered responses, web search capabilities via SerpAPI, email sending through Gmail, and supports audio responses with text-to-speech. The bot maintains conversation context and can handle complex queries requiring multiple tools.

## Key Features
- **Dual Input Processing**: Handles both text and voice messages intelligently
- **AI-Powered Assistant**: Uses GPT-4.1-mini for natural conversations
- **Voice Support**: 
  - Speech-to-Text (STT) transcription for voice messages
  - Text-to-Speech (TTS) for audio responses
- **Web Search**: Google search integration via SerpAPI
- **Email Integration**: Send emails through Gmail directly from chat
- **Conversation Memory**: Maintains context across messages per chat
- **Smart Message Routing**: Automatically detects and routes voice vs text

## Prerequisites
1. **n8n Instance**: Active n8n installation (cloud or self-hosted)
2. **Telegram Bot**:
   - Created via [@BotFather](https://t.me/botfather)
   - Bot token obtained
3. **OpenAI API Key**: 
   - Access to OpenAI API
   - Models needed: gpt-4.1-mini (main agent), gpt-4o-mini (summarization)
4. **SerpAPI Key**: For Google search functionality
5. **Gmail OAuth2**: For email sending (if email functionality needed)

## Step-by-Step Setup Instructions

### Step 1: Create Telegram Bot
1. Open Telegram application
2. Search for **[@BotFather](https://t.me/botfather)**
3. Send `/newbot` command
4. Follow prompts:
   - Bot name: (e.g., "N8N Abdul Bot")
   - Bot username: Must end with "bot" (e.g., "N8nabdul_bot")
5. **Save the Bot Token** provided (format: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
6. (Optional) Configure bot:
   - `/setdescription` - Add bot description
   - `/setabouttext` - Add about information
   - `/setcommands` - Set command list

### Step 2: Import Workflow
1. In n8n, navigate to **Workflows** in left sidebar
2. Click **"+"** button to create new workflow
3. Click three dots (⋮) → **"Import from File"**
4. Select: `Telegram Jarvis _text Agent.json`
5. Workflow will be imported with all nodes and connections

### Step 3: Configure Telegram Trigger
1. Click on **"Telegram Trigger"** node
2. In **Credentials** section:
   - Click **"Create New Credential"** or select existing
   - Select **"Telegram API"**
   - Enter your **Bot Token** from Step 1
   - Click **"Save"**
3. Verify connection shows as successful
4. The webhook will be automatically registered by n8n

### Step 4: Configure OpenAI Chat Model (Main Agent)
1. Click on **"OpenAI Chat Model"** node
2. **Credentials** → **Create New Credential** or select existing
3. Enter your **OpenAI API Key**
4. Set **Model** to: `gpt-4.1-mini`
5. Click **"Save"** and verify connection

### Step 5: Configure OpenAI for Audio Summarization
1. Click on **"OpenAI Chat Model1"** node
2. **Credentials** → Use same or different OpenAI credential
3. Set **Model** to: `gpt-4o-mini`
4. This model optimizes text for speech synthesis

### Step 6: Configure Speech-to-Text (STT)
1. Click on **"STT_Transcribe"** node
2. **Credentials** → Select your OpenAI credential
3. **Resource**: `audio`
4. **Operation**: `transcribe`
5. This converts voice messages to text for processing

### Step 7: Configure Text-to-Speech (TTS)
1. Click on **"TTS_Generate Audio"** node
2. **Credentials** → Select your OpenAI credential
3. **Resource**: `audio`
4. **Input**: Will receive text from summarization chain
5. Generates audio file from text response

### Step 8: Configure SerpAPI (Google Search)
1. Create account at [SerpAPI](https://serpapi.com/)
2. Navigate to dashboard and copy your **API Key**
3. In n8n, click on **"Google search in SerpApi"** node
4. **Credentials** → **Create New Credential**
5. Enter your **SerpAPI Key**
6. Test connection
7. This enables web search functionality in bot

### Step 9: Configure Gmail Integration
1. Click on **"Send a message in Gmail"** node
2. **Credentials** → **Create New Credential**
3. Follow Gmail OAuth2 setup:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Enable **Gmail API**
   - Create **OAuth2 Client ID** (Web application)
   - Copy **Client ID** and **Client Secret**
   - Copy **Redirect URI** from n8n credential setup
   - Add redirect URI in Google Cloud Console
   - Paste credentials in n8n
   - Click **"Connect with Google"**
   - Authorize Gmail access
4. Verify **"Account connected"** status

### Step 10: Configure Conversation Memory
1. Click on **"Simple Memory"** node
2. **Session ID Type**: `Custom Key`
3. **Session Key**: `Telegram_{{ $('Telegram Trigger').item.json.message.chat.id }}`
4. This creates unique memory session per Telegram chat
5. Memory maintains conversation context across messages

### Step 11: Review AI Agent Configuration
1. Click on **"AI Agent"** node
2. Review **System Message** in Options:
   - Defines bot personality as intelligent, friendly assistant
   - Instructions for handling text vs audio messages
   - Guidelines for tool usage (search, email)
   - Tone and response style preferences
3. Modify system message if you want different behavior

### Step 12: Configure Message Routing (Switch Node)
1. Click on **"Voice or Text Message"** switch node
2. **Rule 1 - VoiceConditionSatisfied**:
   - Checks if `message.voice.file_id` exists
   - Routes voice messages to audio processing path
3. **Rule 2 - TextConditionSatisfied**:
   - Checks if `message.text` exists
   - Routes text messages directly to agent
4. This automatically detects message type

### Step 13: Configure Output Nodes
1. **"Send a text message"** node:
   - Automatically configured to use chat ID from trigger
   - Sends text responses to user

2. **"Send Audio Output to Telegram"** node:
   - Configured for binary audio data
   - Sends audio messages back to user

### Step 14: Activate Workflow
1. Toggle **"Active"** switch at top right of workflow canvas
2. Workflow is now live and listening for Telegram messages
3. Test by sending a message to your bot

## How to Execute

### Method 1: Text Message Interaction
1. Open Telegram on your phone or desktop
2. Search for your bot by username (e.g., `@N8nabdul_bot`)
3. Start conversation by sending: *"Hello, how are you?"*
4. Bot will:
   - Detect text message
   - Process through AI agent
   - Generate intelligent response
   - Send text reply back

### Method 2: Voice Message Interaction
1. In Telegram chat with bot, tap and hold microphone icon
2. Record a voice message (e.g., *"What's the weather today?"*)
3. Release to send
4. Bot will:
   - Detect voice message
   - Fetch audio file from Telegram
   - Transcribe using OpenAI STT
   - Process through AI agent
   - Generate response
   - Summarize response for audio format
   - Convert to speech via TTS
   - Send both text AND audio replies

### Method 3: Web Search Query
1. Send message: *"Search for latest developments in AI"*
2. Bot will:
   - Recognize search intent
   - Use SerpAPI tool to search Google
   - Process search results
   - Generate comprehensive response
   - Reply with search findings

### Method 4: Email Request
1. Send message: *"Send an email to john@example.com. Subject: Meeting Tomorrow. Message: Let's discuss the project at 3 PM."*
2. Bot will:
   - Parse email details from message
   - Use Gmail tool to send email
   - Confirm email sent
   - Reply with confirmation

### Method 5: Complex Multi-Tool Query
1. Send: *"Search for best restaurants in New York and email me the top 3"*
2. Bot will:
   - Perform web search
   - Extract top 3 results
   - Compose email with results
   - Send email via Gmail
   - Confirm completion

## Workflow Execution Flow

### Text Message Path
```
Telegram Trigger (text message)
    ↓
Switch Node → Text Condition
    ↓
Text Var (passes text)
    ↓
AI Agent (processes with tools available)
    ├─ OpenAI Chat Model
    ├─ Simple Memory
    ├─ SerpAPI Tool
    └─ Gmail Tool
    ↓
Send Text Message → User
```

### Voice Message Path
```
Telegram Trigger (voice message)
    ↓
Switch Node → Voice Condition
    ↓
Fetch Audio from Telegram
    ↓
STT_Transcribe (OpenAI)
    ↓
Audio Var (passes transcribed text)
    ↓
AI Agent (processes query)
    ├─ OpenAI Chat Model
    ├─ Simple Memory
    ├─ SerpAPI Tool
    └─ Gmail Tool
    ↓
    ├─ Send Text Message → User
    └─ Summarize for Audio → TTS → Send Audio → User
```

## Configuration Details

### AI Agent System Message
The bot is configured to:
- Be intelligent and friendly
- Handle both text and audio inputs
- Use tools (search, email) when needed
- Maintain natural, conversational tone
- Adapt response style based on input type

### Memory Configuration
- **Session Type**: Custom key based on Telegram chat ID
- **Purpose**: Maintains conversation context
- **Scope**: Separate memory per Telegram chat/user

### Tool Integration
1. **SerpAPI Search**:
   - Parameter: Search query from AI agent
   - Returns: Google search results
   - Used for: Web queries, current information

2. **Gmail Send**:
   - Parameters: To, Subject, Message (from AI agent)
   - Returns: Email sent confirmation
   - Used for: Email requests from users

### Audio Processing
- **STT**: OpenAI Whisper model for transcription
- **TTS**: OpenAI TTS for audio generation
- **Summarization**: Optimizes text for speech (removes redundancy)

## Expected Behavior Examples

### Example 1: Simple Conversation
**User**: "Tell me a joke"
**Bot**: "Why don't scientists trust atoms? Because they make up everything! 😄"

### Example 2: Web Search
**User**: "What's the capital of France?"
**Bot**: Uses search tool → "The capital of France is Paris, a beautiful city known for..."

### Example 3: Voice Query
**User**: [Sends voice: "What's the weather like?"]
**Bot**: 
- [Text]: "I'd be happy to help with weather information. However, I'd need your location or you could search for 'weather in [your city]' to get current conditions."
- [Audio]: Summarized audio version of response

### Example 4: Email Request
**User**: "Email sarah@example.com about our meeting tomorrow"
**Bot**: "I've sent an email to sarah@example.com about your meeting tomorrow. The email has been delivered successfully!"

## Troubleshooting

### Issue: Bot Doesn't Respond to Messages
- **Solution**:
  - Verify Telegram Trigger webhook is active (green indicator)
  - Check workflow is active (not paused)
  - Review execution logs for errors
  - Verify bot token is correct
  - Test webhook: Send message and check n8n executions

### Issue: Voice Messages Not Processed
- **Solution**:
  - Check Switch node logic (voice condition)
  - Verify STT node has valid OpenAI credentials
  - Ensure audio file is being fetched from Telegram
  - Check audio format compatibility
  - Review execution logs for transcription errors

### Issue: Audio Response Not Generated
- **Solution**:
  - Verify TTS node configuration
  - Check OpenAI API quota/limits
  - Ensure summarization chain produces valid output
  - Review audio generation node logs
  - Check Telegram audio sending permissions

### Issue: Web Search Doesn't Work
- **Solution**:
  - Verify SerpAPI key is valid and has credits
  - Check API quota/limits in SerpAPI dashboard
  - Ensure tool is connected to AI Agent node
  - Review agent system message for tool usage instructions
  - Check execution logs for API errors

### Issue: Gmail Sending Fails
- **Solution**:
  - Re-authenticate Gmail OAuth2 credentials
  - Verify email format in AI agent output
  - Check Gmail API permissions (send scope)
  - Review OAuth consent screen configuration
  - Check execution logs for specific error messages

### Issue: Memory Not Working (Bot Forgets Context)
- **Solution**:
  - Verify Session Key expression is correct
  - Check Memory node is connected to AI Agent
  - Ensure chat ID is being captured correctly
  - Review memory node configuration
  - Check if memory buffer size needs adjustment

### Issue: Bot Response Too Slow
- **Solution**:
  - Optimize AI model (use faster models if acceptable)
  - Reduce audio processing steps if not needed
  - Check API response times (OpenAI, SerpAPI)
  - Consider caching frequently asked queries
  - Review workflow for unnecessary nodes

## Customization Options

### Modify Bot Personality
1. Edit **AI Agent** node → **System Message**
2. Adjust tone, style, and capabilities description
3. Update tool usage instructions
4. Change response length preferences

### Add New Tools/Capabilities
1. Create new tool node (e.g., Calendar, Drive, Database)
2. Connect to AI Agent as `ai_tool` connection
3. Update system message to describe new tool
4. Test tool integration

### Adjust Audio Settings
1. Modify **Summarization Chain** prompt for different audio styles
2. Change TTS voice/model settings in OpenAI node
3. Adjust audio format/quality parameters
4. Modify summarization length for audio responses

### Change Response Behavior
1. Modify Switch node to add more routing logic
2. Add conditional nodes for specific response types
3. Customize error handling paths
4. Add filtering for specific commands

### Implement Command System
1. Add IF nodes to detect specific commands (e.g., `/search`, `/email`)
2. Route to specialized processing paths
3. Provide help command with available commands

## Security Considerations
- **API Keys**: Store all API keys securely in n8n credentials
- **Bot Token**: Keep Telegram bot token secure
- **OAuth2**: Regularly review OAuth permissions granted
- **Rate Limiting**: Implement rate limiting if bot goes public
- **Privacy**: Be transparent about data processing (OpenAI, etc.)
- **Access Control**: Consider implementing user whitelist if needed

## Important Notes
- **Session Memory**: Each Telegram chat maintains separate conversation context
- **Rate Limits**: Be aware of OpenAI, SerpAPI, and Telegram rate limits
- **Privacy**: Voice messages are processed by OpenAI; review privacy policies
- **Costs**: Monitor API usage (OpenAI, SerpAPI) for cost management
- **Webhook**: Telegram webhook must be accessible from internet
- **Workflow Status**: Currently `"active": false` - activate before use
- **Model Availability**: Verify gpt-4.1-mini and gpt-4o-mini are available in your OpenAI account

## Best Practices
1. **Test Thoroughly**: Test all features before going live
2. **Monitor Logs**: Regularly check execution logs for errors
3. **Update System Message**: Refine based on user interactions
4. **Handle Errors Gracefully**: Add error messages for failed operations
5. **Document Commands**: Provide users with available commands/features
6. **Backup Workflow**: Export workflow regularly for backups

## Scaling Considerations
- For high traffic: Consider load balancing multiple n8n instances
- Implement queuing for API calls to manage rate limits
- Cache frequently accessed information
- Monitor resource usage and scale infrastructure as needed

