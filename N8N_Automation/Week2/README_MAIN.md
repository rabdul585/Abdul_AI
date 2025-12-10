# N8N Automation Workflows - Week 2

This directory contains five comprehensive n8n automation workflows demonstrating various automation capabilities including AI agents, Google Workspace integrations, Telegram bots, and RPA automation.

## 📋 Overview

| Workflow | Purpose | Complexity | Status |
|----------|---------|------------|--------|
| **AI Agent work_Gmail** | AI-powered birthday email generator via chat | Medium | ✅ Active |
| **Gmail_Workflow & Create Event** | Multi-service Google Workspace automation | High | ⚠️ Inactive |
| **My workflow** | Advanced Telegram AI assistant with voice/text | High | ⚠️ Inactive |
| **RPA automation_Notepad** | Scheduled AI quote generator to Notepad | Medium | ⚠️ Inactive |
| **Telegram Jarvis Text Agent** | Intelligent Telegram bot with multi-tool support | High | ⚠️ Inactive |

## 🚀 Quick Start

### Prerequisites
1. **n8n Instance**: Active n8n installation (cloud or self-hosted)
2. **Common Requirements**:
   - OpenAI API Key (for AI workflows)
   - Google Cloud Console account (for Google integrations)
   - Telegram Bot Token (for Telegram workflows)
   - SerpAPI Key (for search functionality)

### Installation Steps

1. **Import Workflows**:
   ```bash
   # In n8n interface:
   Workflows → "+" → Import from File
   ```

2. **Configure Credentials**:
   - Set up OpenAI API credentials
   - Configure Google OAuth2 for Gmail/Calendar/Drive
   - Set up Telegram bot tokens
   - Add SerpAPI credentials (if needed)

3. **Activate Workflows**:
   - Toggle "Active" switch for each workflow
   - Verify webhook/trigger configurations

## 📚 Detailed Documentation

Each workflow has comprehensive documentation:

### 1. [AI Agent work_Gmail](./README_AI_Agent_work_Gmail.md)
- **Purpose**: Automated birthday email generation via chat interface
- **Key Features**: AI chat agent, Gmail integration, personalized emails
- **Setup Time**: ~15 minutes
- **Use Cases**: Birthday reminders, automated email campaigns, AI email assistant

### 2. [Gmail Workflow & Create Event](./README_Gmail_Workflow_Create_Event.md)
- **Purpose**: Multi-service Google Workspace automation
- **Key Features**: Gmail, Drive, Docs, Sheets, Calendar integration
- **Setup Time**: ~30 minutes
- **Use Cases**: Automated Google Workspace operations, multi-step automations

### 3. [My Workflow](./README_My_Workflow.md)
- **Purpose**: Advanced Telegram AI assistant with voice and text support
- **Key Features**: STT, TTS, web search, Gmail, conversation memory
- **Setup Time**: ~25 minutes
- **Use Cases**: Personal AI assistant, voice-activated bot, multi-tool assistant

### 4. [RPA Automation Notepad](./README_RPA_Automation_Notepad.md)
- **Purpose**: Scheduled AI quote generator writing to Notepad via RPA
- **Key Features**: Scheduled execution, AI quote generation, Flask API, RPA
- **Setup Time**: ~20 minutes
- **Use Cases**: Daily quotes, automated file writing, RPA demonstrations

### 5. [Telegram Jarvis Text Agent](./README_Telegram_Jarvis_Text_Agent.md)
- **Purpose**: Intelligent Telegram bot with multi-tool capabilities
- **Key Features**: Voice/text processing, search, email, audio responses
- **Setup Time**: ~25 minutes
- **Use Cases**: Customer support bot, personal assistant, automated helpdesk

## 🔧 Common Setup Procedures

### Google OAuth2 Setup (Required for Gmail/Calendar/Drive workflows)

1. **Google Cloud Console**:
   - Go to [console.cloud.google.com](https://console.cloud.google.com)
   - Create/select project
   - Enable required APIs (Gmail, Drive, Docs, Sheets, Calendar)

2. **OAuth Consent Screen**:
   - Configure app information
   - Set user type (Internal/External)
   - Add test users if external

3. **Create OAuth Client**:
   - Application type: Web application
   - Copy Client ID and Client Secret
   - Add authorized redirect URIs from n8n

4. **Configure in n8n**:
   - Paste credentials in workflow nodes
   - Connect with Google account
   - Authorize permissions

### OpenAI API Setup

1. **Get API Key**:
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Generate API key from dashboard
   - Set usage limits/billing

2. **Configure in n8n**:
   - Create OpenAI credential in n8n
   - Paste API key
   - Test connection

### Telegram Bot Setup

1. **Create Bot**:
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Use `/newbot` command
   - Follow instructions and save token

2. **Configure in n8n**:
   - Add Telegram credential
   - Paste bot token
   - Webhook will auto-register

## 📖 How to Use This Documentation

1. **Choose Your Workflow**: Review the overview table above
2. **Read Individual README**: Open the specific README file for detailed instructions
3. **Follow Step-by-Step**: Each README has comprehensive setup steps
4. **Troubleshoot**: Check troubleshooting sections for common issues
5. **Customize**: Modify workflows based on your needs

## 🎯 Workflow Selection Guide

**Choose AI Agent work_Gmail if you want to:**
- Send automated birthday emails
- Use AI to generate email content
- Interact via chat interface

**Choose Gmail Workflow if you want to:**
- Automate multiple Google services
- Create workflows across Gmail, Drive, Docs, Calendar
- Execute multi-step Google operations

**Choose My Workflow/Telegram Jarvis if you want to:**
- Create intelligent Telegram bots
- Handle voice and text messages
- Enable web search and email from chat
- Build conversational AI assistants

**Choose RPA Notepad if you want to:**
- Schedule automated file writing
- Use RPA techniques (pyautogui)
- Generate daily quotes or content
- Integrate Flask APIs with n8n

## ⚠️ Important Notes

- **Activation**: Most workflows are set to `"active": false` - activate before use
- **Credentials**: All API keys and tokens should be stored securely
- **Rate Limits**: Be aware of API rate limits (OpenAI, SerpAPI, Google)
- **Privacy**: Review privacy policies for services used (OpenAI, Google, etc.)
- **Costs**: Monitor API usage and associated costs
- **Testing**: Test workflows thoroughly before production use

## 🔍 Troubleshooting

### Common Issues

1. **Workflow Not Executing**:
   - Check if workflow is active
   - Verify trigger/webhook configuration
   - Review execution logs

2. **Credential Errors**:
   - Verify API keys are correct
   - Check credential expiration
   - Re-authenticate OAuth2 connections

3. **API Rate Limits**:
   - Monitor API usage
   - Implement delays if needed
   - Upgrade API plans if required

For specific issues, refer to individual workflow README files.

## 📝 File Structure

```
Week2/
├── AI Agent work_Gmail.json
├── Gmail_Worklfow & Create Event.json
├── My workflow.json
├── RPA automation_Notepat (1).json
├── Telegram Jarvis _text Agent.json
├── README_MAIN.md (this file)
├── README_AI_Agent_work_Gmail.md
├── README_Gmail_Workflow_Create_Event.md
├── README_My_Workflow.md
├── README_RPA_Automation_Notepad.md
└── README_Telegram_Jarvis_Text_Agent.md
```

## 🆘 Support

For issues or questions:
1. Check individual workflow README troubleshooting sections
2. Review n8n execution logs
3. Verify all credentials and configurations
4. Consult n8n documentation: [docs.n8n.io](https://docs.n8n.io)

## 📄 License

These workflows are provided as examples for educational and automation purposes. Ensure compliance with terms of service for all integrated platforms (OpenAI, Google, Telegram, SerpAPI).

## 🔄 Updates

- Regularly update API keys and credentials
- Monitor for workflow improvements
- Check for n8n platform updates
- Review security best practices

---

**Last Updated**: 2025-12-06  
**n8n Version**: Compatible with n8n version supporting LangChain nodes  
**Status**: Documentation complete for all workflows

