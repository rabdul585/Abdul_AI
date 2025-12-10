# RPA Automation Notepad - Scheduled Quote Generator

## Purpose
This n8n workflow automates the generation of motivational quotes and writes them to a Notepad file using Robotic Process Automation (RPA) techniques. The workflow runs on a schedule (every minute), generates inspirational quotes using AI, formats them with timestamps, and automatically writes them to a text file via a Flask API that controls the Windows Notepad application.

## Key Features
- **Scheduled Execution**: Runs automatically every minute (configurable)
- **AI-Powered Quote Generation**: Uses OpenAI to create motivational quotes
- **Timestamp Formatting**: Adds formatted date/time stamps (DDMMYYYY HHMMSS)
- **RPA Integration**: Controls Windows Notepad via Flask API
- **File Management**: Creates file on first run, appends on subsequent runs
- **Automated File Operations**: Opens Notepad, types content, saves file

## Prerequisites
1. **n8n Instance**: Active n8n installation
2. **OpenAI API Key**: Access to OpenAI API (model: gpt-4.1-nano)
3. **Flask API Server**: Python Flask application running locally
4. **Python Environment**: Python with required packages:
   - `flask`
   - `pyautogui`
   - `time`
   - `os`
   - `subprocess`
5. **Windows OS**: Required for Notepad automation (pyautogui)
6. **File Directory**: Ensure directory exists: `D:\Abdul_AI\RPA_UI_N8Nauto`

## Step-by-Step Setup Instructions

### Step 1: Set Up Flask API Server

#### 1.1 Create Python Script
Create a file named `notepad_api.py` with the following content:

```python
from flask import Flask, request, jsonify
import pyautogui
import time
import os
import subprocess
import sys

app = Flask(__name__)

FILE_PATH = r"D:\Abdul_AI\RPA_UI_N8Nauto\socialeagledemo.txt"

def open_notepad_with_file(file_path=None):
    if file_path:
        subprocess.Popen(["notepad.exe", file_path])
    else:
        subprocess.Popen(["notepad.exe"])
    time.sleep(1.5)  # Allow Notepad to open

def write_and_save_first_time(user_input):
    # Write the text
    pyautogui.typewrite(user_input, interval=0.05)
    # Save the file (Ctrl+S)
    pyautogui.hotkey('ctrl', 's')
    time.sleep(1)
    # Type the full file path
    pyautogui.typewrite(FILE_PATH, interval=0.03)
    # Press Enter to save
    pyautogui.press('enter')
    time.sleep(1)

def append_and_save(user_input):
    # Move cursor to end (Notepad usually starts at top)
    pyautogui.hotkey('ctrl', 'end')
    pyautogui.press('enter')  # New line
    # Write new input
    pyautogui.typewrite(user_input, interval=0.05)
    # Save
    pyautogui.hotkey('ctrl', 's')
    time.sleep(1)

@app.route('/write-text', methods=['POST'])
def write_text():
    try:
        data = request.get_json()
        user_input = data.get('text', '')
        
        if not user_input:
            return jsonify({'error': 'No text provided'}), 400
        
        file_exists = os.path.exists(FILE_PATH)
        
        if not file_exists:
            # First time - create new file
            open_notepad_with_file()
            write_and_save_first_time(user_input)
            return jsonify({'message': 'File created and text written', 'file': FILE_PATH})
        else:
            # Subsequent times - append to existing file
            open_notepad_with_file(FILE_PATH)
            append_and_save(user_input)
            return jsonify({'message': 'Text appended to file', 'file': FILE_PATH})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
```

#### 1.2 Install Required Packages
Open terminal/command prompt and run:

```bash
pip install flask pyautogui
```

#### 1.3 Create Directory
```bash
mkdir D:\Abdul_AI\RPA_UI_N8Nauto
```

#### 1.4 Start Flask Server
```bash
python notepad_api.py
```

The server will start on `http://127.0.0.1:5000`

#### 1.5 (Optional) Expose via Ngrok
If n8n is on a different machine, expose Flask API via ngrok:

```bash
ngrok http 5000
```

Copy the ngrok URL (e.g., `https://thad-unshepherding-evonne.ngrok-free.dev`)

### Step 2: Import Workflow
1. In n8n, go to **Workflows** → **"+"** → **Import from File**
2. Select: `RPA automation_Notepat (1).json`
3. Workflow will be imported

### Step 3: Configure Schedule Trigger
1. Click on **"Schedule Trigger"** node
2. **Interval**: Every `1` minute (or adjust as needed)
   - You can change to hours, days, etc.
3. Set timezone if needed

### Step 4: Configure Date & Time Node
1. Click on **"Date & time"** node
2. The expression generates timestamp in format: `DDMMYYYY HHMMSS`
3. Expression used:
   ```javascript
   {{new Date().toLocaleString('en-GB',{hour12:false}).replace(/[\/ ]/g,'').replace(',',' ').replace(/:/g,'')}}
   ```
4. Output variable: `Date_Time`

### Step 5: Configure OpenAI Model
1. Click on **"Abdul API key"** node
2. **Credentials** → **Create New Credential** or select existing
3. Enter your **OpenAI API Key**
4. Set model to **gpt-4.1-nano**
5. Verify connection

### Step 6: Configure Basic LLM Chain
1. Click on **"Basic LLM Chain"** node
2. Review the prompt:
   ```
   Hello, Write a Motivation Rumi quote {{ $('Date & time').item.json.Date_Time }}
   Example: You are the best. Focus on your skill now. Time 03-12-2025 22:32:01 IST
   ```
3. Modify prompt if needed for different quote styles

### Step 7: Configure HTTP Request Node
1. Click on **"HTTP Request"** node
2. **Method**: `POST`
3. **URL**: 
   - Local: `http://127.0.0.1:5000/write-text`
   - Or ngrok URL: `https://your-ngrok-url.ngrok-free.dev/write-text`
4. **Body Parameters**:
   - Name: `text`
   - Value: `={{ $json.Date_Time }}{{ $json.text }}`
   - This combines timestamp with AI-generated quote
5. **Send Body**: Enabled

### Step 8: Test Flask API Connection
1. In n8n, manually execute **HTTP Request** node
2. Or test via curl:
   ```bash
   curl -X POST http://127.0.0.1:5000/write-text -H "Content-Type: application/json" -d "{\"text\":\"Test message\"}"
   ```
3. Verify Notepad opens and text is written

### Step 9: Activate Workflow
1. Toggle **"Active"** switch at top right
2. Workflow will now run automatically on schedule

## How to Execute

### Automatic Execution (Scheduled)
1. Once activated, workflow runs automatically every minute
2. Each execution will:
   - Generate timestamp
   - Create motivational quote via AI
   - Combine timestamp + quote
   - Call Flask API
   - Write to Notepad file

### Manual Execution
1. Click **"Execute Workflow"** button
2. Workflow will run once immediately
3. Check Notepad file for output

### Execution Flow
```
Schedule Trigger (every 1 minute)
    ↓
Generate Date & Time (DDMMYYYY HHMMSS)
    ↓
AI Generate Motivational Quote
    ↓
Combine Timestamp + Quote
    ↓
HTTP Request to Flask API
    ↓
Flask API: Open Notepad & Write Text
    ↓
Save File (create new or append)
```

## Configuration Details

### Schedule Trigger
- **Interval**: 1 minute (configurable)
- **Timezone**: Uses n8n instance timezone
- Can be changed to: hours, days, specific times, etc.

### Date Format
- **Format**: `DDMMYYYY HHMMSS`
- **Example**: `03122025 223201`
- Uses JavaScript Date object with locale formatting

### AI Quote Generation
- **Model**: gpt-4.1-nano
- **Prompt**: Requests Rumi-style motivational quote
- **Output**: Short, inspirational message

### Flask API Endpoint
- **URL**: `/write-text`
- **Method**: POST
- **Payload**: `{"text": "timestamp + quote"}`

### File Operations
- **First Run**: Creates `socialeagledemo.txt`
- **Subsequent Runs**: Appends to existing file
- **Location**: `D:\Abdul_AI\RPA_UI_N8Nauto\socialeagledemo.txt`

## Expected Output

### File Structure
Each entry in the file will look like:
```
03122025 223201 You are the best. Focus on your skill now.
04122025 105530 Embrace challenges as opportunities to grow.
05122025 143015 Believe in your journey, trust the process.
```

### Behavior
- **First Execution**: Creates new file with first quote
- **Subsequent Executions**: Appends new quotes on new lines
- **Timestamp**: Each quote has unique timestamp prefix
- **Notepad**: Opens automatically during write operations

## Troubleshooting

### Issue: Flask API Not Responding
- **Solution**:
  - Check Flask server is running: `python notepad_api.py`
  - Verify port 5000 is not blocked
  - Check firewall settings
  - Test with curl or Postman

### Issue: Notepad Doesn't Open
- **Solution**:
  - Ensure Windows OS (pyautogui requirement)
  - Check Notepad is available in system
  - Verify pyautogui can control desktop
  - Run Flask server with admin privileges if needed

### Issue: File Not Saving
- **Solution**:
  - Verify directory exists: `D:\Abdul_AI\RPA_UI_N8Nauto`
  - Check write permissions on directory
  - Ensure file path is correct
  - Review pyautogui timing (sleep intervals)

### Issue: Text Not Appearing
- **Solution**:
  - Increase sleep intervals in Flask script
  - Ensure Notepad window is in focus
  - Check pyautogui coordinates
  - Verify text is being sent correctly

### Issue: Workflow Runs but No Output
- **Solution**:
  - Check execution logs in n8n
  - Verify HTTP Request node response
  - Check Flask API logs for errors
  - Ensure workflow is active

### Issue: Quotes Not Generated
- **Solution**:
  - Verify OpenAI API key is valid
  - Check API quota/limits
  - Review Basic LLM Chain prompt
  - Check model availability (gpt-4.1-nano)

### Issue: Timestamp Format Incorrect
- **Solution**:
  - Verify JavaScript expression in Date & time node
  - Check timezone settings
  - Adjust format expression if needed

## Customization

### Change Schedule Frequency
1. Edit **Schedule Trigger** node
2. Modify interval (minutes, hours, days)
3. Or use cron expression for specific times

### Modify Quote Style
1. Edit **Basic LLM Chain** prompt
2. Change quote type (Rumi, general motivation, etc.)
3. Adjust length or tone requirements

### Change File Location
1. Update `FILE_PATH` in Flask script
2. Update directory path in n8n HTTP Request
3. Ensure directory exists

### Add Error Handling
1. Add Error Trigger node in n8n
2. Add try-catch in Flask script
3. Log errors to separate file

### Modify Timestamp Format
1. Edit JavaScript expression in Date & time node
2. Example formats:
   - ISO: `{{ new Date().toISOString() }}`
   - Custom: Adjust replace() functions

## Security Considerations
- **Local API**: Flask runs on localhost; consider authentication for production
- **File Permissions**: Restrict file directory access
- **API Keys**: Store OpenAI key securely in n8n credentials
- **Automation Safety**: Be cautious with pyautogui on production systems

## Important Notes
- **Windows Only**: This workflow requires Windows for Notepad automation
- **Desktop Access**: pyautogui needs access to control desktop
- **File Paths**: Use raw strings (r"...") in Python for Windows paths
- **Timing**: Adjust sleep intervals based on system speed
- **Workflow Status**: Currently `"active": false` - activate before use
- **Resource Usage**: Running every minute may consume resources; adjust schedule as needed

## Alternative Approaches
1. **Direct File Write**: Instead of Notepad, write directly to file using n8n's File node
2. **Different Editors**: Modify Flask script to use other text editors
3. **Cloud Storage**: Save quotes to Google Drive, Dropbox, etc.
4. **Database**: Store quotes in database instead of text file

