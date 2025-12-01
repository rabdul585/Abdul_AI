# Water Intake Tracker 💧 - Day 6 Challenge

A comprehensive water intake tracking application built with Streamlit to help you stay hydrated and reach your daily 3L goal!

## Features

### 🎯 Core Features
- **Daily Water Logging**: Track your water intake in milliliters
- **3L Daily Goal**: Automatic progress tracking towards 3000ml (3 Liters)
- **Weekly Chart**: Visual representation of 7-day hydration history
- **Progress Monitoring**: Real-time progress bar and percentage display
- **Persistent Storage**: Data saved locally in JSON format

### ⚡ Quick Add Buttons
- 🥤 250ml - Glass
- 🍶 500ml - Standard bottle
- 💧 750ml - Large bottle
- 🥛 1000ml - 1 Liter

### 📊 Statistics Dashboard
- Today's total intake
- Remaining amount to goal
- Goal progress percentage
- Weekly total consumption
- Daily average intake
- Number of goals met this week

### 📈 Visualization
- Interactive 7-day bar chart using Plotly
- Color-coded bars (green for goal met, blue for in-progress)
- Goal line indicator at 3000ml
- Hover tooltips with detailed information

### 💡 Additional Features
- Custom amount input (50-2000ml)
- Add/Remove/Reset functionality
- Hydration tips and benefits
- Recommended drinking schedule
- Interesting water facts
- Goal celebration when 3L reached

## Prerequisites

- Python 3.7 or higher
- Streamlit
- Plotly
- Pandas

## Installation

Install required dependencies:

```bash
pip install streamlit plotly pandas
```

## Usage

Run the Water Intake Tracker:

```bash
streamlit run Python_15_Days_Challenge/Day6/water_tracker.py
```

The app will open in your browser at `http://localhost:8501`.

## How to Use

### 1. Quick Add Water
Click on any of the quick-add buttons to instantly log common water amounts:
- Click "🥤 250ml Glass" for a standard glass of water
- Click "🍶 500ml Bottle" for a standard water bottle
- Click "💧 750ml Large" for a large bottle
- Click "🥛 1000ml Liter" for a full liter

### 2. Custom Amount
For precise tracking:
1. Enter any custom amount between 50-2000ml
2. Click "➕ Add" to add the amount
3. Click "➖ Remove" to subtract the amount
4. Click "🔄 Reset Today" to clear today's intake

### 3. Monitor Progress
- View real-time progress bar
- Check percentage towards 3L goal
- See remaining milliliters needed
- Celebrate when you hit 3000ml! 🎉

### 4. Weekly Analysis
- View 7-day bar chart of your hydration
- Check weekly total and daily average
- See how many days you met your goal
- Track your hydration patterns

## Data Storage

Water intake data is stored in:
```
Python_15_Days_Challenge/Day6/water_data.json
```

The JSON file structure:
```json
{
  "2025-11-19": 2500,
  "2025-11-18": 3200,
  "2025-11-17": 2800
}
```

## Layout

The app is divided into three main columns:

### Left Column (Today's Progress)
- Current intake display
- Progress bar
- Goal status
- Today's statistics

### Middle Column (Logging & Chart)
- Quick add buttons
- Custom amount input
- Add/Remove/Reset buttons
- Weekly hydration chart
- Weekly summary statistics

### Right Column (Tips & Info)
- Hydration benefits
- Tips to drink more water
- Recommended drinking schedule
- Interesting water facts

## Features Breakdown

### Progress Visualization
- **Goal Not Met**: Purple gradient card with current intake
- **Goal Reached**: Green gradient card with celebration message
- **Progress Bar**: Visual indicator from 0-100%
- **Percentage Display**: Exact percentage towards goal

### Weekly Chart
- **X-axis**: Days of the week (Last 7 days)
- **Y-axis**: Water intake in milliliters
- **Green Bars**: Days where 3L goal was met
- **Blue Bars**: Days still working towards goal
- **Red Dashed Line**: 3000ml goal marker
- **Hover Info**: Detailed intake per day

### Smart Features
- Prevents negative values
- Auto-saves after each action
- Immediate UI updates with `st.rerun()`
- Responsive design for different screen sizes
- Emoji-enhanced user interface

## Hydration Tips Included

### Benefits of Staying Hydrated
- Improves brain function
- Boosts energy levels
- Enhances physical performance
- Promotes healthy skin
- Aids in weight management
- Supports kidney function

### Tips to Drink More Water
- Keep a water bottle handy
- Set hourly reminders
- Drink before meals
- Add lemon or fruit for flavor
- Track your progress daily
- Make it a habit!

### Recommended Schedule
- Morning (6-9 AM): 500ml
- Mid-Morning (9-12 PM): 750ml
- Afternoon (12-3 PM): 750ml
- Evening (3-6 PM): 500ml
- Night (6-9 PM): 500ml

**Total: 3000ml (3 Liters)**

## Color Scheme

- **Primary**: Blue gradient (#667eea to #764ba2)
- **Success**: Green gradient (#11998e to #38ef7d)
- **Buttons**: Blue (#0066CC)
- **Chart - Goal Met**: Green (#38ef7d)
- **Chart - In Progress**: Blue (#667eea)
- **Goal Line**: Red (dashed)

## Technical Details

### State Management
- Uses `st.session_state` for in-memory data
- Automatically loads data from JSON on startup
- Saves to JSON after each modification

### Data Persistence
- JSON file created in Day6 folder
- Automatic directory creation if not exists
- Date-based keys (YYYY-MM-DD format)
- Integer values for water intake in ml

### Chart Configuration
- Plotly for interactive charts
- 400px height for optimal viewing
- Unified hover mode
- Outside text labels on bars
- Responsive width with `use_container_width=True`

## Example Workflow

1. **Morning**: Click "🥤 250ml" twice for breakfast drinks
2. **Mid-Morning**: Click "🍶 500ml" for your water bottle
3. **Lunch**: Add custom 300ml for your beverage
4. **Afternoon**: Click "🍶 500ml" again
5. **Evening**: Click "💧 750ml" for your large bottle
6. **Check Progress**: View your chart and statistics
7. **Celebrate**: Hit 3000ml and see the green success card! 🎉

## Troubleshooting

### Data not saving?
- Check if the Day6 folder exists
- Ensure you have write permissions
- Verify water_data.json is not read-only

### Chart not displaying?
- Ensure plotly is installed: `pip install plotly`
- Check browser console for errors
- Try refreshing the page

### Button clicks not working?
- Make sure all dependencies are installed
- Check terminal for error messages
- Ensure Streamlit version is up to date

## Customization

You can customize the daily goal by changing this line:

```python
DAILY_GOAL = 3000  # Change to your preferred goal in ml
```

Modify quick-add button amounts:

```python
# Change the values in the button sections
st.button("🥤 250ml\nGlass", key="add_250")  # Modify 250 to your preferred amount
```

## Run Command

```bash
cd Python_15_Days_Challenge/Day6
streamlit run water_tracker.py
```

## Future Enhancements

Potential features to add:
- Multiple users support
- Export data to CSV
- Monthly/yearly statistics
- Reminder notifications
- Dark/light theme toggle
- Mobile app version
- Integration with fitness apps
- Custom goal setting per user

## License

Free to use and modify for learning purposes.

---

**Challenge Completed**: Day 6 of 15 Days Python Streamlit Challenge ✅

Stay hydrated! 💧
