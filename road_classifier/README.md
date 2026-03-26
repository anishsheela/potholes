# 🛣️ Road Condition Classifier

Multi-user collaborative image classification tool with gamification, leaderboards, and achievements!

## 🎮 Features

- ✅ **Multi-user support** - Multiple people classify simultaneously
- 🏆 **Leaderboard** - Real-time rankings with gold/silver/bronze medals
- 🎯 **Achievement system** - Earn badges and points for milestones
- ⚡ **Keyboard shortcuts** - Press 1/2/3 for classifications, S to skip
- 📊 **Live statistics** - Track overall progress in real-time
- 🔄 **Smart assignment** - No two people get the same image
- 💾 **Auto-save** - All classifications saved to SQLite database
- 📤 **Export** - Download results as JSON

## 🚀 Quick Start

### 1. Setup

```bash
cd road_classifier

# Install dependencies
uv pip install flask

# Create images directory and add your road images
mkdir images
# Copy your road images to the images/ folder
```

### 2. Run the server

```bash
python app.py
```

### 3. Access the interface

The server will show you URLs like:
```
🌐 Share this URL with your friends:
   http://192.168.1.100:5000

💻 Local access:
   http://localhost:5000
```

Share the network URL with your 3-4 friends!

## 🎯 How to Use

1. **Enter your name** - No password required
2. **View image** - Road image appears
3. **Classify** - Click button or press:
   - `Q` for **Excellent** (perfect condition, no damage)
   - `W` for **Good** (minor wear, no potholes)
   - `E` for **Fair** (some cracks, 1-2 small potholes)
   - `R` for **Poor** (multiple/deep potholes, severe damage)
   - `Space` to **Skip** (uncertain/bad image)
4. **Earn points** - Complete achievements and climb the leaderboard!

## 🏅 Achievements

- 🎯 **First Steps** (10 pts) - Classify your first image
- ⚡ **Speed Demon** (50 pts) - Classify 10 images in under 5 minutes
- 🎖️ **Fifty Strong** (50 pts) - Classify 50 images
- 💯 **Century Club** (100 pts) - Classify 100 images
- 🔥 **Half K** (250 pts) - Classify 500 images
- ⭐ **Consistent** (60 pts) - Classify 30 images in one session
- 🌅 **Early Bird** (25 pts) - First person to start labeling

## 📊 Categories

**Excellent** (Q) - Perfect condition, smooth, no visible damage  
**Good** (W) - Minor wear, small cracks, no potholes  
**Fair** (E) - Visible cracks, 1-2 small potholes, still drivable  
**Poor** (R) - Multiple/deep potholes, severe damage, urgent repair needed

## 📥 Export Results

Visit `http://localhost:5000/api/export` to download all classifications as JSON.

Format:
```json
[
  {
    "image": "road001.jpg",
    "label": "Good",
    "username": "Alice",
    "timestamp": "2026-03-13 18:30:45"
  }
]
```

## 🗄️ Database

All data stored in `classifications.db` (SQLite):
- `classifications` - All image classifications
- `users` - User stats and achievements
- `active_sessions` - Who's labeling what (prevents collisions)

## 🔧 Technical Details

**Backend:**
- Python Flask
- SQLite database
- No authentication required

**Frontend:**
- Vanilla JavaScript
- Real-time updates
- Responsive design
- Keyboard shortcuts

## 💡 Tips

- **Keyboard shortcuts** are faster than clicking!
- **Timer** shows how long you're taking per image
- **Skip** uncertain images - consistency matters more than speed
- **Leaderboard** updates every 15 seconds
- Images are assigned uniquely - no duplicates across users

## 🐛 Troubleshooting

**No images showing?**
- Make sure images are in the `images/` directory
- Check file extensions (.jpg, .jpeg, .png, .bmp)

**Can't connect from another device?**
- Check firewall settings
- Make sure devices are on same network
- Use the IP address shown when starting the server

**Database errors?**
- Delete `classifications.db` and restart the server

## 📝 After Classification

Once all images are classified, use the exported JSON to:
1. Train a classification model (EfficientNet, ResNet, etc.)
2. Analyze road condition distribution
3. Generate reports for road maintenance

Enjoy! 🎉
