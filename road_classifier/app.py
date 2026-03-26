#!/usr/bin/env python3
"""
Multi-user Road Condition Classifier
Features: Consensus Review Mode, Active Learning Support, Undo, Leaderboard
"""

from flask import Flask, render_template_string, request, jsonify, send_from_directory
import sqlite3
import os
import json
from datetime import datetime
import random
import argparse

app = Flask(__name__)

# Parse command line arguments for directories
parser = argparse.ArgumentParser(description="Multi-user Road Condition Classifier")
parser.add_argument('--data-dir', type=str, default=os.path.join(os.getcwd(), 'processed_data', 'frames'), help='Path to images directory')
args, unknown = parser.parse_known_args()

IMAGES_DIR = args.data_dir
DB_PATH = 'classifications.db'

# Reward thresholds and achievements
ACHIEVEMENTS = {
    'first_label': {'name': '🎯 First Steps', 'description': 'Classified your first image', 'points': 10},
    'speed_demon': {'name': '⚡ Speed Demon', 'description': 'Classified 10 images in under 5 minutes', 'points': 50},
    'consistency': {'name': '⭐ Analyst', 'description': 'Completed 50 consensus reviews', 'points': 100},
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS classifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_name TEXT NOT NULL,
                  label TEXT NOT NULL,
                  username TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  time_taken REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS active_sessions
                 (username TEXT PRIMARY KEY,
                  current_image TEXT,
                  started_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY,
                  total_labels INTEGER DEFAULT 0,
                  points INTEGER DEFAULT 0,
                  achievements TEXT DEFAULT '[]',
                  session_count INTEGER DEFAULT 0,
                  last_active DATETIME)''')
    conn.commit()
    conn.close()

def get_all_images():
    if not os.path.exists(IMAGES_DIR):
        print(f"Warning: Directory {IMAGES_DIR} not found. Creating it.")
        os.makedirs(IMAGES_DIR, exist_ok=True)
        return []
    
    images = []
    for root, _, files in os.walk(IMAGES_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                rel_path = os.path.relpath(os.path.join(root, f), IMAGES_DIR)
                rel_path = rel_path.replace(os.sep, '/')
                images.append(rel_path)
    return sorted(images)

def get_next_image(username):
    """
    CONSENSUS REVIEW MODE
    Requirements for an image to be 'Available':
    1. The current user has not labeled it.
    2. It does not already have 2 matching labels from ANY users.
    Prioritizes images that already have 1 label to finalize consensus quickly.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    all_images = get_all_images()
    
    c.execute('SELECT image_name, username, label FROM classifications')
    rows = c.fetchall()
    
    img_classifications = {}
    for img, u, label in rows:
        if img not in img_classifications:
            img_classifications[img] = {}
        img_classifications[img][u] = label
        
    c.execute('SELECT current_image FROM active_sessions WHERE username != ?', (username,))
    active = set(row[0] for row in c.fetchall() if row[0])
    conn.close()
    
    available = []
    partially_done = []
    
    for img in all_images:
        if img in active:
            continue
            
        users_who_classified = img_classifications.get(img, {})
        if username in users_who_classified:
            continue
            
        # Check consensus (2 matching labels)
        labels = list(users_who_classified.values())
        consensus = False
        for l in set(labels):
            if labels.count(l) >= 2:
                consensus = True
                break
                
        if not consensus:
            available.append(img)
            if len(users_who_classified) > 0:
                partially_done.append(img)
    
    # Priority: Images that just need one more vote to finish consensus
    if partially_done:
        return random.choice(partially_done)
    if available:
        return random.choice(available)
    return None

def update_active_session(username, image_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO active_sessions (username, current_image, started_at)
                 VALUES (?, ?, ?)''', (username, image_name, datetime.now()))
    conn.commit()
    conn.close()

def clear_active_session(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM active_sessions WHERE username = ?', (username,))
    conn.commit()
    conn.close()

def save_classification(image_name, label, username, time_taken):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO classifications (image_name, label, username, time_taken)
                 VALUES (?, ?, ?, ?)''', (image_name, label, username, time_taken))
    c.execute('''INSERT INTO users (username, total_labels, last_active)
                 VALUES (?, 1, ?)
                 ON CONFLICT(username) DO UPDATE SET
                 total_labels = total_labels + 1,
                 last_active = ?''', (username, datetime.now(), datetime.now()))
    c.execute('SELECT total_labels FROM users WHERE username = ?', (username,))
    total_labels = c.fetchone()[0]
    conn.commit()
    conn.close()
    # Mocking achievements for now
    return []

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    all_images = get_all_images()
    total_images = len(all_images)
    
    c.execute('SELECT image_name, username, label FROM classifications')
    rows = c.fetchall()
    
    img_classifications = {}
    for img, u, label in rows:
        if img not in img_classifications:
            img_classifications[img] = {}
        img_classifications[img][u] = label
        
    consensus_count = 0
    for img, user_dict in img_classifications.items():
        labels = list(user_dict.values())
        for l in set(labels):
            if labels.count(l) >= 2:
                consensus_count += 1
                break
                
    total_users = 0
    c.execute('SELECT COUNT(DISTINCT username) FROM users')
    row = c.fetchone()
    if row: total_users = row[0]
    
    c.execute('SELECT label, COUNT(*) FROM classifications GROUP BY label')
    label_distribution = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute('SELECT COUNT(*) FROM active_sessions WHERE current_image IS NOT NULL')
    active_users = c.fetchone()[0]
    conn.close()
    
    return {
        'total_images': total_images,
        'classified': consensus_count,
        'remaining': total_images - consensus_count,
        'progress_percent': round((consensus_count / total_images * 100) if total_images > 0 else 0, 1),
        'total_users': total_users,
        'active_users': active_users,
        'label_distribution': label_distribution,
        'raw_classifications': len(rows)
    }

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/next_image', methods=['POST'])
def next_image():
    username = request.json.get('username', 'Anonymous')
    if not username: return jsonify({'error': 'Username required'}), 400
    
    image = get_next_image(username)
    if image:
        update_active_session(username, image)
        return jsonify({'image': image, 'url': f'/images/{image}'})
    else:
        clear_active_session(username)
        return jsonify({'completed': True, 'message': 'Dataset Complete & Consensus Reached! 🎉'})

@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.json
    if not all([data.get('image'), data.get('label'), data.get('username')]):
        return jsonify({'error': 'Missing required fields'}), 400
    save_classification(data['image'], data['label'], data['username'], data.get('time_taken'))
    clear_active_session(data['username'])
    return jsonify({'success': True, 'new_achievements': []})

@app.route('/api/undo', methods=['POST'])
def undo():
    username = request.json.get('username', 'Anonymous')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, image_name FROM classifications WHERE username = ? ORDER BY timestamp DESC LIMIT 1', (username,))
    row = c.fetchone()
    if row:
        class_id, img_name = row
        c.execute('DELETE FROM classifications WHERE id = ?', (class_id,))
        c.execute('UPDATE users SET total_labels = total_labels - 1 WHERE username = ?', (username,))
        conn.commit()
        update_active_session(username, img_name)
        conn.close()
        return jsonify({'success': True, 'image': img_name, 'url': f'/images/{img_name}'})
    conn.close()
    return jsonify({'success': False, 'error': 'No recent actions to undo.'})

@app.route('/api/skip', methods=['POST'])
def skip_image():
    clear_active_session(request.json.get('username', 'Anonymous'))
    return jsonify({'success': True})

@app.route('/api/stats')
def stats():
    return jsonify(get_stats())

@app.route('/images/<path:filename>')
def serve_image(filename):
    directory = os.path.abspath(os.path.join(IMAGES_DIR, os.path.dirname(filename)))
    file = os.path.basename(filename)
    return send_from_directory(directory, file)

@app.route('/sample/<path:filename>')
def serve_sample(filename):
    directory = os.path.abspath(os.path.join(os.getcwd(), 'sample_image'))
    return send_from_directory(directory, filename)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Road Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f0f2f5; color: #333; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .panel { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        .header { text-align: center; margin-bottom: 20px; }
        .header h1 { color: #2c3e50; }
        .btn { background: #3498db; color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 1.1rem; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        
        .image-container { text-align: center; min-height: 500px; display: flex; align-items: center; justify-content: center; background: #e0e0e0; border-radius: 10px; margin: 20px 0; }
        .image-container img { max-width: 100%; max-height: 70vh; border-radius: 5px; }

        .btn-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px; }
        .class-btn { padding: 20px; font-size: 1.2rem; font-weight: bold; border: none; border-radius: 12px; cursor: pointer; transition: transform 0.1s; position: relative; color: white; }
        .class-btn:active { transform: scale(0.95); }
        .shortcut { position: absolute; top: 10px; right: 10px; font-size: 0.8rem; background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px; }
        
        .excellent { background: #27ae60; }
        .good { background: #f1c40f; color: #333; }
        .fair { background: #e67e22; }
        .poor { background: #c0392b; }
        .invalid { background: #7f8c8d; }

        .util-btn { background: #bdc3c7; color: #333; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; }
        
        .guidelines { margin-top: 30px; padding: 20px; background: #e8f4fd; border-left: 5px solid #3498db; border-radius: 5px; }
        .guideline-images { display: flex; gap: 15px; margin-top: 15px; overflow-x: auto;}
        .guideline-images img { height: 150px; border-radius: 8px; border: 2px solid #ccc; }
        
        .progress-bar { background: #ddd; height: 20px; border-radius: 10px; overflow: hidden; margin-bottom: 20px; }
        .progress-fill { background: #27ae60; height: 100%; width: 0%; transition: 0.3s; text-align: center; color: white; font-size: 0.8rem; font-weight: bold;}
    </style>
</head>
<body>
<div class="container">
    <div id="login-screen" class="panel">
        <div class="header">
            <h1>Road Condition Consensus Annotator</h1>
            <p>Enter your username to begin auditing images.</p>
        </div>
        <div style="text-align: center; margin: 30px 0;">
            <input type="text" id="username-input" placeholder="Your Name" style="padding: 12px; font-size: 1.1rem; border-radius: 6px; border: 1px solid #ccc;">
            <button class="btn" onclick="start()">Start Annotating</button>
        </div>

        <div class="guidelines">
            <h2>Annotation Guidelines</h2>
            <ul>
                <li><b>Excellent:</b> Flawless, new road. No patches/cracks.</li>
                <li><b>Good:</b> Minor wear, hairline cracks. Perfectly comfortable.</li>
                <li><b>Fair:</b> Noticeable bumps/patches. You feel it, but no swerving.</li>
                <li><b>Poor:</b> Severe damage. Deep potholes requiring braking/swerving.</li>
                <li><b>Invalid:</b> Pitch black, heavy rain, sky, garage, non-road.</li>
            </ul>
            <div class="guideline-images">
                <div><i>Excellent Validation</i><br><img src="/sample/excellent.jpg" onerror="this.style.display='none'"></div>
                <div><i>Good Validation</i><br><img src="/sample/good.jpg" onerror="this.style.display='none'"></div>
                <div><i>Fair Validation</i><br><img src="/sample/fair.jpg" onerror="this.style.display='none'"></div>
                <div><i>Poor Validation</i><br><img src="/sample/poor.jpg" onerror="this.style.display='none'"></div>
            </div>
            <p style="margin-top: 10px; color: #c0392b;"><i>Note: This system runs in Consensus Mode. Images require 2 matching labels from different users to finalize.</i></p>
        </div>
    </div>

    <div id="classifier-screen" style="display: none;">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill">0%</div></div>
        
        <div class="panel">
            <div class="btn-grid">
                <button class="class-btn excellent" onclick="classify('Excellent')"><span class="shortcut">Q</span>✨ Excellent</button>
                <button class="class-btn good" onclick="classify('Good')"><span class="shortcut">W</span>✅ Good</button>
                <button class="class-btn fair" onclick="classify('Fair')"><span class="shortcut">E</span>⚠️ Fair</button>
                <button class="class-btn poor" onclick="classify('Poor')"><span class="shortcut">R</span>❌ Poor</button>
                <button class="class-btn invalid" onclick="classify('Invalid')"><span class="shortcut">T / Space</span>🚫 Invalid</button>
            </div>

            <div style="display: flex; justify-content: space-between;">
                <button class="util-btn" onclick="undoLast()"><span style="font-weight:bold;">Z / Backspace</span> ↩️ Undo Last</button>
                <div style="font-weight: bold; color: #3498db;">Consensus Completed: <span id="stat-completed">0</span></div>
            </div>

            <div class="image-container" id="image-container">
                <p>Loading...</p>
            </div>
        </div>
    </div>
</div>

<script>
    let username = '';
    let currentImage = null;
    let startTime = null;

    function start() {
        const val = document.getElementById('username-input').value.trim();
        if(!val) return;
        username = val;
        document.getElementById('login-screen').style.display = 'none';
        document.getElementById('classifier-screen').style.display = 'block';
        loadNextImage();
        updateStats();
        setInterval(updateStats, 5000);
    }

    async function loadNextImage() {
        try {
            const res = await fetch('/api/next_image', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username})
            });
            const data = await res.json();
            if(data.completed) {
                document.getElementById('image-container').innerHTML = `<h2>🎉 ${data.message}</h2>`;
                currentImage = null;
                return;
            }
            currentImage = data.image;
            document.getElementById('image-container').innerHTML = `<img src="${data.url}">`;
            startTime = Date.now();
        } catch(e) { console.error(e); }
    }

    async function classify(label) {
        if(!currentImage) return;
        const timeTaken = (Date.now() - startTime)/1000;
        try {
            await fetch('/api/classify', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: currentImage, label, username, time_taken: timeTaken})
            });
            loadNextImage();
            updateStats();
        } catch(e) { console.error(e); }
    }

    async function undoLast() {
        try {
            const res = await fetch('/api/undo', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username})
            });
            const data = await res.json();
            if(data.success) {
                currentImage = data.image;
                document.getElementById('image-container').innerHTML = `<img src="${data.url}">`;
                startTime = Date.now();
                updateStats();
            } else {
                alert("Nothing to undo!");
            }
        } catch(e) { console.error(e); }
    }

    async function updateStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            document.getElementById('stat-completed').innerText = `${data.classified} / ${data.total_images}`;
            document.getElementById('progress-fill').style.width = `${data.progress_percent}%`;
            document.getElementById('progress-fill').innerText = `${data.progress_percent}%`;
        } catch(e) { console.error(e); }
    }

    document.addEventListener('keydown', (e) => {
        if(document.getElementById('login-screen').style.display !== 'none') return;
        
        switch(e.key.toLowerCase()) {
            case 'q': classify('Excellent'); break;
            case 'w': classify('Good'); break;
            case 'e': classify('Fair'); break;
            case 'r': classify('Poor'); break;
            case 't': 
            case ' ': e.preventDefault(); classify('Invalid'); break;
            case 'z':
            case 'backspace': e.preventDefault(); undoLast(); break;
        }
    });
</script>
</body>
</html>
"""

if __name__ == '__main__':
    init_db()
    print("🚀 Road Consensus Classifier Starting!")
    print(f"📁 Root Data Directory: {IMAGES_DIR}")
    app.run(host='0.0.0.0', port=5000, debug=True)
