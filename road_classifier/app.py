#!/usr/bin/env python3
"""
Multi-user Road Condition Classifier
Features: Consensus Review Mode, Active Learning Support, Undo, Gamified Leaderboard & Streaks
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
                  last_active DATETIME,
                  current_streak INTEGER DEFAULT 0,
                  max_streak INTEGER DEFAULT 0)''')
                  
    # Retrofit existing dbs
    try:
        c.execute('ALTER TABLE users ADD COLUMN current_streak INTEGER DEFAULT 0')
        c.execute('ALTER TABLE users ADD COLUMN max_streak INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass # Columns already exist

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
    
    # 1. Look up existing classifications for this image
    c.execute('SELECT username, label FROM classifications WHERE image_name = ?', (image_name,))
    existing = c.fetchall()
    
    # 2. Insert the new classification
    c.execute('''INSERT INTO classifications (image_name, label, username, time_taken)
                 VALUES (?, ?, ?, ?)''', (image_name, label, username, time_taken))
                 
    new_achievements = []
    points_earned = 0
    streak_active = False
    resolved_tie = False
    matching_users = []
    
    if not existing:
        # First person to classify.
        # We don't break their streak because they don't know if they are right or wrong yet
        streak_active = True
        pass
    else:
        # 2nd or 3rd person
        matching_users = [row[0] for row in existing if row[1] == label]
        disagreeing_users = [row[0] for row in existing if row[1] != label]
        
        if matching_users:
            points_earned = 10
            # If there was a disagreement previously, we just broke the tie!
            if disagreeing_users:
                points_earned = 50
                resolved_tie = True
            streak_active = True
            # Retroactively reward the users we agreed with
            for m_user in matching_users:
                c.execute('UPDATE users SET points = points + 10 WHERE username = ?', (m_user,))
        else:
            # Disagreement! Break our streak
            c.execute('UPDATE users SET current_streak = 0 WHERE username = ?', (username,))
            streak_active = False
            
    # Update current user stats
    c.execute('''INSERT INTO users (username, total_labels, last_active, points, current_streak, max_streak)
                 VALUES (?, 1, ?, ?, ?, ?)
                 ON CONFLICT(username) DO UPDATE SET
                 total_labels = total_labels + 1,
                 last_active = ?,
                 points = points + ?,
                 current_streak = CASE WHEN ? THEN current_streak + 1 ELSE 0 END,
                 max_streak = CASE WHEN ? AND (current_streak + 1 > max_streak) THEN current_streak + 1 ELSE max_streak END
                 ''', (username, datetime.now(), points_earned, 1 if streak_active else 0, 1 if streak_active else 0,
                       datetime.now(), points_earned, streak_active, streak_active))
                       
    # Fetch new achievements / stats
    c.execute('SELECT points, current_streak, max_streak, achievements FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    current_points, current_streak, max_streak, ach_str = row
    current_achievements = json.loads(ach_str) if ach_str else []
    
    # Check new achievements
    if resolved_tie and 'tie_breaker' not in current_achievements:
        ach = {'id': 'tie_breaker', 'name': '⚖️ The Judge', 'description': 'Resolved a tied consensus', 'points': 50}
        new_achievements.append(ach)
        c.execute('UPDATE users SET points = points + 50 WHERE username = ?', (username,))
        current_achievements.append('tie_breaker')
        points_earned += 50
        
    if current_streak >= 10 and 'streak_10' not in current_achievements:
        ach = {'id': 'streak_10', 'name': '🔥 On Fire', 'description': '10 Consensus Matches in a Row!', 'points': 100}
        new_achievements.append(ach)
        c.execute('UPDATE users SET points = points + 100 WHERE username = ?', (username,))
        current_achievements.append('streak_10')
        points_earned += 100
        
    if current_streak >= 50 and 'perfectionist' not in current_achievements:
        ach = {'id': 'perfectionist', 'name': '👑 Perfectionist', 'description': '50 Match Streak', 'points': 500}
        new_achievements.append(ach)
        c.execute('UPDATE users SET points = points + 500 WHERE username = ?', (username,))
        current_achievements.append('perfectionist')
        points_earned += 500
        
    if new_achievements:
        c.execute('UPDATE users SET achievements = ? WHERE username = ?', (json.dumps(current_achievements), username))
        
    conn.commit()
    conn.close()
    
    return {
        'points_earned': points_earned, 
        'current_streak': current_streak if streak_active else 0,
        'new_achievements': new_achievements,
        'consensus_reached': True if matching_users else False,
        'streak_active': streak_active
    }

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
    consensus_images = set()
    for img, user_dict in img_classifications.items():
        labels = list(user_dict.values())
        for l in set(labels):
            if labels.count(l) >= 2:
                consensus_count += 1
                consensus_images.add(img)
                break

    on_disk_images = set(all_images)
    annotated_and_on_disk = len(consensus_images & on_disk_images)
    # True total = already annotated (including deleted from disk) + remaining on disk not yet annotated
    true_total = consensus_count + (total_images - annotated_and_on_disk)

    active_users = 0
    c.execute('SELECT COUNT(*) FROM active_sessions WHERE current_image IS NOT NULL')
    row = c.fetchone()
    if row: active_users = row[0]
    conn.close()

    return {
        'total_images': true_total,
        'classified': consensus_count,
        'remaining': true_total - consensus_count,
        'progress_percent': round((consensus_count / true_total * 100) if true_total > 0 else 0, 1),
        'active_users': active_users
    }

def get_leaderboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT username, total_labels, points, current_streak, max_streak, achievements
                 FROM users ORDER BY points DESC, total_labels DESC LIMIT 10''')
    leaderboard = []
    for row in c.fetchall():
        achievements = json.loads(row[5]) if row[5] else []
        leaderboard.append({
            'username': row[0],
            'total_labels': row[1],
            'points': row[2],
            'current_streak': row[3],
            'max_streak': row[4],
            'achievements_count': len(achievements)
        })
    conn.close()
    return leaderboard

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
    res = save_classification(data['image'], data['label'], data['username'], data.get('time_taken'))
    clear_active_session(data['username'])
    return jsonify({'success': True, 'result': res})

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

@app.route('/api/stats')
def stats():
    return jsonify(get_stats())

@app.route('/api/leaderboard')
def leaderboard_api():
    return jsonify(get_leaderboard())

@app.route('/api/contributors')
def contributors_api():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT username, total_labels FROM users
                 WHERE LOWER(username) != 'anish'
                 ORDER BY total_labels DESC''')
    result = [{'username': row[0], 'total_labels': row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify(result)

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
    <title>Road Classifier - Gamified Consensus</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f0f2f5; color: #333; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
        .panel { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        .header { text-align: center; margin-bottom: 20px; }
        .header h1 { color: #2c3e50; font-size: 2.2rem;}
        .btn { background: #3498db; color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 1.1rem; cursor: pointer; transition: transform 0.1s;}
        .btn:active { transform: scale(0.95); }
        
        /* Two Column Layout */
        .main-grid { display: grid; grid-template-columns: 3fr 1fr; gap: 20px; }
        @media (max-width: 1024px) { .main-grid { grid-template-columns: 1fr; } }

        .image-container { text-align: center; min-height: 500px; display: flex; align-items: center; justify-content: center; background: #e0e0e0; border-radius: 10px; margin: 20px 0; position: relative;}
        .image-container img { max-width: 100%; max-height: 70vh; border-radius: 5px; }

        .btn-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 20px; }
        .class-btn { padding: 15px 10px; font-size: 1.1rem; font-weight: bold; border: none; border-radius: 12px; cursor: pointer; transition: transform 0.1s; position: relative; color: white; }
        .class-btn:active { transform: scale(0.95); }
        .shortcut { position: absolute; top: 5px; right: 5px; font-size: 0.75rem; background: rgba(0,0,0,0.3); padding: 3px 6px; border-radius: 4px; }
        
        .excellent { background: #27ae60; }
        .good { background: #f1c40f; color: #333; }
        .fair { background: #e67e22; }
        .poor { background: #c0392b; }
        .invalid { background: #7f8c8d; }
        .util-btn { background: #bdc3c7; color: #333; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;}
        .util-btn:hover { background: #95a5a6; }

        /* Streak UI */
        .streak-container { font-size: 1.5rem; font-weight: bold; color: #e67e22; opacity: 0; transition: opacity 0.3s; text-align: right; margin-bottom: 10px;}
        .streak-container.on-fire { opacity: 1; animation: pulse 1s infinite alternate; }
        @keyframes pulse { 0% { transform: scale(1); } 100% { transform: scale(1.05); color: #c0392b; }}

        /* Achievements Popup */
        .toast-container { position: fixed; bottom: 20px; right: 20px; display: flex; flex-direction: column; gap: 10px; z-index: 1000; }
        .toast { background: white; border-left: 5px solid #27ae60; padding: 15px 25px; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); animation: slideIn 0.3s forwards, shrinkOut 0.3s 4s forwards; }
        .toast h4 { margin: 0 0 5px 0; color: #2c3e50; }
        .toast p { margin: 0; font-size: 0.9rem; color: #7f8c8d; }
        @keyframes slideIn { from{ transform: translateX(110%); } to{ transform: translateX(0); } }
        @keyframes shrinkOut { from{ opacity: 1; transform: scale(1); } to{ opacity: 0; transform: scale(0.8); } }

        /* Leaderboard */
        .lb-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; margin-bottom: 8px; background: #f8f9fa; border-radius: 8px; }
        .lb-rank { font-weight: bold; font-size: 1.2rem; width: 30px; }
        .lb-name { flex-grow: 1; font-weight: 600; }
        .lb-stats { text-align: right; }
        .lb-pts { color: #27ae60; font-weight: bold; font-size: 1.1rem; }
        .lb-streak { color: #e67e22; font-size: 0.8rem; font-weight: bold; }

        .guidelines { margin-top: 20px; padding: 20px; background: #e8f4fd; border-radius: 8px; }
        .rules-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-top: 15px; }
        .rule-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #dce4ec; display: flex; flex-direction: column; }
        .rule-card h4 { margin-bottom: 5px; font-size: 1.1rem; }
        .rule-card h4.exc { color: #27ae60; }
        .rule-card h4.gd { color: #f39c12; }
        .rule-card h4.fr { color: #d35400; }
        .rule-card h4.pr { color: #c0392b; }
        .rule-card h4.inv { color: #7f8c8d; }
        .rule-card p { font-size: 0.9rem; color: #555; flex-grow: 1; }
        .rule-card img { width: 100%; height: 110px; object-fit: cover; border-radius: 6px; margin-top: 10px; cursor: pointer; transition: transform 0.2s; border: 1px solid #eee; }
        .rule-card img:hover { transform: scale(1.03); box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        
        /* Lightbox */
        #lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 9999; align-items: center; justify-content: center; cursor: pointer; }
        #lightbox img { max-width: 90%; max-height: 90%; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
        
        .progress-bar { background: #ddd; height: 15px; border-radius: 8px; overflow: hidden; margin-bottom: 10px; }
        .progress-fill { background: #3498db; height: 100%; width: 0%; transition: width 0.3s; text-align: center; color: white; font-size: 0.7rem; font-weight: bold; line-height: 15px; }

        /* Thank-you banner */
        .thanks-banner { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); border-radius: 12px; padding: 30px 35px; margin-bottom: 30px; text-align: center; }
        .thanks-banner h2 { font-size: 1.6rem; color: #fff; margin-bottom: 10px; text-shadow: 0 1px 3px rgba(0,0,0,0.15); }
        .thanks-banner p { color: rgba(255,255,255,0.95); font-size: 1rem; max-width: 620px; margin: 0 auto 20px auto; line-height: 1.6; }
        .thanks-banner .signature { color: rgba(255,255,255,0.85); font-style: italic; font-size: 0.95rem; margin-bottom: 25px; }
        .contributors { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 5px; }
        .contributor-card { background: rgba(255,255,255,0.35); backdrop-filter: blur(4px); border-radius: 10px; padding: 10px 16px; text-align: center; max-width: 130px; }
        .contributor-card .c-name { font-weight: 700; font-size: 0.95rem; color: #fff; text-transform: capitalize; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 110px; }
        .contributor-card .c-count { font-size: 0.8rem; color: rgba(255,255,255,0.9); margin-top: 2px; }
        .contributor-card.hidden { display: none; }
        .see-more-btn { background: rgba(255,255,255,0.25); border: 1px solid rgba(255,255,255,0.6); color: #fff; border-radius: 20px; padding: 6px 18px; font-size: 0.85rem; cursor: pointer; margin-top: 12px; transition: background 0.2s; }
        .see-more-btn:hover { background: rgba(255,255,255,0.4); }
    </style>
</head>
<body>
<div class="toast-container" id="toast-container"></div>

<div id="lightbox" onclick="closeLightbox()">
    <img id="lightbox-img" src="">
</div>

<div class="container">
    <div id="login-screen" class="panel">

        <div class="thanks-banner">
            <h2>🙏 Thank You to Our Collaborators</h2>
            <p>This project wouldn't have been possible without your time and careful judgement. Every label you submitted has directly contributed to training an AI model that assesses real road conditions.</p>
            <p class="signature">Your work is now part of my research. Thank you from the bottom of my heart &mdash; Anish Anilkumar</p>
            <a href="https://roads.anishsheela.com/" target="_blank" rel="noopener" style="display:inline-block; margin-bottom: 18px; background: rgba(255,255,255,0.9); color: #d35400; font-weight: 700; font-size: 1rem; padding: 10px 24px; border-radius: 25px; text-decoration: none; box-shadow: 0 2px 10px rgba(0,0,0,0.15); transition: background 0.2s;" onmouseover="this.style.background='#fff'" onmouseout="this.style.background='rgba(255,255,255,0.9)'">🗺️ View the Research &amp; Live Map &rarr; roads.anishsheela.com</a>
            <div class="contributors" id="contributors-list">
                <div class="contributor-card" style="color:rgba(255,255,255,0.7); font-size:0.9rem;">Loading...</div>
            </div>
            <button class="see-more-btn" id="see-more-btn" style="display:none;" onclick="toggleContributors()"></button>
        </div>

        <div class="header">
            <h1>🚦 Road Validation: Consensus & Glory</h1>
            <p>You only earn points if someone else agrees with you. Be accurate.</p>
        </div>
        <div style="text-align: center; margin: 30px 0;">
            <input type="text" id="username-input" placeholder="Enter your name" style="padding: 12px; font-size: 1.1rem; border-radius: 6px; border: 1px solid #ccc;">
            <button class="btn" onclick="start()">Join the Fight</button>
        </div>

        <div class="guidelines">
            <h2 style="text-align: center; color: #2c3e50;">The Rules of the Road</h2>
            <div class="rules-grid">
                <div class="rule-card">
                    <h4 class="exc">✨ Excellent</h4>
                    <p>Flawless, new road. No patches/cracks.</p>
                    <img src="/sample/excellent.jpg" onclick="openLightbox(this.src)" onerror="this.style.display='none'">
                </div>
                <div class="rule-card">
                    <h4 class="gd">✅ Good</h4>
                    <p>Minor wear, hairline cracks. Perfectly comfortable.</p>
                    <img src="/sample/good.jpg" onclick="openLightbox(this.src)" onerror="this.style.display='none'">
                </div>
                <div class="rule-card">
                    <h4 class="fr">⚠️ Fair</h4>
                    <p>Noticeable bumps/patches. You feel it, no swerving.</p>
                    <img src="/sample/fair.jpg" onclick="openLightbox(this.src)" onerror="this.style.display='none'">
                </div>
                <div class="rule-card">
                    <h4 class="pr">❌ Poor</h4>
                    <p>Severe damage. Deep potholes requiring braking.</p>
                    <img src="/sample/poor.jpg" onclick="openLightbox(this.src)" onerror="this.style.display='none'">
                </div>
                <div class="rule-card">
                    <h4 class="inv">🚫 Invalid</h4>
                    <p>Pitch black, heavy rain, sky, garage, non-road.</p>
                    <img src="/sample/invalid.jpg" onclick="openLightbox(this.src)" onerror="this.style.display='none'">
                </div>
            </div>
        </div>
    </div>

    <div id="classifier-screen" class="main-grid" style="display: none;">
        <!-- Left Column: Classifier -->
        <div class="panel">
            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                <div style="flex-grow: 1; padding-right: 20px;">
                    <div style="font-size: 0.9rem; margin-bottom: 5px; color: #7f8c8d;">Consensus Progress: <span id="stat-completed">0</span> Images</div>
                    <div class="progress-bar"><div class="progress-fill" id="progress-fill">0%</div></div>
                </div>
                <div class="streak-container" id="streak-counter">🔥 Streak: <span id="streak-val">0</span></div>
            </div>
            
            <div class="btn-grid">
                <button class="class-btn excellent" onclick="classify('Excellent')"><span class="shortcut">Q</span>✨ Excellent</button>
                <button class="class-btn good" onclick="classify('Good')"><span class="shortcut">W</span>✅ Good</button>
                <button class="class-btn fair" onclick="classify('Fair')"><span class="shortcut">E</span>⚠️ Fair</button>
                <button class="class-btn poor" onclick="classify('Poor')"><span class="shortcut">R</span>❌ Poor</button>
                <button class="class-btn invalid" onclick="classify('Invalid')"><span class="shortcut">T / Sp</span >🚫 Invalid</button>
            </div>

            <button class="util-btn" onclick="undoLast()">↩️ Undo Last (Z / Backspace)</button>

            <div class="image-container" id="image-container"><p>Loading...</p></div>
        </div>

        <!-- Right Column: Leaderboard -->
        <div class="panel" style="padding: 15px;">
            <h3 style="margin-bottom: 15px; color: #2c3e50; text-align: center;">🏆 Top Annotators</h3>
            <div id="leaderboard-container">
                <p style="text-align: center; color: #999;">Loading...</p>
            </div>
            <hr style="margin: 20px 0; border: 0; border-top: 1px solid #eee;">
            <div style="text-align: center; font-size: 0.9rem; color: #7f8c8d;">
                Active Users: <strong id="active-users-count">0</strong>
            </div>
        </div>
    </div>
</div>

<script>
    let username = '';
    let currentImage = null;
    let startTime = null;

    const CONTRIBUTORS_VISIBLE = 6;
    let contributorsExpanded = false;

    async function loadContributors() {
        try {
            const res = await fetch('/api/contributors');
            const data = await res.json();
            const container = document.getElementById('contributors-list');
            const btn = document.getElementById('see-more-btn');
            if (!data || data.length === 0) { container.innerHTML = ''; return; }

            container.innerHTML = data.map((u, i) => `
                <div class="contributor-card${i >= CONTRIBUTORS_VISIBLE ? ' hidden' : ''}">
                    <div class="c-name">${u.username}</div>
                    <div class="c-count">${u.total_labels} labels</div>
                </div>
            `).join('');

            if (data.length > CONTRIBUTORS_VISIBLE) {
                btn.style.display = 'inline-block';
                btn.textContent = `See all ${data.length} contributors ▾`;
            }
        } catch(e) { console.error(e); }
    }

    function toggleContributors() {
        contributorsExpanded = !contributorsExpanded;
        document.querySelectorAll('.contributor-card.hidden, .contributor-card').forEach((el, i) => {
            if (i >= CONTRIBUTORS_VISIBLE) {
                el.classList.toggle('hidden', !contributorsExpanded);
            }
        });
        const btn = document.getElementById('see-more-btn');
        const total = document.querySelectorAll('.contributor-card').length;
        btn.textContent = contributorsExpanded ? 'Show less ▴' : `See all ${total} contributors ▾`;
    }

    // Populate contributors immediately on page load
    loadContributors();

    function start() {
        const val = document.getElementById('username-input').value.trim();
        if(!val) return;
        username = val;
        document.getElementById('login-screen').style.display = 'none';
        document.getElementById('classifier-screen').style.display = 'grid'; // because main-grid
        loadNextImage();
        updateStats();
        updateLeaderboard();
        setInterval(updateStats, 5000);
        setInterval(updateLeaderboard, 10000);
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

    function showToast(title, msg) {
        const cont = document.getElementById('toast-container');
        const d = document.createElement('div');
        d.className = 'toast';
        d.innerHTML = `<h4>${title}</h4><p>${msg}</p>`;
        cont.appendChild(d);
        setTimeout(() => cont.removeChild(d), 4500);
    }

    function handleResult(res) {
        const streakEl = document.getElementById('streak-counter');
        const streakVal = document.getElementById('streak-val');
        
        // Update Streak UI
        if(res.streak_active && res.current_streak > 0) {
            streakEl.classList.add('on-fire');
            streakVal.innerText = res.current_streak;
        } else {
            streakEl.classList.remove('on-fire');
        }

        // Show generic Consensus Toast if points earned but no big achievement mapping
        if(res.consensus_reached && res.points_earned > 0) {
            showToast("✅ Consensus Reached!", `You matched someone! +${res.points_earned} Points`);
        }

        // Show Achievements
        if(res.new_achievements && res.new_achievements.length > 0) {
            for(let ach of res.new_achievements) {
                showToast(ach.name, `${ach.description} (+${ach.points} Pts)`);
            }
        }
    }

    async function classify(label) {
        if(!currentImage) return;
        const timeTaken = (Date.now() - startTime)/1000;
        try {
            const req = await fetch('/api/classify', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: currentImage, label, username, time_taken: timeTaken})
            });
            const body = await req.json();
            if(body.success) handleResult(body.result);
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
                showToast("Undo Successful", "Your last vote was deleted.");
                updateStats();
            } else {
                showToast("Nothing to Undo", "No recent actions found.");
            }
        } catch(e) { console.error(e); }
    }

    async function updateStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            document.getElementById('stat-completed').innerText = `${data.classified} / ${data.total_images}`;
            document.getElementById('active-users-count').innerText = data.active_users;
            document.getElementById('progress-fill').style.width = `${data.progress_percent}%`;
            document.getElementById('progress-fill').innerText = `${data.progress_percent}%`;
        } catch(e) { console.error(e); }
    }

    async function updateLeaderboard() {
        try {
            const res = await fetch('/api/leaderboard');
            const data = await res.json();
            const lb = document.getElementById('leaderboard-container');
            if(data.length === 0) {
                lb.innerHTML = '<p style="text-align: center; color: #999;">No contenders yet.</p>';
                return;
            }
            lb.innerHTML = data.map((u, i) => {
                const rankIcon = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `${i+1}.`;
                return `
                    <div class="lb-item">
                        <div class="lb-rank">${rankIcon}</div>
                        <div class="lb-name">${u.username}<br><span style="font-size:0.75rem; color:#aaa">${u.total_labels} labeled</span></div>
                        <div class="lb-stats">
                            <div class="lb-pts">${u.points} pts</div>
                            <div class="lb-streak">🔥 Max: ${u.max_streak}</div>
                        </div>
                    </div>
                `;
            }).join('');
        } catch(e) { console.error(e); }
    }

    function openLightbox(src) {
        document.getElementById('lightbox-img').src = src;
        document.getElementById('lightbox').style.display = 'flex';
    }

    function closeLightbox() {
        document.getElementById('lightbox').style.display = 'none';
        document.getElementById('lightbox-img').src = '';
    }

    document.addEventListener('keydown', (e) => {
        // Prevent hotkeys if lightbox is open
        if(document.getElementById('lightbox').style.display === 'flex') {
            if(e.key === 'Escape') closeLightbox();
            return;
        }

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
    print("🚀 Gamified Road Consensus Classifier Starting!")
    print(f"📁 Root Data Directory: {IMAGES_DIR}")
    print(f"🌐 Access locally via http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
