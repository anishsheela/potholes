#!/usr/bin/env python3
"""
Multi-user Road Condition Classifier
Features: Leaderboard, rewards, real-time progress tracking
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import sqlite3
import os
import json
from datetime import datetime
import random

app = Flask(__name__)

import argparse
# Parse command line arguments for directories
parser = argparse.ArgumentParser(description="Multi-user Road Condition Classifier")
parser.add_argument('--unfiltered-dir', type=str, default=os.path.join(os.getcwd(), 'unfiltered_images'), help='Path to unfiltered/raw frames')
parser.add_argument('--filtered-dir', type=str, default=os.path.join(os.getcwd(), 'images'), help='Path to filtered frames')
parser.add_argument('--use-filtered', action='store_true', help='Use filtered frames instead of unfiltered')

args, unknown = parser.parse_known_args()

# Configuration
if args.use_filtered:
    IMAGES_DIR = args.filtered_dir
else:
    IMAGES_DIR = args.unfiltered_dir

DB_PATH = 'classifications.db'

# Reward thresholds and achievements
ACHIEVEMENTS = {
    'first_label': {'name': '🎯 First Steps', 'description': 'Classified your first image', 'points': 10},
    'speed_demon': {'name': '⚡ Speed Demon', 'description': 'Classified 10 images in under 5 minutes', 'points': 50},
    'fifty': {'name': '🎖️ Fifty Strong', 'description': 'Classified 50 images', 'points': 50},
    'century': {'name': '💯 Century Club', 'description': 'Classified 100 images', 'points': 100},
    'half_k': {'name': '🔥 Half K', 'description': 'Classified 500 images', 'points': 250},
    'consistency': {'name': '⭐ Consistent', 'description': 'Classified 30 images in one session', 'points': 60},
    'early_bird': {'name': '🌅 Early Bird', 'description': 'First person to start labeling', 'points': 25},
}

def init_db():
    """Initialize database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Classifications table
    c.execute('''CREATE TABLE IF NOT EXISTS classifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_name TEXT NOT NULL,
                  label TEXT NOT NULL,
                  username TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  time_taken REAL)''')
    
    # Active sessions (who's labeling what right now)
    c.execute('''CREATE TABLE IF NOT EXISTS active_sessions
                 (username TEXT PRIMARY KEY,
                  current_image TEXT,
                  started_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # User stats and achievements
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
    """Get list of all images in the images directory"""
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        return []
    
    images = []
    for root, _, files in os.walk(IMAGES_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Get relative path so it can handle nested structures like 'anish/video1/frame.jpg'
                rel_path = os.path.relpath(os.path.join(root, f), IMAGES_DIR)
                # Ensure we use forward slashes for URLs/DB consistency across platforms
                rel_path = rel_path.replace(os.sep, '/')
                images.append(rel_path)
    return sorted(images)

def get_next_image(username):
    """Get next unclassified image for a user, avoiding images being worked on by others"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    all_images = get_all_images()
    
    # Get already classified images
    c.execute('SELECT DISTINCT image_name FROM classifications')
    classified = set(row[0] for row in c.fetchall())
    
    # Get images currently being worked on by others
    c.execute('SELECT current_image FROM active_sessions WHERE username != ?', (username,))
    active = set(row[0] for row in c.fetchall() if row[0])
    
    conn.close()
    
    # Find available images
    available = [img for img in all_images if img not in classified and img not in active]
    
    if available:
        return random.choice(available)
    return None

def update_active_session(username, image_name):
    """Mark that a user is currently working on an image"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''INSERT OR REPLACE INTO active_sessions (username, current_image, started_at)
                 VALUES (?, ?, ?)''', (username, image_name, datetime.now()))
    
    conn.commit()
    conn.close()

def clear_active_session(username):
    """Clear user's active session"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM active_sessions WHERE username = ?', (username,))
    conn.commit()
    conn.close()

def save_classification(image_name, label, username, time_taken):
    """Save a classification and update user stats"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Save classification
    c.execute('''INSERT INTO classifications (image_name, label, username, time_taken)
                 VALUES (?, ?, ?, ?)''', (image_name, label, username, time_taken))
    
    # Update user stats
    c.execute('''INSERT INTO users (username, total_labels, last_active)
                 VALUES (?, 1, ?)
                 ON CONFLICT(username) DO UPDATE SET
                 total_labels = total_labels + 1,
                 last_active = ?''', (username, datetime.now(), datetime.now()))
    
    # Get updated user stats
    c.execute('SELECT total_labels FROM users WHERE username = ?', (username,))
    total_labels = c.fetchone()[0]
    
    conn.commit()
    conn.close()
    
    # Check for new achievements
    new_achievements = check_achievements(username, total_labels, time_taken)
    
    return new_achievements

def check_achievements(username, total_labels, time_taken):
    """Check if user earned any new achievements"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT achievements, points FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    current_achievements = json.loads(row[0]) if row[0] else []
    current_points = row[1]
    
    new_achievements = []
    points_earned = 0
    
    # First label
    if total_labels == 1 and 'first_label' not in current_achievements:
        new_achievements.append('first_label')
        points_earned += ACHIEVEMENTS['first_label']['points']
    
    # Fifty
    if total_labels == 50 and 'fifty' not in current_achievements:
        new_achievements.append('fifty')
        points_earned += ACHIEVEMENTS['fifty']['points']
    
    # Century club
    if total_labels == 100 and 'century' not in current_achievements:
        new_achievements.append('century')
        points_earned += ACHIEVEMENTS['century']['points']
    
    # Half K
    if total_labels == 500 and 'half_k' not in current_achievements:
        new_achievements.append('half_k')
        points_earned += ACHIEVEMENTS['half_k']['points']
    
    # Speed demon (classified in under 30 seconds)
    if time_taken and time_taken < 30:
        c.execute('''SELECT COUNT(*) FROM classifications 
                     WHERE username = ? AND time_taken < 30 
                     AND datetime(timestamp) > datetime('now', '-5 minutes')''', (username,))
        fast_count = c.fetchone()[0]
        if fast_count >= 10 and 'speed_demon' not in current_achievements:
            new_achievements.append('speed_demon')
            points_earned += ACHIEVEMENTS['speed_demon']['points']
    
    # Consistency (30 in one session - within 2 hours)
    if total_labels >= 30:
        c.execute('''SELECT COUNT(*) FROM classifications 
                     WHERE username = ? 
                     AND datetime(timestamp) > datetime('now', '-2 hours')''', (username,))
        session_count = c.fetchone()[0]
        if session_count >= 30 and 'consistency' not in current_achievements:
            new_achievements.append('consistency')
            points_earned += ACHIEVEMENTS['consistency']['points']
    
    # Update achievements and points
    if new_achievements:
        all_achievements = current_achievements + new_achievements
        c.execute('''UPDATE users SET achievements = ?, points = points + ?
                     WHERE username = ?''', (json.dumps(all_achievements), points_earned, username))
        conn.commit()
    
    conn.close()
    
    return [{'id': ach, **ACHIEVEMENTS[ach]} for ach in new_achievements]

def get_leaderboard():
    """Get current leaderboard"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT username, total_labels, points, achievements
                 FROM users
                 ORDER BY points DESC, total_labels DESC
                 LIMIT 10''')
    
    leaderboard = []
    for row in c.fetchall():
        leaderboard.append({
            'username': row[0],
            'total_labels': row[1],
            'points': row[2],
            'achievements': json.loads(row[3]) if row[3] else []
        })
    
    conn.close()
    return leaderboard

def get_stats():
    """Get overall stats"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    total_images = len(get_all_images())
    
    c.execute('SELECT COUNT(DISTINCT image_name) FROM classifications')
    classified_count = c.fetchone()[0]
    
    c.execute('SELECT COUNT(DISTINCT username) FROM users')
    total_users = c.fetchone()[0]
    
    c.execute('SELECT label, COUNT(*) FROM classifications GROUP BY label')
    label_distribution = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute('SELECT COUNT(*) FROM active_sessions WHERE current_image IS NOT NULL')
    active_users = c.fetchone()[0]
    
    conn.close()
    
    return {
        'total_images': total_images,
        'classified': classified_count,
        'remaining': total_images - classified_count,
        'progress_percent': round((classified_count / total_images * 100) if total_images > 0 else 0, 1),
        'total_users': total_users,
        'active_users': active_users,
        'label_distribution': label_distribution
    }

@app.route('/')
def index():
    """Main classification interface"""
    return render_template('index.html')

@app.route('/api/next_image', methods=['POST'])
def next_image():
    """Get next image for user to classify"""
    data = request.json
    username = data.get('username', 'Anonymous')
    
    if not username or username.strip() == '':
        return jsonify({'error': 'Username required'}), 400
    
    image = get_next_image(username)
    
    if image:
        update_active_session(username, image)
        return jsonify({
            'image': image,
            'url': f'/images/{image}'
        })
    else:
        clear_active_session(username)
        return jsonify({'completed': True, 'message': 'All images classified! 🎉'})

@app.route('/api/classify', methods=['POST'])
def classify():
    """Save a classification"""
    data = request.json
    image_name = data.get('image')
    label = data.get('label')
    username = data.get('username', 'Anonymous')
    time_taken = data.get('time_taken')
    
    if not all([image_name, label, username]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    new_achievements = save_classification(image_name, label, username, time_taken)
    clear_active_session(username)
    
    return jsonify({
        'success': True,
        'new_achievements': new_achievements
    })

@app.route('/api/skip', methods=['POST'])
def skip_image():
    """Skip current image (mark as difficult/uncertain)"""
    data = request.json
    username = data.get('username', 'Anonymous')
    
    clear_active_session(username)
    return jsonify({'success': True})

@app.route('/api/leaderboard')
def leaderboard():
    """Get current leaderboard"""
    return jsonify(get_leaderboard())

@app.route('/api/stats')
def stats():
    """Get overall statistics"""
    return jsonify(get_stats())

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the images directory"""
    # Create the absolute path to the intended directory containing the file
    directory = os.path.abspath(os.path.join(IMAGES_DIR, os.path.dirname(filename)))
    file = os.path.basename(filename)
    return send_from_directory(directory, file)

@app.route('/api/export')
def export_data():
    """Export all classifications as JSON"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT image_name, label, username, timestamp 
                 FROM classifications 
                 ORDER BY timestamp''')
    
    results = []
    for row in c.fetchall():
        results.append({
            'image': row[0],
            'label': row[1],
            'username': row[2],
            'timestamp': row[3]
        })
    
    conn.close()
    
    return jsonify(results)

if __name__ == '__main__':
    init_db()
    
    # Get local IP for sharing with friends
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print("🚀 Road Condition Classifier Server Starting!")
    print("="*60)
    print(f"\n📁 Images directory: {IMAGES_DIR}")
    print(f"   (Put your road images here!)\n")
    print(f"🌐 Share this URL with your friends:")
    print(f"   http://{local_ip}:5000")
    print(f"\n💻 Local access:")
    print(f"   http://localhost:5000")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
