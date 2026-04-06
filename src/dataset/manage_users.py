import sqlite3
import argparse
import json
import sys

DB_PATH = 'road_classifier/classifications.db'

def list_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, total_labels, points FROM users ORDER BY total_labels DESC")
    users = c.fetchall()
    
    print("\n--- Current Annotators Leaderboard ---")
    print(f"{'Username':<25} | {'Labels':<8} | {'Points'}")
    print("-" * 50)
    for u in users:
        print(f"{u[0]:<25} | {u[1]:<8} | {u[2]}")
    print("-" * 50)
    conn.close()

def merge_users(source_user, target_user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Verify both users exist
    c.execute("SELECT * FROM users WHERE username=?", (source_user,))
    src = c.fetchone()
    if not src:
         print(f"❌ Error: Source user '{source_user}' does not exist.")
         conn.close()
         return

    c.execute("SELECT * FROM users WHERE username=?", (target_user,))
    tgt = c.fetchone()
    if not tgt:
         print(f"❌ Error: Target user '{target_user}' does not exist.")
         conn.close()
         return

    print(f"Initiating merge: '{source_user}' -> '{target_user}'")

    try:
        # 1. Reassign classifications
        c.execute("UPDATE classifications SET username=? WHERE username=?", (target_user, source_user))
        moved_classifications = c.rowcount

        # 2. Combine user statistics
        # src/tgt tuple indices map directly to schema columns:
        # 0: username, 1: total_labels, 2: points, 3: achievements, 4: session_count, 5: last_active, 6: current_streak, 7: max_streak
        
        new_total_labels = (tgt[1] or 0) + (src[1] or 0)
        new_points = (tgt[2] or 0) + (src[2] or 0)
        new_session_count = (tgt[4] or 0) + (src[4] or 0)
        
        # Merge JSON achievements, deduplicating
        src_achievements_json = src[3] or '[]'
        tgt_achievements_json = tgt[3] or '[]'
        try:
            src_achievements = json.loads(src_achievements_json)
            tgt_achievements = json.loads(tgt_achievements_json)
        except json.JSONDecodeError:
            src_achievements = []
            tgt_achievements = []
            
        merged_achievements = list(set(src_achievements + tgt_achievements))
        merged_achievements_json = json.dumps(merged_achievements)

        # Adopt maximum streak
        new_max_streak = max(tgt[7] or 0, src[7] or 0)
        
        c.execute("""
            UPDATE users 
            SET total_labels=?, points=?, achievements=?, session_count=?, max_streak=? 
            WHERE username=?
        """, (new_total_labels, new_points, merged_achievements_json, new_session_count, new_max_streak, target_user))

        # 3. Purge the source user record
        c.execute("DELETE FROM users WHERE username=?", (source_user,))
        c.execute("DELETE FROM active_sessions WHERE username=?", (source_user,))

        conn.commit()
        print(f"✅ Successfully merged '{source_user}' into '{target_user}'.")
        print(f"   -> Reassigned {moved_classifications} distinct image classifications.")
        print(f"   -> New combined label count for {target_user}: {new_total_labels}")

    except Exception as e:
        conn.rollback()
        print(f"❌ An error occurred during merge: {e}. Changes rolled back safely.")
    finally:
        conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Manage and Merge Database Annotator Accounts")
    parser.add_argument('--list', action='store_true', help="List all users in the database by label count")
    parser.add_argument('--merge', nargs=2, metavar=('SOURCE', 'TARGET'), help="Merge first user into the second user")
    
    args = parser.parse_args()

    if args.list:
        list_users()
    elif args.merge:
        # Allow case-sensitive input exactly as typed
        merge_users(args.merge[0], args.merge[1])
    else:
        parser.print_help()
