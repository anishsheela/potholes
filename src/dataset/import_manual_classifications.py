import os
import sqlite3

def import_manual_classifications():
    db_path = 'road_classifier/classifications.db'
    base_dir = 'dataset/classification_bak/training'
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found at {base_dir}")
        return
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check what we've already imported to avoid duplicates
    c.execute("SELECT image_name FROM classifications WHERE username='manual_import_1'")
    existing = set(row[0] for row in c.fetchall())
    
    classes = ['Excellent', 'Good', 'Fair', 'Poor']
    added_count = 0
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # We use the exact relative path from the repo root
                rel_path = os.path.join(cls_dir, f)
                
                if rel_path in existing:
                    continue
                
                # Insert two votes with the identical label so it instantly reaches a '2-vote consensus'
                # when export_consensus_dataset.py runs.
                c.execute("INSERT INTO classifications (image_name, label, username, time_taken) VALUES (?, ?, ?, ?)",
                          (rel_path, cls, 'manual_import_1', 1.0))
                c.execute("INSERT INTO classifications (image_name, label, username, time_taken) VALUES (?, ?, ?, ?)",
                          (rel_path, cls, 'manual_import_2', 1.0))
                
                added_count += 1

    conn.commit()
    conn.close()
    
    print(f"✅ Successfully imported {added_count} manually classified images into {db_path}!")
    print("These have been given 2 automated 'votes' each, so they will immediately register as consensus.")

if __name__ == '__main__':
    import_manual_classifications()
