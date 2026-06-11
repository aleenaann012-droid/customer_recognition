import sqlite3
import os

# Connect to the SQLite database
DB_PATH = os.path.join("database", "customers.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def clean_duplicates():
    # Find all unique combinations of staff_id and date
    cursor.execute("SELECT DISTINCT staff_id, date FROM staff_attendance")
    records = cursor.fetchall()

    for staff_id, date in records:
        # Get all entries for this staff member on this date, ordered by entry_time
        cursor.execute('''
            SELECT attendance_id, entry_time, exit_time 
            FROM staff_attendance 
            WHERE staff_id = ? AND date = ? 
            ORDER BY entry_time ASC
        ''', (staff_id, date))
        
        entries = cursor.fetchall()
        
        # If there are duplicates
        if len(entries) > 1:
            first_entry_id = entries[0][0]
            
            # Find the latest exit time among all entries
            latest_exit_time = None
            for entry in entries:
                if entry[2] is not None:
                    if latest_exit_time is None or entry[2] > latest_exit_time:
                        latest_exit_time = entry[2]
            
            # Update the first entry with the latest exit time
            if latest_exit_time:
                cursor.execute('''
                    UPDATE staff_attendance 
                    SET exit_time = ? 
                    WHERE attendance_id = ?
                ''', (latest_exit_time, first_entry_id))
            
            # Delete all other extra entries
            ids_to_delete = [entry[0] for entry in entries[1:]]
            cursor.execute(f'''
                DELETE FROM staff_attendance 
                WHERE attendance_id IN ({','.join('?'*len(ids_to_delete))})
            ''', ids_to_delete)
            
            print(f"Cleaned {len(ids_to_delete)} duplicate entries for staff_id {staff_id} on {date}")
            
    conn.commit()
    conn.close()
    print("Database cleanup complete!")

if __name__ == "__main__":
    clean_duplicates()
