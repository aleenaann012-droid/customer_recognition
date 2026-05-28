import sqlite3

conn = sqlite3.connect("database/customers.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers(
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    age INTEGER,
    gender TEXT,
    
    visit_count INTEGER DEFAULT 1,
    
    entry_time TEXT,
    exit_time TEXT,
    
    visit_date TEXT,
    last_seen TEXT,
    
    embedding BLOB
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS staff(
    staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    name TEXT,
    role TEXT,
    
    embedding BLOB
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS staff_attendance(
    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    staff_id INTEGER,
    
    entry_time TEXT,
    exit_time TEXT,
    
    date TEXT,
    
    FOREIGN KEY(staff_id) REFERENCES staff(staff_id)
)
""")

conn.commit()
conn.close()
