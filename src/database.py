import sqlite3

conn = sqlite3.connect("database/customers.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers(
    customer_id INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    entry_time TEXT,
    exit_time TEXT,
    visit_count INTEGER,
    embedding BLOB
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS staff(
    staff_id INTEGER PRIMARY KEY,
    name TEXT,
    role TEXT,
    embedding BLOB
)
""")
conn.commit()
conn.close()
