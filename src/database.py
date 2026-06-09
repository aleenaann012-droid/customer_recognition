import sqlite3
import numpy as np

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
cursor.execute("""
CREATE TABLE IF NOT EXISTS visitor_stats(
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    age_group TEXT,
    gender TEXT,

    visit_time TEXT
)
""")

conn.commit()
conn.close()

DB_PATH = "database/customers.db"

#Load Staff
def load_staff():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT staff_id, name, role, embedding
        FROM staff
    """)

    staff_details = []
    staff_embeddings = []

    for staff_id, name, role, emb_blob in cursor.fetchall():

        staff_details.append({
            "staff_id": staff_id,
            "name": name,
            "role": role
        })

        staff_embeddings.append(
            np.frombuffer(
                emb_blob,
                dtype=np.float32
            )
        )

    conn.close()

    return staff_details, staff_embeddings

    #Load staff attendance
def insert_staff_attendance(staff_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO staff_attendance
            (staff_id, entry_time, date)
            VALUES
            (?, datetime('now'), date('now'))
        """, (staff_id,))

        conn.commit()
        conn.close()

    #Insert visitor
def insert_visitor(age_group, gender):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO visitor_stats
            (age_group, gender)
            VALUES (?, ?)
        """, (age_group, gender))

        conn.commit()
        conn.close()

def load_customers():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT customer_id, embedding
        FROM customers
    """)

    customer_details = []
    customer_embeddings = []

    for cid, emb_blob in cursor.fetchall():
        if emb_blob:
            customer_details.append({"customer_id": cid})
            customer_embeddings.append(np.frombuffer(emb_blob, dtype=np.float32))

    conn.close()
    return customer_details, customer_embeddings

def insert_customer(age, gender, embedding_array):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO customers
        (age, gender, visit_count, entry_time, visit_date, embedding)
        VALUES (?, ?, 1, datetime('now'), date('now'), ?)
    """, (age, gender, embedding_array.tobytes()))
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def update_customer(customer_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE customers
        SET visit_count = visit_count + 1, last_seen = datetime('now')
        WHERE customer_id = ?
    """, (customer_id,))
    
    conn.commit()
    conn.close()
