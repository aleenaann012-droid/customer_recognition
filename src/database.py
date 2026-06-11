import sqlite3
import numpy as np
import logging
import os

# Configure logging for database operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "customers.db")
logging.info(f"Database initialization: Using DB_PATH -> {os.path.abspath(DB_PATH)}")

try:
    conn = sqlite3.connect(DB_PATH)
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
    logging.info("Database tables verified/created successfully.")
except Exception as e:
    logging.error(f"Failed to initialize database: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()

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

        try:
            cursor.execute("""
                INSERT INTO staff_attendance
                (staff_id, entry_time, date)
                VALUES
                (?, datetime('now'), date('now'))
            """, (staff_id,))

            conn.commit()
            logging.info(f"Successfully inserted staff attendance for staff_id: {staff_id}")
        except Exception as e:
            logging.error(f"Error executing staff attendance INSERT: {e}")
            conn.rollback()
        finally:
            conn.close()

    #Insert visitor
def insert_visitor(age_group, gender):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO visitor_stats
                (age_group, gender)
                VALUES (?, ?)
            """, (age_group, gender))

            conn.commit()
            logging.info(f"Successfully inserted visitor stats: age_group={age_group}, gender={gender}")
        except Exception as e:
            logging.error(f"Error executing visitor stats INSERT: {e}")
            conn.rollback()
        finally:
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
    
    try:
        cursor.execute("""
            INSERT INTO customers
            (age, gender, visit_count, entry_time, visit_date, embedding)
            VALUES (?, ?, 1, datetime('now'), date('now'), ?)
        """, (age, gender, embedding_array.tobytes()))
        
        new_id = cursor.lastrowid
        conn.commit()
        logging.info(f"Successfully inserted new customer: new_id={new_id}, age={age}, gender={gender}")
    except Exception as e:
        logging.error(f"Error executing customer INSERT: {e}")
        conn.rollback()
        new_id = None
    finally:
        conn.close()
    return new_id

def update_customer(customer_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE customers
            SET visit_count = visit_count + 1, last_seen = datetime('now')
            WHERE customer_id = ?
        """, (customer_id,))
        
        conn.commit()
        logging.info(f"Successfully updated returning customer: customer_id={customer_id}")
    except Exception as e:
        logging.error(f"Error executing customer UPDATE: {e}")
        conn.rollback()
    finally:
        conn.close()

def update_staff_exit(staff_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE staff_attendance
            SET exit_time = datetime('now')
            WHERE attendance_id = (
                SELECT attendance_id FROM staff_attendance 
                WHERE staff_id = ? AND date = date('now') 
                ORDER BY attendance_id DESC LIMIT 1
            )
        """, (staff_id,))
        
        conn.commit()
        logging.info(f"Successfully updated exit time for staff_id: {staff_id}")
    except Exception as e:
        logging.error(f"Error executing staff exit UPDATE: {e}")
        conn.rollback()
    finally:
        conn.close()

def update_customer_exit(customer_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE customers
            SET exit_time = datetime('now')
            WHERE customer_id = ?
        """, (customer_id,))
        conn.commit()
        logging.info(f"Successfully updated exit time for customer_id: {customer_id}")
    except Exception as e:
        logging.error(f"Error executing customer exit UPDATE: {e}")
        conn.rollback()
    finally:
        conn.close()
