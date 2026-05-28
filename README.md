
Smart Customer Recognition & Analytics System

AI-powered retail customer analytics system using Face Recognition, Age Detection, and Realtime Monitoring.

📌 Project Overview

This project is a realtime computer vision system designed for retail shops and stores.
The system uses AI-powered face detection and recognition to:

Detect customers entering/exiting the shop
Recognize repeat customers
Differentiate staff and customers
Estimate customer age groups
Generate customer analytics for business insights

The project uses:

Python
OpenCV
InsightFace (ArcFace)
RetinaFace
SQLite
🚀 Features
✅ Face Detection

Detects human faces from live camera feed.

✅ Face Recognition

Recognizes returning customers using facial embeddings.

✅ Staff Recognition

Registered staff members are identified separately and excluded from customer analytics.

✅ Customer Tracking

Unregistered customers are automatically assigned unique IDs and tracked during future visits.

✅ Age Estimation

Predicts customer age using InsightFace age-gender model.

✅ Customer Analytics

Generates insights such as:

Most frequent age group
Repeat customer count
Total customer visits
Peak customer timings
🧠 System Workflow
ENTRY/EXIT CAMERA
        ↓
Face Detection
        ↓
Embedding Generation
        ↓
Check Staff Database
        ↓
IF STAFF
    → Ignore Customer Analytics

ELSE
    ↓
Check Customer Database
    ↓
IF EXISTING CUSTOMER
    → Update Visit Count

ELSE
    → Create New Customer ID

        ↓
Age Prediction
        ↓
Store Data in Database
        ↓
Generate Analytics
🏗️ Technologies Used
Category	Technology
Programming Language	Python
Face Detection	RetinaFace
Face Recognition	InsightFace ArcFace
Computer Vision	OpenCV
Database	SQLite
Numerical Operations	NumPy
Realtime Inference	ONNX Runtime
📂 Project Structure
FaceRecognitionProject/
│
├── database/
│   └── customers.db
│
├── datasets/
│
├── logs/
│
├── models/
│
├── src/
│   ├── main.py
│   ├── detect_face.py
│   ├── recognize_face.py
│   ├── register_staff.py
│   ├── customer_match.py
│   ├── analytics.py
│   └── database.py
│
├── requirements.txt
│
└── README.md
🗄️ Database Design
Staff Table

Stores registered shop employees.

CREATE TABLE IF NOT EXISTS staff(
    staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    role TEXT,
    embedding BLOB
);
Customers Table

Stores customer information and visit history.

CREATE TABLE IF NOT EXISTS customers(
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    gender TEXT,
    visit_count INTEGER,
    embedding BLOB,
    last_seen TEXT
);
🔍 How Face Recognition Works

The system does not store images directly for recognition.

Instead:

Face is converted into embeddings (numerical vectors)
Embeddings are compared using similarity metrics
If similarity exceeds threshold:
Same person detected
Otherwise:
New customer created
👨‍💼 Staff vs Customer Logic
Staff Members
Registered manually
Stored in staff database
Excluded from customer analytics
Customers
Unregistered by default
Automatically assigned customer IDs
Visit history tracked
📊 Analytics Generated
Repeat customers
Most common age group
Customer visit count
Daily customer statistics
Peak shopping hours
