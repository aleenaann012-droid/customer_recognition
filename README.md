Smart Customer Recognition & Analytics System

An AI-powered retail analytics system that uses realtime face recognition and demographic analysis to identify repeat customers, distinguish staff members from visitors, and generate customer insights for smart retail environments.

Overview

This project is designed for retail stores and smart shop environments where customer behavior and demographics can help improve business decisions. The system uses computer vision and deep learning techniques to monitor customers through entry and exit cameras.

The application performs:

Face detection
Face recognition
Staff identification
Customer visit tracking
Age estimation
Customer analytics generation

The system automatically recognizes returning customers using facial embeddings while excluding registered staff members from analytics.

Features
Realtime face detection
Realtime face recognition
Staff and customer separation
Repeat customer identification
Automatic customer ID generation
Age estimation using AI
Visit count tracking
Customer demographic analytics
Entry and exit monitoring
Local database storage
Technologies Used
Programming Language
Python
Computer Vision & AI
OpenCV
InsightFace (ArcFace)
RetinaFace
ONNX Runtime
Database
SQLite
Libraries
NumPy
Pandas
System Workflow
Capture live video from cameras
Detect faces in realtime
Generate face embeddings
Compare embeddings with registered staff database
If matched:
Mark as staff
Exclude from customer analytics
If not matched:
Compare with existing customer database
Identify returning customer or create new customer profile
Predict customer age
Store analytics in database
Staff and Customer Logic
Registered Staff

Staff members are registered manually before system deployment. Their facial embeddings are stored separately and excluded from customer statistics.

Unregistered Customers

Customers are automatically assigned unique customer IDs. Their embeddings are stored and used for identifying repeat visits in future sessions.

Analytics Generated
Repeat customer count
Most common customer age group
Customer visit frequency
Daily customer statistics
Returning vs new customer analysis
