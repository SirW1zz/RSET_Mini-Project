# Smart Attendance & Persistence Management System Version 2 (AAPT v2.0.0)

An autonomous, vision-based attendance ecosystem that uses existing CCTV infrastructure for continuous monitoring.

## System Components
1. **Vision Pipeline (Python)**
    - Uses OpenCV for video handling and InsightFace/FaceNet for facial embeddings.
    - Features 16 zone Zooming, Teacher Trigger for passive operation, and Bunk Detection.
    - Integrates DeepSORT/BoT-SORT for Multi-Object Tracking.
2. **Backend Config & Database**
    - PostgreSQL database powered by Supabase.
3. **Dashboards**
    - Streamlit web frontend covering Admin, Teacher, and Student/Parent use cases.
4. **PDF Generator**
    - Creates a report that shows the status of that session


## Folder Structure
- `database/`: Local database, facial features and enrollment details stores as JSON files.
- `vision/`: Contains the primary Python Vision script (`main.py`) running real-time tracking, along with `requirements.txt`.
- `dashboard/`: Contains the Streamlit dashboard app (`app.py`)

## Getting Started
1. Set up your `.env` file using the provided template `.env` in the root directory.
2. Execute the `database/01_schema.sql` in your Supabase SQL editor.
3. Install Python requirements in both `vision/` and `dashboard/`:
   ```bash
   pip install -r vision/requirements.txt
   pip install -r dashboard/requirements.txt
   ```
4. Run the Vision script: `python vision/main.py`.
6. Run the Streamlit Dashboard: `streamlit run dashboard/app.py`.
