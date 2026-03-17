import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
from dotenv import load_dotenv
import cv2
from PIL import Image, ImageOps
from datetime import datetime
import tempfile
from fpdf import FPDF
import face_recognition as fr

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(page_title="AAPT Dashboard", layout="wide", page_icon="🏫")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DB_DIR = os.path.join(BASE_DIR, "local_db")
FACES_DIR = os.path.join(LOCAL_DB_DIR, "faces")
STUDENT_FACES_DIR = os.path.join(FACES_DIR, "students")
TEACHER_FACES_DIR = os.path.join(FACES_DIR, "teachers")
STUDENTS_JSON = os.path.join(LOCAL_DB_DIR, "students.json")
TEACHERS_JSON = os.path.join(LOCAL_DB_DIR, "teachers.json")
ADMINS_JSON = os.path.join(LOCAL_DB_DIR, "admins.json")
LOGS_JSON = os.path.join(LOCAL_DB_DIR, "logs.json")

def ensure_db():
    os.makedirs(STUDENT_FACES_DIR, exist_ok=True)
    os.makedirs(TEACHER_FACES_DIR, exist_ok=True)
    if not os.path.exists(STUDENTS_JSON):
        with open(STUDENTS_JSON, 'w') as f: json.dump({}, f)
    if not os.path.exists(TEACHERS_JSON):
        with open(TEACHERS_JSON, 'w') as f: json.dump({}, f)
    if not os.path.exists(ADMINS_JSON):
        with open(ADMINS_JSON, 'w') as f: json.dump({}, f)
    if not os.path.exists(LOGS_JSON):
        with open(LOGS_JSON, 'w') as f: json.dump({"scan_logs": [], "attendance_logs": []}, f)

ensure_db()

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Initialize Session State from DB
if 'db_loaded' not in st.session_state:
    st.session_state.students_db = load_data(STUDENTS_JSON)
    st.session_state.teachers_db = load_data(TEACHERS_JSON)
    st.session_state.admins_db = load_data(ADMINS_JSON)
    logs = load_data(LOGS_JSON)
    st.session_state.scan_logs = logs.get("scan_logs", [])
    st.session_state.attendance_logs = logs.get("attendance_logs", [])
    
    # Auth states
    st.session_state.admin_logged_in = False
    st.session_state.teacher_mfa_verified = False # Secondary factor for teachers
    
    st.session_state.db_loaded = True

def sync_logs():
    save_data(LOGS_JSON, {"scan_logs": st.session_state.scan_logs, "attendance_logs": st.session_state.attendance_logs})

def save_uploaded_photo(uploaded_file, dest_dir, name):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    img_rgb = img.convert('RGB')
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(dest_dir, f"{name}_{ts}.jpg")
    img_rgb.save(file_path, format="JPEG")
    return file_path

# Custom App Styling
st.markdown("""
<style>
    .metric-container {
        border-radius: 10px;
        padding: 20px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }

    /* Vertical and Horizontal Centering for Sidebar Content */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }

    /* Force the minimize button to stay at the top right */
    [data-testid="stSidebar"] button[kind="header"] {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }

    /* Target the Radio navigation to look like wider buttons */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 15px;
        width: 100%;
        padding: 40px 10px;
    }

    /* Hide the radio circles and prevent text wrapping */
    [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        margin-bottom: 0;
        white-space: nowrap;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Style the label as a premium button */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
        padding: 18px 25px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        justify-content: center;
        width: 100% !important;
        min-width: 200px;
    }

    /* Hover effects */
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(31, 119, 180, 0.2);
        border-color: #1f77b4;
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        color: white;
    }

    /* Active state */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: 600;
        border-color: #1f77b4;
        box-shadow: 0 4px 20px rgba(31, 119, 180, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Helper: load/cache face encodings from database
def get_known_encodings(db_type="students"):
    state_key = f"known_encodings_{db_type}"
    names_key = f"known_names_{db_type}"
    
    # Return from cache if available
    if state_key in st.session_state and names_key in st.session_state:
        return st.session_state[state_key], st.session_state[names_key]
    
    known_encodings = []
    known_names = []
    
    db = st.session_state.students_db if db_type == "students" else st.session_state.teachers_db
    
    for name, data in db.items():
        paths = data.get("photo_paths", [data.get("photo_path")])
        for p in paths:
            if p and os.path.exists(p):
                try:
                    img = fr.load_image_file(p)
                    encs = fr.face_encodings(img)
                    if encs:
                        known_encodings.append(encs[0])
                        known_names.append(name)
                except Exception:
                    pass
    
    # Store in cache
    st.session_state[state_key] = known_encodings
    st.session_state[names_key] = known_names
    return known_encodings, known_names

# Helper function to process the image with a 4x4 Precision Grid
def process_precision_grid(image_file, is_bgr=False, tolerance=0.50, upsample=1, model="hog"):
    if is_bgr:
        img = image_file.copy()
    else:
        pil_img = Image.open(image_file)
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w, _ = img.shape
    
    # 1. CLEAN SCAN: Detect faces BEFORE drawing any lines (Fixes axis-detection issues)
    img_clean_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect all face locations and compute their 128-d encodings
    # CNN model is much more accurate but slower. HOG is faster for real-time.
    face_locations = fr.face_locations(img_clean_rgb, number_of_times_to_upsample=upsample, model=model)
    face_encodings = fr.face_encodings(img_clean_rgb, face_locations)
    
    # 2. DRAW 4x4 GRID (16 Zones) for display
    # Horizontal lines
    for i in range(1, 4):
        y = int(h * i / 4)
        cv2.line(img, (0, y), (w, y), (255, 0, 0), 2)
    # Vertical lines
    for i in range(1, 4):
        x = int(w * i / 4)
        cv2.line(img, (x, 0), (x, h), (255, 0, 0), 2)
    
    # Label zones G1 to G16
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col + 1
            y_pos = int((row + 0.1) * (h / 4)) + 30
            x_pos = int(col * (w / 4)) + 20
            cv2.putText(img, f"G{idx}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Load cached enrolled student encodings
    known_encodings, known_names = get_known_encodings("students")
    
    detected_students = set()
    total_faces = len(face_locations)
    
    # Global Best-Match logic
    face_matches = []
    for face_idx, face_enc in enumerate(face_encodings):
        if known_encodings:
            distances = fr.face_distance(known_encodings, face_enc)
            for student_idx, dist in enumerate(distances):
                if dist <= tolerance:
                    face_matches.append((dist, face_idx, known_names[student_idx]))
    
    face_matches.sort(key=lambda x: x[0])
    
    assignments = {}
    used_student_names = set()
    used_face_indices = set()
    
    for dist, face_idx, name in face_matches:
        if face_idx not in used_face_indices and name not in used_student_names:
            conf = max(0, int((1.0 - (dist / 0.7)) * 100))
            assignments[face_idx] = (name, conf, dist)
            used_student_names.add(name)
            used_face_indices.add(face_idx)
            detected_students.add(name)
    
    # Render identities on the grid image
    for face_idx, (top, right, bottom, left) in enumerate(face_locations):
        if face_idx in assignments:
            name, conf, dist = assignments[face_idx]
            # Bolder safety thresholds
            # dist <= 0.40 is very strong (Green)
            # 0.40 < dist <= tolerance is weak (Orange)
            color = (0, 165, 255) if dist > 0.40 else (0, 255, 0)
            thickness = 4
        else:
            name, conf = "Unknown", 0
            color = (0, 0, 255)
            thickness = 2
            
        cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
        label = f"{name} ({conf}%)" if name != "Unknown" else "Unknown"
        cv2.putText(img, label, (left + 2, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    img_rgb_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb_final), total_faces, list(detected_students)

def generate_pdf_report(session_start_time, subject, teacher):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, text="AAPT Session Attendance Report", ln=True, align='C')
    pdf.ln(10)
    
    # Metadata
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text=f"Subject: {subject}", ln=True)
    pdf.cell(0, 10, text=f"Instructor: {teacher}", ln=True)
    pdf.cell(0, 10, text=f"Session Start: {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Filter logs for this session
    session_logs = [log for log in st.session_state.scan_logs if datetime.strptime(log['Timestamp'], "%Y-%m-%d %H:%M:%S") >= session_start_time]
    
    # Determine final status for everyone seen in this session (overwritten by their MOST RECENT occurrence)
    final_status = {}
    for log in session_logs:
        present = log.get('Students_Identified', '').split(', ') if log.get('Students_Identified') and log.get('Students_Identified') != "None" else []
        bunked = log.get('Students_Bunked', '').split(', ') if log.get('Students_Bunked') else []
        
        for p in present:
            if p: final_status[p.strip()] = "Present"
        for b in bunked:
            if b: final_status[b.strip()] = "Bunked"
            
    present_list = sorted([s for s, status in final_status.items() if status == "Present"])
    bunked_list = sorted([s for s, status in final_status.items() if status == "Bunked"])
    
    # Final Summary
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, text="Final Attendance Summary", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, text=f"Total Present at Session End: {len(present_list)}", ln=True)
    pdf.multi_cell(0, 8, text=f"Present Students: {', '.join(present_list) if present_list else 'None'}")
    pdf.ln(5)
    pdf.cell(0, 8, text=f"Total Bunked/Missing at Session End: {len(bunked_list)}", ln=True)
    pdf.multi_cell(0, 8, text=f"Bunked Students: {', '.join(bunked_list) if bunked_list else 'None'}")
    pdf.ln(10)
    
    # Breakdown History
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, text="Interval Scans Breakdown", ln=True)
    pdf.set_font("Helvetica", size=10)
    for log in session_logs:
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 6, text=f"[{log['Timestamp']}] {log['Status']}", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 6, text=f"  - Detected: {log['Faces_Detected']} faces", ln=True)
        if log.get('Students_Identified') and log.get('Students_Identified') != "None":
            pdf.multi_cell(0, 6, text=f"  - Present: {log['Students_Identified']}")
        if log.get('Students_Bunked'):
            pdf.multi_cell(0, 6, text=f"  - Bunked: {log['Students_Bunked']}")
        pdf.ln(2)
        
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    pdf.output(tmp_path)
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
    os.remove(tmp_path)
    return pdf_bytes

# ==========================================
# UI ROUTING AND LAYOUT
# ==========================================
# Empty labels as requested for a cleaner look
# Centered Sidebar Role Switcher
role = st.sidebar.radio("Navigation", ["Admin", "Teacher", "Student/Parent"], label_visibility="collapsed")

# ==========================================
# ADMIN DASHBOARD
# ==========================================
if role == "Admin":
    st.title("Admin Security 🔒")
    
    if not st.session_state.admins_db:
        st.info("👋 Welcome! No administrator found. Please register the primary admin account.")
        new_adm_user = st.text_input("New Admin Username")
        new_adm_pass = st.text_input("New Admin Password", type="password")
        if st.button("Register Primary Admin"):
            if new_adm_user and new_adm_pass:
                st.session_state.admins_db[new_adm_user] = new_adm_pass
                save_data(ADMINS_JSON, st.session_state.admins_db)
                st.success("Admin registered! Please log in.")
                st.rerun()
            else:
                st.warning("Please provide both username and password.")
    
    elif not st.session_state.admin_logged_in:
        adm_user = st.text_input("Admin Username")
        adm_pass = st.text_input("Password", type="password")
        if st.button("Login as Admin"):
            if adm_user in st.session_state.admins_db and st.session_state.admins_db[adm_user] == adm_pass:
                st.session_state.admin_logged_in = True
                st.success("Access Granted.")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    
    else:
        st.sidebar.success(f"Logged in as Admin")
        if st.sidebar.button("Logout Admin"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.title("Admin Dashboard ⚙️")
        st.markdown("Manage student/teacher enrollments and view system logs.")
        
        tab_enroll, tab_db, tab_logs = st.tabs(["Enrollment Forms", "Database Management", "System Logs"])
    
        with tab_enroll:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Student Enrollment")
                student_name = st.text_input("Student Name")
                
                stu_cam_tab, stu_up_tab = st.tabs(["Live Camera", "Upload Photo"])
                student_photo = None
                
                with stu_cam_tab:
                    stu_cam = st.camera_input("Take Student Photo", key="student_cam")
                    if stu_cam: student_photo = stu_cam
                        
                with stu_up_tab:
                    stu_up = st.file_uploader("Upload Student Photo (Clear Face)", type=['png', 'jpg', 'jpeg'], key="student_pic")
                    if stu_up: student_photo = stu_up
                    
                if st.button("Register Student / Add Photo"):
                    if student_name and student_photo:
                        # Validate face detection before saving
                        pil_img = Image.open(student_photo)
                        img_np = np.array(pil_img.convert("RGB"))
                        face_locs = fr.face_locations(img_np)
                        
                        if not face_locs:
                            st.error("🚨 Face not detected in this photo! The AI won't be able to recognize this student. Please take a clearer photo.")
                        else:
                            photo_path = save_uploaded_photo(student_photo, STUDENT_FACES_DIR, student_name)
                            if student_name in st.session_state.students_db:
                                if "photo_paths" not in st.session_state.students_db[student_name]:
                                    old = st.session_state.students_db[student_name].get("photo_path")
                                    st.session_state.students_db[student_name]["photo_paths"] = [old] if old else []
                                st.session_state.students_db[student_name]["photo_paths"].append(photo_path)
                                save_data(STUDENTS_JSON, st.session_state.students_db)
                                if "known_encodings_students" in st.session_state: del st.session_state["known_encodings_students"]
                                st.success(f"Added additional photo for existing Student: {student_name}")
                            else:
                                st.session_state.students_db[student_name] = {"photo_path": photo_path, "photo_paths": [photo_path]}
                                save_data(STUDENTS_JSON, st.session_state.students_db)
                                if "known_encodings_students" in st.session_state: del st.session_state["known_encodings_students"]
                                st.success(f"Registered New Student: {student_name}")
                    else:
                        st.warning("Please provide both name and photo.")

            with col2:
                st.subheader("Teacher Enrollment")
                teacher_name = st.text_input("Teacher Name")
                teacher_subject = st.text_input("Assigned Subject")
                teacher_pass = st.text_input("Set Security Password", type="password", help="Teacher will use this + Face to login.")
                
                tea_cam_tab, tea_up_tab = st.tabs(["Live Camera", "Upload Photo"])
                teacher_photo = None
                
                with tea_cam_tab:
                    tea_cam = st.camera_input("Take Teacher Photo", key="teacher_cam")
                    if tea_cam: teacher_photo = tea_cam
                        
                with tea_up_tab:
                    tea_up = st.file_uploader("Upload Teacher Photo (Clear Face)", type=['png', 'jpg', 'jpeg'], key="teacher_pic")
                    if tea_up: teacher_photo = tea_up
                    
                if st.button("Register Teacher / Add Photo"):
                    if teacher_name and teacher_subject and teacher_photo and teacher_pass:
                        # Validate face detection before saving
                        pil_img = Image.open(teacher_photo)
                        img_np = np.array(pil_img.convert("RGB"))
                        face_locs = fr.face_locations(img_np)
                        
                        if not face_locs:
                            st.error("🚨 Face not detected in this photo! The AI won't be able to recognize you. Please take a clearer photo.")
                        else:
                            photo_path = save_uploaded_photo(teacher_photo, TEACHER_FACES_DIR, teacher_name)
                            if teacher_name in st.session_state.teachers_db:
                                if "photo_paths" not in st.session_state.teachers_db[teacher_name]:
                                    old = st.session_state.teachers_db[teacher_name].get("photo_path")
                                    st.session_state.teachers_db[teacher_name]["photo_paths"] = [old] if old else []
                                st.session_state.teachers_db[teacher_name]["photo_paths"].append(photo_path)
                                save_data(TEACHERS_JSON, st.session_state.teachers_db)
                                if "known_encodings_teachers" in st.session_state: del st.session_state["known_encodings_teachers"]
                                st.success(f"Added additional photo for existing Teacher: {teacher_name}")
                            else:
                                st.session_state.teachers_db[teacher_name] = {
                                    "subject": teacher_subject,
                                    "password": teacher_pass,
                                    "photo_path": photo_path,
                                    "photo_paths": [photo_path]
                                }
                                save_data(TEACHERS_JSON, st.session_state.teachers_db)
                                if "known_encodings_teachers" in st.session_state: del st.session_state["known_encodings_teachers"]
                                st.success(f"Registered New Teacher: {teacher_name} for subject {teacher_subject}")
                    else:
                        st.warning("Please provide name, subject, password, and photo.")
                        
        with tab_db:
            st.subheader("Database Management")
            st.write("Edit names, subjects, or completely delete enrolled records.")
            
            db_col1, db_col2 = st.columns(2)
            
            with db_col1:
                st.markdown("### Enrolled Students")
                if st.session_state.students_db:
                    student_df = pd.DataFrame([{"Name": name} for name in st.session_state.students_db.keys()])
                    student_df.index = range(1, len(student_df) + 1)
                    st.dataframe(student_df, use_container_width=True)
                    
                    edit_student_name = st.selectbox("Select Student to Modify", list(st.session_state.students_db.keys()), key="edit_student_select")
                    
                    if st.button("🧪 AI Health Check: Is this student recognized?", key="check_stu_fr"):
                        known_enc, _ = get_known_encodings("students")
                        sdata = st.session_state.students_db[edit_student_name]
                        photo_to_test = sdata.get("photo_paths", [sdata.get("photo_path")])[0]
                        if photo_to_test and os.path.exists(photo_to_test):
                            test_img = fr.load_image_file(photo_to_test)
                            test_encs = fr.face_encodings(test_img)
                            if test_encs:
                                matches = fr.compare_faces(known_enc, test_encs[0], tolerance=0.45)
                                if True in matches:
                                    st.success(f"✅ AI sees {edit_student_name} clearly in the database.")
                                else:
                                    st.warning(f"⚠️ AI is struggling with this photo. Please add a clearer one.")
                            else:
                                st.error("🚨 Photo is too dark/blurry for the AI to read.")
                        else:
                            st.error("No photo found.")

                    with st.expander("📸 Add Extra Photos (Improve Recognition Angle)"):
                        extra_stu_photo = st.file_uploader(f"Upload another angle for {edit_student_name}", type=['png', 'jpg', 'jpeg'], key="extra_stu_pic")
                        if st.button("Append Photo to Student Model", key="btn_add_stu_pic"):
                            if extra_stu_photo:
                                photo_path = save_uploaded_photo(extra_stu_photo, STUDENT_FACES_DIR, edit_student_name)
                                if "photo_paths" not in st.session_state.students_db[edit_student_name]:
                                    old = st.session_state.students_db[edit_student_name].get("photo_path")
                                    st.session_state.students_db[edit_student_name]["photo_paths"] = [old] if old else []
                                st.session_state.students_db[edit_student_name]["photo_paths"].append(photo_path)
                                save_data(STUDENTS_JSON, st.session_state.students_db)
                                if "known_encodings_students" in st.session_state: del st.session_state["known_encodings_students"]
                                st.success(f"Success! Model will now recognize {edit_student_name} from this new angle.")
                            else:
                                st.warning("Please upload a photo.")
                                
                    new_student_name = st.text_input("New Name (Optional, to rename)", key="new_student_name")
                    
                    st_colA, st_colB = st.columns(2)
                    with st_colA:
                        if st.button("Rename Student"):
                            if new_student_name and new_student_name != edit_student_name and new_student_name not in st.session_state.students_db:
                                st.session_state.students_db[new_student_name] = st.session_state.students_db.pop(edit_student_name)
                                save_data(STUDENTS_JSON, st.session_state.students_db)
                                if "known_encodings_students" in st.session_state: del st.session_state["known_encodings_students"]
                                st.success("Student renamed successfully!")
                                st.rerun()
                    with st_colB:
                        if st.button("Delete Student", type="primary"):
                            sdata = st.session_state.students_db.pop(edit_student_name)
                            for p in sdata.get("photo_paths", [sdata.get("photo_path")]):
                                if p and os.path.exists(p): os.remove(p)
                            save_data(STUDENTS_JSON, st.session_state.students_db)
                            if "known_encodings_students" in st.session_state: del st.session_state["known_encodings_students"]
                            st.success(f"Deleted Student: {edit_student_name}")
                            st.rerun()
                else:
                    st.info("No students enrolled.")

            with db_col2:
                st.markdown("### Enrolled Teachers")
                if st.session_state.teachers_db:
                    teacher_df = pd.DataFrame([{"Name": name, "Subject": data["subject"]} for name, data in st.session_state.teachers_db.items()])
                    teacher_df.index = range(1, len(teacher_df) + 1)
                    st.dataframe(teacher_df, use_container_width=True)
                    
                    edit_teacher_name = st.selectbox("Select Teacher to Modify", list(st.session_state.teachers_db.keys()), key="edit_teacher_select")
                    
                    with st.expander("📸 Add Extra Photos (Improve Recognition Angle)"):
                        extra_tea_photo = st.file_uploader(f"Upload another angle for {edit_teacher_name}", type=['png', 'jpg', 'jpeg'], key="extra_tea_pic")
                        if st.button("Append Photo to Teacher Model", key="btn_add_tea_pic"):
                            if extra_tea_photo:
                                photo_path = save_uploaded_photo(extra_tea_photo, TEACHER_FACES_DIR, edit_teacher_name)
                                if "photo_paths" not in st.session_state.teachers_db[edit_teacher_name]:
                                    old = st.session_state.teachers_db[edit_teacher_name].get("photo_path")
                                    st.session_state.teachers_db[edit_teacher_name]["photo_paths"] = [old] if old else []
                                st.session_state.teachers_db[edit_teacher_name]["photo_paths"].append(photo_path)
                                save_data(TEACHERS_JSON, st.session_state.teachers_db)
                                if "known_encodings_teachers" in st.session_state: del st.session_state["known_encodings_teachers"]
                                st.success(f"Success! Model will now recognize {edit_teacher_name} from this new angle.")
                            else:
                                st.warning("Please upload a photo.")
                                
                    new_teacher_name = st.text_input("New Name (Optional, to rename)", key="new_teacher_name")
                    new_teacher_subject = st.text_input("New Subject (Optional, to edit)", key="new_teacher_subject")
                    new_teacher_pass = st.text_input("New Password (Optional, to edit)", type="password", key="new_teacher_pass")
                    
                    t_colA, t_colB = st.columns(2)
                    with t_colA:
                        if st.button("Update Teacher"):
                            updated = False
                            if new_teacher_subject:
                                st.session_state.teachers_db[edit_teacher_name]["subject"] = new_teacher_subject
                                updated = True
                            if new_teacher_pass:
                                st.session_state.teachers_db[edit_teacher_name]["password"] = new_teacher_pass
                                updated = True
                            
                            if new_teacher_name and new_teacher_name != edit_teacher_name and new_teacher_name not in st.session_state.teachers_db:
                                st.session_state.teachers_db[new_teacher_name] = st.session_state.teachers_db.pop(edit_teacher_name)
                                updated = True
                                edit_teacher_name = new_teacher_name
                                
                            if updated:
                                save_data(TEACHERS_JSON, st.session_state.teachers_db)
                                if "known_encodings_teachers" in st.session_state: del st.session_state["known_encodings_teachers"]
                                st.success("Teacher updated successfully!")
                                st.rerun()
                    with t_colB:
                        if st.button("Delete Teacher", type="primary"):
                            tdata = st.session_state.teachers_db.pop(edit_teacher_name)
                            for p in tdata.get("photo_paths", [tdata.get("photo_path")]):
                                if p and os.path.exists(p): os.remove(p)
                            save_data(TEACHERS_JSON, st.session_state.teachers_db)
                            if "known_encodings_teachers" in st.session_state: del st.session_state["known_encodings_teachers"]
                            st.success(f"Deleted Teacher: {edit_teacher_name}")
                            st.rerun()
                else:
                    st.info("No teachers enrolled.")

            st.markdown("---")
            st.subheader("🔑 Admin Credential Management")
            adm_edit_col1, adm_edit_col2 = st.columns(2)
            with adm_edit_col1:
                current_admin = st.selectbox("Select Admin to Edit", list(st.session_state.admins_db.keys()))
                new_adm_pass_change = st.text_input("Change Admin Password", type="password", key="change_adm_pass")
                if st.button("Update Admin Password"):
                    if new_adm_pass_change:
                        st.session_state.admins_db[current_admin] = new_adm_pass_change
                        save_data(ADMINS_JSON, st.session_state.admins_db)
                        st.success("Admin password updated!")
                    else:
                        st.warning("Please enter a new password.")
            with adm_edit_col2:
                st.warning("Critical Operations")
                if st.button("Delete This Admin Account", type="primary"):
                    if len(st.session_state.admins_db) > 1:
                        st.session_state.admins_db.pop(current_admin)
                        save_data(ADMINS_JSON, st.session_state.admins_db)
                        st.success(f"Deleted Admin: {current_admin}")
                        st.rerun()
                    else:
                        st.error("Cannot delete the only admin! Add another admin first.")

        with tab_logs:
            st.subheader("Global Security & Health Logs")
            if len(st.session_state.scan_logs) > 0:
                st.dataframe(pd.DataFrame(st.session_state.scan_logs), use_container_width=True)
            else:
                st.write("No database logs generated yet.")

# ==========================================
# TEACHER DASHBOARD
# ==========================================
elif role == "Teacher":
    st.title("Teacher Dashboard 👨‍🏫")
    st.markdown("Initiate class session and record precise quadrant-based attendance.")
    
    if 'session_started' not in st.session_state:
        st.session_state.session_started = False
    if 'active_teacher' not in st.session_state:
        st.session_state.active_teacher = None
    if 'pending_teacher' not in st.session_state:
        st.session_state.pending_teacher = None
    if 'session_ended_data' not in st.session_state:
        st.session_state.session_ended_data = None
        
    st.info("💡 **Camera Tip:** You can use your Mac's camera or select your iPhone camera via **Continuity Camera** for maximum resolution and distance precision.")

    if st.session_state.session_ended_data:
        st.success("Session Ended. Your final attendance report has been successfully compiled.")
        st.download_button(
            label="📄 Download End-of-Session PDF Report",
            data=st.session_state.session_ended_data['pdf_bytes'],
            file_name=f"Attendance_Report_{st.session_state.session_ended_data['date']}.pdf",
            mime="application/pdf"
        )
        if st.button("Close & Return to Dashboard"):
            st.session_state.session_ended_data = None
            st.rerun()

    elif not st.session_state.session_started:
        if not st.session_state.pending_teacher:
            st.subheader("Factor 1: Identity Verification (Face)")
            st.write("Please capture your photo to authenticate.")
            teacher_photo = st.camera_input("Take Teacher Photo")
            if teacher_photo:
                if not st.session_state.teachers_db:
                    st.error("No teachers enrolled! Please register via Admin Dashboard.")
                else:
                    with st.spinner("Analyzing face..."):
                        pil_img = Image.open(teacher_photo).convert("RGB")
                        img_np = np.array(ImageOps.exif_transpose(pil_img))
                        face_locs = fr.face_locations(img_np)
                        face_encs = fr.face_encodings(img_np, face_locs)
                        
                        if not face_encs:
                            st.error("No face detected. Please try again.")
                        else:
                            known_encodings, known_names = get_known_encodings("teachers")
                            match_found = False
                            if known_encodings:
                                # Use default strict auth tolerance
                                matches = fr.compare_faces(known_encodings, face_encs[0], tolerance=0.50)
                                if True in matches:
                                    idx = matches.index(True)
                                    st.session_state.pending_teacher = known_names[idx]
                                    st.success(f"Face recognized as {st.session_state.pending_teacher}!")
                                    st.rerun()
                            
                            if not match_found:
                                st.error("Face not recognized. Please ensure you are enrolled.")
        else:
            st.subheader("Factor 2: Security Password")
            st.write(f"Welcome, **{st.session_state.pending_teacher}**. Please provide your password to unlock the session.")
            t_pass_input = st.text_input("Teacher Password", type="password")
            
            t_mfa_col1, t_mfa_col2 = st.columns(2)
            with t_mfa_col1:
                if st.button("Unlock Session"):
                    t_info = st.session_state.teachers_db[st.session_state.pending_teacher]
                    if t_pass_input == t_info.get("password"):
                        st.session_state.active_teacher = st.session_state.pending_teacher
                        st.session_state.subject = t_info['subject']
                        st.session_state.session_started = True
                        st.session_state.session_start_time = datetime.now()
                        st.session_state.pending_teacher = None
                        st.success("MFA Successful. Session started.")
                        st.rerun()
                    else:
                        st.error("Invalid password.")
            with t_mfa_col2:
                if st.button("Wrong Person? (Rescan)", type="secondary"):
                    st.session_state.pending_teacher = None
                    st.rerun()
            
    else:
        # Check if the session has exceeded 1 hour (3600 seconds)
        elapsed_seconds = (datetime.now() - st.session_state.session_start_time).total_seconds()
        if elapsed_seconds > 3600:
            st.session_state.session_started = False
            st.session_state.active_teacher = None
            if 'class_scanned' in st.session_state:
                del st.session_state.class_scanned
            st.warning("⏱️ The 1-hour session limit has been reached. Session forcefully terminated.")
            st.rerun()
            
        mins_running = int(elapsed_seconds // 60)
        st.success(f"🟢 Active Session: **{st.session_state.subject}** (Instructor: {st.session_state.active_teacher}) — Running for {mins_running} minutes")
        
        col_end1, col_end2, _ = st.columns([2.5, 2.5, 4])
        with col_end1:
            if st.button("End Session & Close", use_container_width=True):
                st.session_state.session_started = False
                st.session_state.active_teacher = None
                if 'class_scanned' in st.session_state:
                    del st.session_state.class_scanned
                st.rerun()
        with col_end2:
            if st.button("End Session & Generate PDF Report", type="primary", use_container_width=True):
                pdf_bytes = generate_pdf_report(
                    st.session_state.session_start_time,
                    st.session_state.subject,
                    st.session_state.active_teacher
                )
                st.session_state.session_ended_data = {
                    'pdf_bytes': pdf_bytes,
                    'date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                }
                st.session_state.session_started = False
                st.session_state.active_teacher = None
                if 'class_scanned' in st.session_state:
                    del st.session_state.class_scanned
                st.rerun()
            
        st.markdown("---")
        st.markdown("---")
        st.subheader("Step 2: Precise Class Attendance Scan")
        st.write("Capture a high-resolution photo of the class.")
        st.info("🔭 **Pro Tip for Distance**: For large rooms, use the **High-Resolution Upload** or **Remote Shutter** for better quality. Avoid the 'Dashboard Camera' if you are standing far away as browser resolution is limited.")
        # Initialize timer variable locally to prevent NameError in non-remote tabs
        timer_seconds = 0
        
        camera_tab, upload_tab, remote_tab = st.tabs(["Dashboard Camera", "High-Resolution Upload", "Remote Shutter Delay (Mac Native)"])
        
        with st.expander("🛠️ Advanced AI Tuning (For High Accuracy)"):
            st.write("Adjust these weights if the system is missing students or misidentifying them.")
            st.warning("⚠️ **Misidentification Fix:** If the AI is calling Person A by Person B's name, **LOWER** the slider (Target 0.45). If a correct person has a low %, add more photos for them in Admin.")
            
            ai_model_mode = st.radio("AI Engine Mode", ["Standard (Fast)", "Ultra Accuracy (Precise/Slow)"], 
                                     index=0, horizontal=True, help="Ultra Accuracy uses CNN-based AI. It is much slower but can detect faces at difficult angles.")
            model_type = "cnn" if "Ultra" in ai_model_mode else "hog"
            
            col_tune1, col_tune2 = st.columns(2)
            with col_tune1:
                ai_sensitivity = st.slider("Recognition Sensitivity (Tolerance)", 0.40, 0.70, value=0.45, step=0.01, 
                                           help="Lower is stricter. Default 0.45 is best for large classes to prevent false names.")
            with col_tune2:
                upsample_lvl = st.select_slider("Detection Detail (Upsampling)", options=[1, 2, 3], value=2,
                                                help="Higher values detect smaller faces but are much slower. 2 is perfect for typical classrooms.")
        
        photo_to_process = None
        is_native_cv2_frame = False
        
        with camera_tab:
            st.warning("⚠️ Note: Browser cameras take the photo instantly on click. For a delayed snapshot so you can back away, use the 'Remote Shutter Delay' tab.")
            student_photo = st.camera_input("Take Class Photo", key="class_photo_cam")
            if student_photo:
                photo_to_process = student_photo
                
        with upload_tab:
            st.write("Upload a photo taken from any camera module (like your phone's primary lens).")
            student_upload = st.file_uploader("Upload Class Photo", type=['png', 'jpg', 'jpeg'], key="class_photo_upload")
            if student_upload:
                photo_to_process = student_upload
                
        with remote_tab:
            st.write("This directly activates your Mac's camera through the OS hardware to allow a true shutter delay.")
            timer_seconds = st.slider("Self-Test Timer (Seconds to get into frame)", 0, 10, value=0, key="shutter_timer")
            if st.button("Start Remote Snapshot Sequence"):
                import time
                timer_placeholder = st.empty()
                if timer_seconds > 0:
                    for i in range(timer_seconds, 0, -1):
                        timer_placeholder.info(f"⏳ Stand back! Camera fires in **{i}** seconds...")
                        time.sleep(1)
                
                timer_placeholder.warning("📸 SNAP!")
                
                # Activate hardware camera directly via standard OS ID 0
                cap = cv2.VideoCapture(0)
                # Warm up camera buffer for auto-exposure to adjust
                for _ in range(10): cap.read() 
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    photo_to_process = frame
                    is_native_cv2_frame = True
                    timer_placeholder.success("Photo captured strictly post-delay!")
                else:
                    timer_placeholder.error("Could not access OS hardware camera. Check permissions.")
                
        if photo_to_process is not None:
            # We process natively instead of dealing with Streamlit component delays
            if timer_seconds > 0 and not is_native_cv2_frame:
                timer_placeholder = st.empty()
                import time
                for i in range(timer_seconds, 0, -1):
                    timer_placeholder.info(f"⏳ Processing begins in **{i}** seconds... step into frame!")
                    time.sleep(1)
                timer_placeholder.empty()
                
            st.write("### 16-Zone Precision Grid Results")
            with st.spinner(f"Detecting faces via {model_type.upper()} Clean-Scan AI (Upsample x{upsample_lvl})..."):
                if is_native_cv2_frame:
                    result_img, face_count, detected_students = process_precision_grid(photo_to_process, is_bgr=True, tolerance=ai_sensitivity, upsample=upsample_lvl, model=model_type)
                else:
                    result_img, face_count, detected_students = process_precision_grid(photo_to_process, tolerance=ai_sensitivity, upsample=upsample_lvl, model=model_type)
            
            if result_img:
                st.image(result_img, caption=f"Precision Grid Scan Complete. Detected {face_count} faces.", use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Faces Detected", face_count)
                with col2:
                    if detected_students:
                        st.write("**Identified Students:**")
                        for student in detected_students:
                            st.success(f"✅ {student}")
                    else:
                        st.info("No matching students found or enrolled.")
                
                if st.button("Commit Attendance to Database"):
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.scan_logs.append({
                        "Timestamp": now,
                        "Subject": st.session_state.subject,
                        "Faces_Detected": face_count,
                        "Students_Identified": ", ".join(detected_students),
                        "Status": "Committed"
                    })
                    
                    for st_name in detected_students:
                        st.session_state.attendance_logs.append({
                            "Student": st_name,
                            "Date": now,
                            "Subject": st.session_state.subject,
                            "Status": "Present"
                        })
                    
                    sync_logs()
                    st.success("Attendance successfully committed to Database!")
            else:
                st.error("Error processing the image.")
                
        st.markdown("---")
        st.subheader("Step 3: Automated Interval Scanning")
        st.write("Automatically capture and log attendance via Mac Native camera at regular intervals.")
        st.info("💡 **Perfect for passively logging attendance without clicking buttons during the class.**")
        interval = st.selectbox("Interval (Minutes)", [1, 5, 10, 15])
        
        col_auto1, col_auto2, _ = st.columns([2.5, 2.5, 4])
        with col_auto1:
            start_auto = st.button(f"Start {interval}-Minute Auto-Scan", type="primary", use_container_width=True)
        with col_auto2:
            if st.button("🛑 Stop Auto-Scan", help="Click to cancel background scanning task.", use_container_width=True):
                st.rerun()
            
        if start_auto:
            st.warning("⚠️ **Dashboard Locked & Scanning Active.** The dashboard will now automatically take photos in the background. To stop this and resume standard dashboard usage, click the Stop button above or refresh your browser!")
            scan_placeholder = st.empty()
            preview_placeholder = st.empty()
            
            st.markdown("### Session Scan History")
            results_placeholder = st.empty()
            accumulated_html = ""
            
            import time
            previous_detected_students = None
            
            while True:
                # Check 1-hour hard limit inside the loop
                loop_elapsed_seconds = (datetime.now() - st.session_state.session_start_time).total_seconds()
                if loop_elapsed_seconds > 3600:
                    st.error("🛑 1-Hour Session Limit Reached. Automated scanning suspended. Please restart the session.")
                    break
                    
                cap = None
                last_frame = None
                
                # We countdown using smaller sleeps so the user sees live progress
                for remaining in range(interval * 60, 0, -1):
                    mins, secs = divmod(remaining, 60)
                    scan_placeholder.info(f"⏳ **Auto-Scan Active.** Next automated snapshot in: **{mins:02d}:{secs:02d}**")
                    
                    if remaining <= 5:
                        if cap is None:
                            cap = cv2.VideoCapture(0)
                        
                        start_tick = time.time()
                        while time.time() - start_tick < 1.0: # Loop reading quickly to render live stream smoothly
                            ret, live_frame = cap.read()
                            if ret:
                                last_frame = live_frame
                                preview_placeholder.image(live_frame, channels="BGR", use_container_width=True, caption=f"🔴 Live Preview - Snapping in {remaining}s...")
                            time.sleep(0.05) 
                    else:
                        time.sleep(1)
                
                scan_placeholder.warning("📸 Snapshot acquired! Processing photo...")
                
                if cap:
                    cap.release()
                    cap = None
                preview_placeholder.empty()
                
                if last_frame is not None:
                    # UPDATED: Use the 16-zone precision grid instead of the old broken process_quadrants
                    result_img, face_count, detected_students = process_precision_grid(last_frame, is_bgr=True, tolerance=ai_sensitivity, upsample=upsample_lvl, model=model_type)
                    
                    if face_count >= 0:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        bunked_students = []
                        if previous_detected_students is not None:
                            # A student is bunked if they were present LAST scan, but are missing in THIS scan
                            bunked_students = [s for s in previous_detected_students if s not in detected_students]
                            
                        # Update the state to THIS scan's students for the next loop
                        previous_detected_students = detected_students
                        
                        # Add to Scan Logs
                        log_entry = {
                            "Timestamp": now,
                            "Subject": st.session_state.subject,
                            "Faces_Detected": face_count,
                            "Students_Identified": ", ".join(detected_students) if detected_students else "None",
                            "Status": f"Auto-Committed ({interval}m)"
                        }
                        if bunked_students:
                            log_entry["Students_Bunked"] = ", ".join(bunked_students)
                        st.session_state.scan_logs.append(log_entry)
                        
                        # Update Attendance Lists
                        for st_name in detected_students:
                            st.session_state.attendance_logs.append({
                                "Student": st_name,
                                "Date": now,
                                "Subject": st.session_state.subject,
                                "Status": "Present"
                            })
                            
                        for st_name in bunked_students:
                            st.session_state.attendance_logs.append({
                                "Student": st_name,
                                "Date": now,
                                "Subject": st.session_state.subject,
                                "Status": "Bunked"
                            })
                            
                        sync_logs()
                        
                        # Accumulate the UI dynamically with ALL scan results 
                        present_str = ", ".join(detected_students) if detected_students else "No one detected"
                        bunked_str = ", ".join(bunked_students) if bunked_students else "None"
                        
                        new_block = f'''
                        <div style="padding: 15px; border-radius: 8px; background-color: #2b2b2b; border: 1px solid #444; margin-bottom: 10px;">
                            <h4 style="margin-top: 0; color: white;">Scan Results ({now})</h4>
                            <p style="color: #4CAF50; margin-bottom: 5px;"><strong>✅ Present:</strong> {present_str}</p>
                            <p style="color: #F44336; margin-bottom: 0;"><strong>❌ Bunked:</strong> {bunked_str}</p>
                        </div>
                        '''
                        
                        accumulated_html = new_block + accumulated_html
                        results_placeholder.markdown(accumulated_html, unsafe_allow_html=True)
                        
                        scan_placeholder.success(f"✅ Auto-Scan Complete ({now})! Logged {face_count} faces. Resuming timer...")
                        time.sleep(5) # Let success msg linger briefly before resuming countdown loop
                else:
                    scan_placeholder.error("Could not access OS hardware camera. Skipping this interval.")
                    time.sleep(5)

# ==========================================
# STUDENT/PARENT DASHBOARD
# ==========================================
elif role == "Student/Parent":
    st.title("Student & Parent Portal 🎓")
    st.markdown("Track precise attendance metrics via facial matching.")
    
    st.subheader("Look up Attendance")
    search_name = st.text_input("Enter Student Full Name to View Data")
    
    if search_name:
        if search_name not in st.session_state.students_db:
            st.error(f"Student '{search_name}' is not registered in the system.")
        else:
            student_logs = [log for log in st.session_state.attendance_logs if log['Student'].lower() == search_name.lower()]
            
            if not student_logs:
                st.info(f"No attendance logs found for {search_name} yet.")
            else:
                st.success(f"Found {len(student_logs)} attendance records for {search_name}.")
                df = pd.DataFrame(student_logs)
                st.dataframe(df, use_container_width=True)
                
                fig = px.pie(df, names='Status', title=f"Attendance Breakdown for {search_name}", color_discrete_sequence=['#2ca02c'])
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Developed for AAPT - Advanced Autonomous Persistence Tracking")
