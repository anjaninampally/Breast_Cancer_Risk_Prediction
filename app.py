import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify
import pickle
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import timedelta
from datetime import datetime
from collections import defaultdict
import calendar
import plotly.graph_objs as go
import plotly.io as pio
from flask_login import login_required
from flask_login import LoginManager
from flask_login import current_user
import pytz
from datetime import date
app = Flask(__name__)

from datetime import timedelta
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.permanent_session_lifetime = timedelta(days=7)
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # This ensures @login_required redirects to /login

from flask_login import UserMixin
class User(UserMixin):
    def __init__(self, id, username, email=None):
        self.id = id
        self.username = username
        self.email = email

    def get_id(self):
        return str(self.id)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.permanent_session_lifetime = timedelta(days=7)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    model = None
def get_db():
    db_path = os.path.join(os.path.dirname(__file__), 'database.db')
    #db = sqlite3.connect(db_path, timeout=10) 
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db
def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                features TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS patient_info (
                user_id INTEGER PRIMARY KEY,
                name TEXT,
                parent_name TEXT,
                dob TEXT,
                weight REAL,
                height REAL,
                previous_health_issues TEXT,
                parent_health_issues TEXT
            );

            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                doctor_id INTEGER,
                doctor_name TEXT,
                patient_name TEXT,
                reason TEXT,
                phone TEXT,
                address TEXT,
                appointment_time TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)
        db.commit()

init_db()
@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user_data = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user_data:
        #return User(id=user_data['id'], username=user_data['username'])
        return User(id=user_data['id'], username=user_data['username'], email=user_data['email'])
    return None
def generate_prediction_trend_plot():
    db = get_db()
    user_id = current_user.id
    rows = db.execute("SELECT timestamp FROM predictions WHERE user_id = ?", (user_id,)).fetchall()

    date_counts = defaultdict(int)
    for row in rows:
        date_str = row['timestamp'].split(' ')[0]  # Get date part only
        date_counts[date_str] += 1

    sorted_dates = sorted(date_counts.items())
    dates = [d for d, _ in sorted_dates]
    counts = [c for _, c in sorted_dates]

    fig = go.Figure(data=go.Scatter(x=dates, y=counts, mode='lines+markers'))
    fig.update_layout(title='Prediction Trend Over Time', xaxis_title='Date', yaxis_title='Predictions')
    return fig.to_html(full_html=False)
def generate_confidence_score_plot():
    db = get_db()
    user_id = current_user.id
    rows = db.execute("SELECT timestamp, confidence FROM predictions WHERE user_id = ?", (user_id,)).fetchall()

    timestamps = [row['timestamp'] for row in rows]
    confidences = [row['confidence'] for row in rows]

    fig = go.Figure(data=go.Scatter(x=timestamps, y=confidences, mode='lines+markers'))
    fig.update_layout(title='Confidence Scores Over Time', xaxis_title='Timestamp', yaxis_title='Confidence (%)')
    return fig.to_html(full_html=False)
def generate_top_features_plot():
    # Placeholder logic — replace with real feature impact analysis if available
    features = ['Clump Thickness', 'Uniform Cell Size', 'Uniform Cell Shape', 'Bare Nuclei']
    importances = [0.25, 0.2, 0.18, 0.15]

    fig = go.Figure(data=go.Bar(x=features, y=importances))
    fig.update_layout(title='Top Influential Features (Example)', xaxis_title='Feature', yaxis_title='Importance')
    return fig.to_html(full_html=False)
def generate_prediction_distribution_plot():
    db = get_db()
    user_id = current_user.id
    rows = db.execute("SELECT prediction FROM predictions WHERE user_id = ?", (user_id,)).fetchall()

    malignant = sum(1 for r in rows if r['prediction'] == 'Malignant')
    benign = sum(1 for r in rows if r['prediction'] == 'Benign')

    fig = go.Figure(data=[go.Pie(
        labels=['Malignant', 'Benign'],
        values=[malignant, benign],
        hole=0.4
    )])
    fig.update_layout(title='Malignant vs Benign Predictions')
    return fig.to_html(full_html=False)
def generate_recommendation(risk_level):
    if risk_level == 'High':
        return "Consult an Oncologist Immediately"
    elif risk_level == 'Medium':
        return "Schedule Routine Checkup"
    else:
        return "Maintain Healthy Lifestyle"
def get_treatment_suggestions(risk_level):
    if risk_level == 'High':
        return ["Chemotherapy", "MRI Scan", "Consult an Oncologist"]
    elif risk_level == 'Medium':
        return ["Mammography", "Balanced Diet", "Follow-up Screening"]
    elif risk_level == 'Low':
        return ["Annual Screening", "Self-Examination Tips"]
    return []
# --- Perform migration if new columns are missing ---
def migrate_db():
    db = get_db()
    cursor = db.execute("PRAGMA table_info(users);")
    existing_columns = [row['name'] for row in cursor.fetchall()]
    new_columns = {
        'notifications': "BOOLEAN DEFAULT 1",
        'darkmode': "BOOLEAN DEFAULT 0"
    }

    for column, definition in new_columns.items():
        if column not in existing_columns:
            db.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
            print(f"✅ Added column: {column}")

    db.commit()

migrate_db()

@app.before_request
def load_logged_in_user():
    #user_id = current_user.id
    if not current_user.is_authenticated:
        g.current_user = type('AnonymousUser', (), {'is_authenticated': False})()
    else:
        user_id = current_user.id
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        g.current_user = type('User', (), {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'is_authenticated': True
        })()
        session['darkmode'] = user['darkmode']
        session['notifications'] = user['notifications']
@app.context_processor
def inject_user():
    return dict(current_user=g.current_user)
from collections import namedtuple
def get_last_prediction(user_id):
    db = get_db()
    row = db.execute(
        "SELECT prediction, confidence, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,)
    ).fetchone()
    
    if row:
        # Determine risk level from prediction (Malignant = High, Benign = Low)
        risk_level = 'High' if row['prediction'] == 'Malignant' else 'Low'
        
        # Create a simple structure to hold prediction data
        Prediction = namedtuple('Prediction', ['risk_level', 'confidence', 'timestamp'])
        return Prediction(risk_level=risk_level, confidence=row['confidence'], timestamp=row['timestamp'])
    return None
import plotly.graph_objs as go
import plotly.io as pio
import sqlite3
def generate_risk_plot(user_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT confidence FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", (user_id,))
    confidences = [row[0] for row in c.fetchall()]
    conn.close()

    if not confidences:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=confidences, mode='lines+markers', name='Confidence Score'))
    fig.update_layout(title='Recent Prediction Confidence', xaxis_title='Prediction #', yaxis_title='Confidence')

    return pio.to_html(fig, full_html=False)
from flask_login import login_required, current_user
from flask import render_template, request, jsonify
import requests
from flask_login import login_required, current_user
@app.route("/")
@login_required
def home():
    db = get_db()
    user_id = current_user.id 
    row = db.execute(
        "SELECT prediction, confidence, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (current_user.id,)
    ).fetchone()
    #db.close()

    health_status = 'High' if row and row['prediction'] == 'Malignant' else 'Low'
    patient = db.execute("SELECT * FROM patient_info WHERE user_id = ?", (user_id,)).fetchone()
    #db.close()
    max_date = date.today().isoformat()
    db.close()
    return render_template(
        "home.html",
        user_data=current_user, 
        last_prediction=row,
        health_status=health_status,
        patient=patient,
        max_date=max_date,
    )
@app.route("/nearby-hospitals")
@login_required
def nearby_hospitals():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify([])

    try:
        # Step 1: Get the user's city or town using reverse geocoding
        reverse = requests.get("https://nominatim.openstreetmap.org/reverse", params={
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10,
            "addressdetails": 1
        }, headers={"User-Agent": "breast-cancer-app"}).json()

        address = reverse.get("address", {})
        city_or_town = address.get("city") or address.get("town") or address.get("village") or address.get("state")

        if not city_or_town:
            return jsonify([])

        # Step 2: Search for hospitals within that city
        search = requests.get("https://nominatim.openstreetmap.org/search", params={
            "q": f"hospital in {city_or_town}",
            "format": "json",
            "limit": 5,
            "addressdetails": 1
        }, headers={"User-Agent": "breast-cancer-app"})

        data = search.json()
        hospitals = []
        for h in data:
            name = h.get("display_name").split(',')[0]
            addr = h.get("address", {})
            formatted_address = ", ".join(filter(None, [
                addr.get("road"),
                addr.get("suburb"),
                addr.get("city") or addr.get("town"),
                addr.get("state"),
                addr.get("postcode"),
                addr.get("country")
            ]))
            hospitals.append({
                "name": name,
                "address": formatted_address
            })
        return jsonify(hospitals)
    except Exception as e:
        return jsonify([])

@app.route('/submit-info', methods=['POST'])
@login_required
def submit_info():
    db = get_db()
    user_id = current_user.id

    # Get form data
    name = request.form['name']
    parent_name = request.form['parent_name']
    dob = request.form['dob']
    weight = request.form['weight']
    height = request.form['height']
    prev_issues = request.form.get('previous_health_issues', '')
    parent_issues = request.form.get('parent_health_issues', '')

    # Insert or update record
    existing = db.execute("SELECT * FROM patient_info WHERE user_id = ?", (user_id,)).fetchone()
    if existing:
        db.execute("""
            UPDATE patient_info
            SET name = ?, parent_name = ?, dob = ?, weight = ?, height = ?, previous_health_issues = ?, parent_health_issues = ?
            WHERE user_id = ?
        """, (name, parent_name, dob, weight, height, prev_issues, parent_issues, user_id))
    else:
        db.execute("""
            INSERT INTO patient_info (user_id, name, parent_name, dob, weight, height, previous_health_issues, parent_health_issues)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name, parent_name, dob, weight, height, prev_issues, parent_issues))

    db.commit()
    return redirect(url_for('home'))
@app.route('/edit-info')
@login_required
def edit_info():
    db = get_db()
    user_id = current_user.id
    patient = db.execute("SELECT * FROM patient_info WHERE user_id = ?", (user_id,)).fetchone()
    max_date = date.today().isoformat()

    # Fetch prediction too if needed
    last_prediction = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,)
    ).fetchone()

    # You may calculate health_status here if used
    health_status = None
    if last_prediction:
        health_status = 'High' if float(last_prediction['confidence']) >= 70 else 'Low'

    return render_template(
        'home.html',
        patient=patient,
        edit_data=patient,
        max_date=max_date,
        user_data=current_user,
        last_prediction=last_prediction,
        health_status=health_status
    )
'''@app.route('/treatment')
def treatment():
    return render_template('treatment.html')'''

@app.route('/contact')
def contact():
    return render_template('contact.html')

import numpy as np

@app.route('/predict', methods=['POST', 'GET'])
@login_required
def predict():
    if request.method == 'POST':
        if model is None:
            flash("Prediction model not available.", "danger")
            return redirect(url_for('home'))

        feature_names = [
            'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
            'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
            'bland_chromatin', 'normal_nucleoli', 'mitoses'
        ]

        try:
            features = [float(request.form[name]) for name in feature_names]
        except (ValueError, KeyError):
            flash('Invalid input data', 'danger')
            return redirect(url_for('predict'))

        try:
            prediction_code = model.predict([features])[0]
            result = 'Benign' if prediction_code == 2 else 'Malignant'

            # Estimate confidence using decision function + sigmoid
            decision = model.decision_function([features])[0]
            confidence = 1 / (1 + np.exp(-decision))
            confidence = round(confidence * 100, 2)

            db = get_db()
            db.execute(
                "INSERT INTO predictions (user_id, features, prediction, confidence) VALUES (?, ?, ?, ?)",
                (current_user.id, str(features), result, confidence)
            )
            db.commit()

            return render_template('results.html', prediction=result, confidence=confidence, features=features)

        except Exception as e:
            print("Error making prediction:", e)
            flash("An error occurred while making prediction.", "danger")
            return redirect(url_for('predict'))

    return render_template('predict.html')



@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    user_id = current_user.id
    filter_type = request.args.get('filter_type')

    # Apply filter
    if filter_type in ['Benign', 'Malignant']:
        rows = db.execute("SELECT * FROM predictions WHERE user_id = ? AND prediction = ?", (user_id, filter_type)).fetchall()
    else:
        rows = db.execute("SELECT * FROM predictions WHERE user_id = ?", (user_id,)).fetchall()

    total_predictions = len(rows)
    malignant_count = sum(1 for r in rows if r['prediction'] == 'Malignant')
    benign_count = sum(1 for r in rows if r['prediction'] == 'Benign')

    # Prepare pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Malignant', 'Benign'],
        values=[malignant_count, benign_count],
        hole=0.4
    )])
    fig.update_layout(title='Prediction Distribution', margin=dict(t=50, b=0, l=0, r=0))
    distribution_plot = fig.to_html(full_html=False)

    # Feature plot placeholder
    feature_plot = None  # Optional: add real plot

    # Recent predictions (last 5 after filtering)
    recent_predictions = rows[-5:]
    
    
    # Convert timestamps
    local_tz = pytz.timezone("Asia/Kolkata")
    converted_predictions = []
    from datetime import datetime
    for pred in recent_predictions:
        timestamp = pred['timestamp']
        if isinstance(timestamp, str):
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        else:
            dt = timestamp
        dt = pytz.utc.localize(dt).astimezone(local_tz)
        dt_str = dt.strftime('%d %B %Y, %I:%M %p').lstrip('0')  # ✅ Safe for Windows
        converted_predictions.append({
            'prediction': pred['prediction'],
            'confidence': pred['confidence'],
            'timestamp': dt_str
        })


    # Generate charts
    plot_trend = generate_prediction_trend_plot()
    plot_confidence = generate_confidence_score_plot()
    plot_features = generate_top_features_plot()
    plot_distribution = generate_prediction_distribution_plot()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        malignant_count=malignant_count,
        benign_count=benign_count,
        plot_html=distribution_plot,
        feature_accuracy_plot=feature_plot,
        recent_predictions=converted_predictions,
        plot_trend=plot_trend,
        plot_confidence=plot_confidence,
        plot_features=plot_features,
        plot_distribution=plot_distribution,
        selected_filter=filter_type
    )

from flask import request, render_template, redirect, url_for, flash
from flask_login import login_user
from werkzeug.security import check_password_hash

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form

        db = get_db()
        user_data = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user_data and check_password_hash(user_data['password'], password):
            user = User(id=user_data['id'], username=user_data['username'])
            login_user(user, remember=remember)

            session['darkmode'] = user_data['darkmode']
            session['notifications'] = user_data['notifications']

            flash('Login successful.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

from flask_login import logout_user

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


import re
from flask import request, flash, redirect, url_for, render_template
from werkzeug.security import generate_password_hash

from werkzeug.security import generate_password_hash

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        db = get_db()
        try:
            hashed_password = generate_password_hash(password)
            db.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                       (username, email, hashed_password))
            db.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'danger')
    return render_template('register.html')


from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
import datetime

# Setup Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.example.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your@example.com'
app.config['MAIL_PASSWORD'] = 'your-password'
mail = Mail(app)

s = URLSafeTimedSerializer(app.secret_key)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user:
            token = s.dumps(email, salt='reset-password')
            expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            db.execute("UPDATE users SET reset_token = ?, token_expiration = ? WHERE id = ?", (token, expiration, user['id']))
            db.commit()
            reset_link = url_for('reset_password', token=token, _external=True)
            msg = Message("Password Reset Request", sender="your@example.com", recipients=[email])
            msg.body = f"Click the link to reset your password: {reset_link}"
            mail.send(msg)
            flash('Password reset link sent to your email.', 'info')
        else:
            flash('No account found with that email.', 'danger')
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='reset-password', max_age=3600)
    except:
        flash('The reset link is invalid or has expired.', 'danger')
        return redirect(url_for('login'))

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user or user['reset_token'] != token:
        flash('Invalid reset request.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form['password']
        db.execute("UPDATE users SET password = ?, reset_token = NULL, token_expiration = NULL WHERE email = ?",
                   (generate_password_hash(new_password), email))
        db.commit()
        flash('Password reset successful. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/privacy')
def privacy():
    return render_template('privacy.html')
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    db = get_db()
    user_id = current_user.id

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'account_info':
            # Update username and email
            new_username = request.form.get('username')
            new_email = request.form.get('email')
            try:
                db.execute("UPDATE users SET username = ?, email = ? WHERE id = ?", (new_username, new_email, user_id))
                db.commit()
                flash("Account info updated successfully.", "success")
            except sqlite3.IntegrityError:
                flash("Username or email already exists.", "danger")

        elif form_type == 'preferences':
            # Update dark mode and notifications
            dark_mode = 1 if request.form.get('darkmode') == 'on' else 0
            notifications = 1 if request.form.get('notifications') == 'on' else 0
            db.execute("UPDATE users SET darkmode = ?, notifications = ? WHERE id = ?", (dark_mode, notifications, user_id))
            db.commit()
            session['darkmode'] = dark_mode
            session['notifications'] = notifications
            flash("Preferences updated successfully.", "success")

    user = db.execute('SELECT username, email, darkmode, notifications FROM users WHERE id = ?', (user_id,)).fetchone()
    return render_template(
        'settings.html',
        user=user,
        darkmode_enabled=bool(user['darkmode']),
        notifications_enabled=bool(user['notifications'])
    )

@app.route('/settings/delete-account', methods=['POST'])
@login_required
def delete_account():
    db = get_db()
    user_id = current_user.id
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    db.commit()
    session.clear()
    flash("Your account has been deleted.", "info")
    return redirect(url_for('register'))
'''def init_db():
    import sqlite3
    with sqlite3.connect('database.db') as conn:
        with open('schema.sql') as f:
            conn.executescript(f.read())'''
    #print("✅ Database initialized.")
# Example doctor recommendations
doctors = [
    {'id': 1, 'name': 'Dr. Anjali Kumar', 'specialty': 'Oncologist', 'phone': '+91-9876543210',
     'address': '123 Cancer Care St, Mumbai, Maharashtra', 'lat': 19.0760, 'lon': 72.8777},
    {'id': 2, 'name': 'Dr. Rajesh Singh', 'specialty': 'Surgeon', 'phone': '+91-9123456789',
     'address': '456 Health Ave, Delhi, Delhi', 'lat': 28.6139, 'lon': 77.2090},
    {'id': 3, 'name': 'Dr. Priya Sharma', 'specialty': 'Radiologist', 'phone': '+91-9988776655',
     'address': '789 Wellness Rd, Pune, Maharashtra', 'lat': 18.5204, 'lon': 73.8567},
    {'id': 4, 'name': 'Dr. Sameer Patel', 'specialty': 'Oncologist', 'phone': '+91-9876543211',
     'address': '321 Hope St, Ahmedabad, Gujarat', 'lat': 23.0225, 'lon': 72.5714},
    {'id': 5, 'name': 'Dr. Kavita Joshi', 'specialty': 'Surgeon', 'phone': '+91-9123456790',
     'address': '654 Care Blvd, Jaipur, Rajasthan', 'lat': 26.9124, 'lon': 75.7873},
    {'id': 6, 'name': 'Dr. Arjun Mehta', 'specialty': 'Radiologist', 'phone': '+91-9988776600',
     'address': '987 Health Lane, Lucknow, Uttar Pradesh', 'lat': 26.8467, 'lon': 80.9462},
    {'id': 7, 'name': 'Dr. Neha Verma', 'specialty': 'Oncologist', 'phone': '+91-9876543299',
     'address': '123 Healing Rd, Bhopal, Madhya Pradesh', 'lat': 23.2599, 'lon': 77.4126},
    {'id': 8, 'name': 'Dr. Rohit Gupta', 'specialty': 'Surgeon', 'phone': '+91-9123456780',
     'address': '456 Cure Ave, Chennai, Tamil Nadu', 'lat': 13.0827, 'lon': 80.2707},
    {'id': 9, 'name': 'Dr. Anjali Singh', 'specialty': 'Radiologist', 'phone': '+91-9988776611',
     'address': '789 Wellness Blvd, Hyderabad, Telangana', 'lat': 17.3850, 'lon': 78.4867},
    {'id': 10, 'name': 'Dr. Vikram Rao', 'specialty': 'Oncologist', 'phone': '+91-9876543200',
     'address': '321 Hope Lane, Kolkata, West Bengal', 'lat': 22.5726, 'lon': 88.3639},
]
from flask import request, jsonify, render_template, redirect, url_for, flash
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
@app.route('/treatment')
@login_required
def treatment():
    # Example explanation content
    field_explanations = {
        "Clump Thickness": "Indicates the thickness of cell clusters. Higher values may suggest abnormal cell growth.",
        "Uniform Cell Size": "Measures consistency in cell size. Irregular sizes can indicate malignancy.",
        "Uniform Cell Shape": "Assesses similarity in cell shapes. Variation can be a sign of cancer.",
        "Marginal Adhesion": "Refers to how closely cells stick together. Looser adhesion may point to malignancy.",
        "Single Epithelial Size": "Size of the epithelial cells. Larger sizes are often abnormal.",
        "Bare Nuclei": "Counts nuclei that lack surrounding cytoplasm. High numbers are linked to cancer.",
        "Bland Chromatin": "Describes texture of the nucleus. Coarse texture may be cancerous.",
        "Normal Nucleoli": "Presence of small, round structures in the nucleus. Prominent nucleoli may indicate malignancy.",
        "Mitoses": "Cell division rate. Higher rates are seen in malignant tumors."
    }

    prediction_info = {
        "Benign": "Non-cancerous. Tumors grow slowly and are usually not life-threatening.",
        "Malignant": "Cancerous. Can grow quickly and spread to other parts of the body."
    }

    confidence_info = (
        "The confidence score represents how certain the model is about its prediction. "
        "A higher percentage indicates stronger certainty. For example, 63% confidence in a benign prediction means "
        "the model is 63% sure that the tumor is non-cancerous."
    )

    treatments = {
        "Benign": [
            "Regular monitoring and check-ups",
            "Surgical removal if necessary",
            "Healthy lifestyle and diet maintenance"
        ],
        "Malignant": [
            "Chemotherapy",
            "Radiation therapy",
            "Surgical removal of the tumor",
            "Hormone therapy or immunotherapy"
        ]
    }
    return render_template('treatment.html',
                           field_explanations=field_explanations,
                           prediction_info=prediction_info,
                           confidence_info=confidence_info,
                           treatments=treatments,
                           doctors=doctors)

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

@app.route('/nearby-doctors')
def nearby_doctors():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    # Dummy data — replace with real API or database query
    doctors = [
        {"name": "Dr. Anjali Mehta", "address": "City Hospital, Sector 5", "phone": "9876543210"},
        {"name": "Dr. Rahul Khanna", "address": "Sunrise Clinic, Block B", "phone": "9123456789"}
    ]
    return jsonify(doctors)

@app.route('/book-appointment')
def book_appointment():
    doctor_name = request.args.get('doctor')
    return render_template("book_appointment.html", doctor_name=doctor_name)

@app.route('/submit_appointment', methods=["POST"])
def submit_appointment():
    doctor_name = request.form.get("doctor_name")
    patient_name = request.form.get("patient_name")
    email = request.form.get("email")
    date = request.form.get("date")
    time = request.form.get("time")

    # Save appointment logic...

    return render_template("confirmation.html",
                           doctor_name=doctor_name,
                           patient_name=patient_name, 
                           date=date, 
                           time=time)





@app.errorhandler(404)
def page_not_found(e):
    return "404 Not Found", 404
if __name__ == '__main__':
    app.run(debug=True, threaded=False)
