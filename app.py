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

app = Flask(__name__)

from datetime import timedelta

app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)  # Optional, adjust as needed

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # This ensures @login_required redirects to /login

from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

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
    db = sqlite3.connect('database.db')
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
        """)
        db.commit()

init_db()

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user_data = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user_data:
        return User(id=user_data['id'], username=user_data['username'])
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST', 'GET'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            feature_names = [
                'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
                'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
                'bland_chromatin', 'normal_nucleoli', 'mitoses'
            ]
            features = []
            for name in feature_names:
                value = request.form.get(name, '').strip()
                try:
                    features.append(float(value))
                except ValueError:
                    flash(f'Invalid input for {name.replace("_", " ").title()}', 'danger')
                    return redirect(url_for('predict'))

            prediction = model.predict([features])[0]
            result = 'Benign' if prediction == 2 else 'Malignant'
            proba = model.predict_proba([features])[0].max() if hasattr(model, "predict_proba") else 0.9
            confidence = round(proba * 100, 2)

            db = get_db()
            db.execute(
                "INSERT INTO predictions (user_id, features, prediction, confidence) VALUES (?, ?, ?, ?)",
                (current_user.id, str(features), result, confidence)
            )
            db.commit()

            return render_template('results.html', prediction=result, confidence=confidence, features=features)
        except Exception as e:
            print("Error making prediction:", e)
            return f"Error: {e}"
        if model is None:
            flash("Prediction model not available.", "danger")
            return redirect(url_for('home'))


    return render_template('predict.html')

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    user_id = current_user.id

    # Fetch prediction records
    rows = db.execute("SELECT * FROM predictions WHERE user_id = ?", (user_id,)).fetchall()

    total_predictions = len(rows)
    malignant_count = sum(1 for r in rows if r['prediction'] == 'Malignant')
    benign_count = total_predictions - malignant_count

    # Prepare pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Malignant', 'Benign'],
        values=[malignant_count, benign_count],
        hole=0.4
    )])
    fig.update_layout(title='Prediction Distribution', margin=dict(t=50, b=0, l=0, r=0))
    distribution_plot = fig.to_html(full_html=False)

    # (Optional) Feature-wise analysis placeholder — replace with real logic later
    feature_plot = None  # or generate another plotly chart

    # Fetch recent predictions (last 5)
    recent_predictions = db.execute(
        "SELECT prediction, confidence, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5",
        (user_id,)
    ).fetchall()
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
        recent_predictions=recent_predictions,
        plot_trend=plot_trend,
        plot_confidence=plot_confidence,
        plot_features=plot_features,
        plot_distribution=plot_distribution
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


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, generate_password_hash(password))
            )
            db.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'danger')
    return render_template('register.html')

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
    return redirect(url_for('home'))
'''def init_db():
    import sqlite3
    with sqlite3.connect('database.db') as conn:
        with open('schema.sql') as f:
            conn.executescript(f.read())'''
    #print("✅ Database initialized.")

# Only run this once to create tables
#init_db()

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
