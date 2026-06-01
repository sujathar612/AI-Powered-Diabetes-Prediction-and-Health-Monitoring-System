from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, Response, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, current_user

from config import Config
from models import db, User, Prediction, LifestyleEntry, VitalSigns, MedicationLog, HealthGoals, Badge, UserBadge, UserPoints, Streak, Doctor, Appointment
import joblib
import numpy as np
from datetime import datetime, date, timedelta
import json
import time
import pandas as pd
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the trained model
model = joblib.load('diabetes_model.pkl')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ===== GAMIFICATION FUNCTIONS =====

def award_points(user_id, points, reason):
    """Award points to user and check for badges"""
    user_points = UserPoints.query.filter_by(user_id=user_id).first()
    if not user_points:
        user_points = UserPoints(user_id=user_id)
        db.session.add(user_points)

    user_points.total_points += points
    user_points.last_updated = datetime.utcnow()

    # Check for badges
    check_badges(user_id)

    db.session.commit()

def check_badges(user_id):
    """Check if user qualifies for new badges"""
    user_points = UserPoints.query.filter_by(user_id=user_id).first()
    if not user_points:
        return

    badges = Badge.query.all()
    for badge in badges:
        if user_points.total_points >= badge.points_required:
            # Check if user already has this badge
            existing = UserBadge.query.filter_by(user_id=user_id, badge_id=badge.id).first()
            if not existing:
                user_badge = UserBadge(user_id=user_id, badge_id=badge.id)
                db.session.add(user_badge)

def update_streak(user_id, streak_type):
    """Update user streak for specific type"""
    streak = Streak.query.filter_by(user_id=user_id, streak_type=streak_type).first()
    if not streak:
        streak = Streak(user_id=user_id, streak_type=streak_type)
        db.session.add(streak)

    streak.current_count += 1
    if streak.current_count > streak.longest_count:
        streak.longest_count = streak.current_count
    streak.last_updated = datetime.utcnow()

    # Award points for streak
    if streak.current_count % 7 == 0:  # Weekly streak bonus
        award_points(user_id, 50, f"7-day {streak_type} streak")

    db.session.commit()

def initialize_gamification():
    """Initialize default badges"""
    badges = [
        Badge(name="First Steps", description="Complete your first health entry", icon="fas fa-shoe-prints", points_required=10, category="consistency"),
        Badge(name="Consistent Logger", description="Log health data for 7 consecutive days", icon="fas fa-calendar-check", points_required=100, category="consistency"),
        Badge(name="Goal Achiever", description="Complete your first health goal", icon="fas fa-trophy", points_required=200, category="achievement"),
        Badge(name="Health Champion", description="Maintain perfect health score for a week", icon="fas fa-star", points_required=500, category="achievement"),
    ]

    for badge in badges:
        if not Badge.query.filter_by(name=badge.name).first():
            db.session.add(badge)
    db.session.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory('uploads', filename)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # Send email notification
        # Removed mailing functionality as per user request
        flash('Message received.', 'info')
        return render_template('contact.html', success=True)
    return render_template('contact.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        user_feedback = request.form.get('feedback')
        # Save feedback to a file or list (simple implementation)
        with open('feedback.txt', 'a') as f:
            f.write(user_feedback + '\n')
        flash('Feedback submitted successfully!', 'success')
        return render_template('feedback.html', success=True)
    return render_template('feedback.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard with full prediction history and real-time monitoring"""
    try:
        today = date.today()

        # Fetch all prediction history for current user
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()

        # Fetch today's lifestyle entry
        today_lifestyle = LifestyleEntry.query.filter_by(user_id=current_user.id, date=today).first()

        # Fetch recent vitals (last 7 days)
        recent_vitals = VitalSigns.query.filter_by(user_id=current_user.id).filter(
            VitalSigns.date >= today - timedelta(days=7)
        ).order_by(VitalSigns.date.desc()).all()

        # Fetch today's medication logs
        today_meds = MedicationLog.query.filter_by(user_id=current_user.id, date_taken=today).all()

        # Fetch active health goals
        active_goals = HealthGoals.query.filter_by(user_id=current_user.id, is_active=True).all()

        # Calculate health score (example logic: average of last 7 days heart rate normalized)
        if recent_vitals:
            heart_rates = [v.heart_rate for v in recent_vitals if v.heart_rate]
            if heart_rates:
                avg_hr = sum(heart_rates) / len(heart_rates)
                health_score = max(0, min(100, int(100 - (avg_hr - 60) * 1.5)))  # simplistic formula
            else:
                health_score = 80
        else:
            health_score = 80

        health_status = "Good" if health_score >= 75 else "Needs Attention"

        return render_template('dashboard_enhanced.html',
                               predictions=predictions,
                               today_lifestyle=today_lifestyle,
                               recent_vitals=recent_vitals,
                               today_meds=today_meds,
                               active_goals=active_goals,
                               health_score=health_score,
                               health_status=health_status)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return f"<h1>Dashboard Error</h1><pre>{error_msg}</pre>", 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/checkup')
@login_required
def checkup():
    return render_template('checkup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.form
        pregnancies = float(data['pregnancies'])
        glucose = float(data['glucose'])
        bloodpressure = float(data['bloodpressure'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        dpf = float(data['dpf'])
        age = float(data['age'])

        features = np.array([[pregnancies, glucose, bloodpressure, 0, insulin, bmi, dpf, age]])
        prediction = model.predict(features)
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

        # Determine diabetes type if diabetic
        diabetes_type = None
        if result == 'Diabetic':
            if insulin < 10 and age < 30:
                diabetes_type = 'Type 1'
            else:
                diabetes_type = 'Type 2'

        # Store prediction in database
        pred = Prediction(
            user_id=current_user.id,
            pregnancies=pregnancies,
            glucose=glucose,
            bloodpressure=bloodpressure,
            skinthickness=0,  # Set to 0 since removed from form
            insulin=insulin,
            bmi=bmi,
            dpf=dpf,
            age=age,
            result=result,
            diabetes_type=diabetes_type
        )
        db.session.add(pred)
        db.session.commit()

        prediction_display = result
        if diabetes_type:
            prediction_display += f' ({diabetes_type})'

        flash(f'Prediction: {prediction_display}.', 'info')
        return render_template('checkup.html',
                             prediction_text=f'Prediction: {prediction_display}',
                             pregnancies=pregnancies,
                             glucose=glucose,
                             bloodpressure=bloodpressure,
                             insulin=insulin,
                             bmi=bmi,
                             dpf=dpf,
                             age=age)
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return render_template('checkup.html', prediction_text=f'Error: {str(e)}')

# ===== REAL-TIME PATIENT MONITORING ROUTES =====

@app.route('/monitoring')
@login_required
def monitoring():
    """Main monitoring dashboard"""
    today = date.today()

    # Get today's lifestyle entry
    today_lifestyle = LifestyleEntry.query.filter_by(
        user_id=current_user.id,
        date=today
    ).first()

    # Get recent vital signs (last 7 days)
    recent_vitals = VitalSigns.query.filter_by(user_id=current_user.id).filter(
        VitalSigns.date >= today - timedelta(days=7)
    ).order_by(VitalSigns.date.desc()).limit(7).all()

    # Get today's medication logs
    today_meds = MedicationLog.query.filter_by(
        user_id=current_user.id,
        date_taken=today
    ).all()

    # Get active health goals
    active_goals = HealthGoals.query.filter_by(
        user_id=current_user.id,
        is_active=True
    ).all()

    return render_template('monitoring.html',
                         today_lifestyle=today_lifestyle,
                         recent_vitals=recent_vitals,
                         today_meds=today_meds,
                         active_goals=active_goals)

@app.route('/lifestyle/add', methods=['GET', 'POST'])
@login_required
def add_lifestyle():
    """Add daily lifestyle entry"""
    if request.method == 'POST':
        try:
            entry = LifestyleEntry(
                user_id=current_user.id,
                exercise_minutes=int(request.form.get('exercise_minutes', 0)),
                exercise_type=request.form.get('exercise_type'),
                diet_description=request.form.get('diet_description'),
                water_glasses=int(request.form.get('water_glasses', 0)),
                sleep_hours=float(request.form.get('sleep_hours', 0)),
                mood_rating=int(request.form.get('mood_rating', 5)),
                stress_level=int(request.form.get('stress_level', 5)),
                notes=request.form.get('notes')
            )
            db.session.add(entry)
            db.session.commit()
            flash('Lifestyle entry added successfully!', 'success')
            return redirect(url_for('monitoring'))
        except Exception as e:
            flash(f'Error adding lifestyle entry: {str(e)}', 'error')

    return render_template('add_lifestyle.html')

@app.route('/vitals/add', methods=['GET', 'POST'])
@login_required
def add_vitals():
    """Add vital signs"""
    if request.method == 'POST':
        try:
            vitals = VitalSigns(
                user_id=current_user.id,
                heart_rate=int(request.form.get('heart_rate')) if request.form.get('heart_rate') else None,
                systolic_bp=int(request.form.get('systolic_bp')) if request.form.get('systolic_bp') else None,
                diastolic_bp=int(request.form.get('diastolic_bp')) if request.form.get('diastolic_bp') else None,
                weight=float(request.form.get('weight')) if request.form.get('weight') else None,
                temperature=float(request.form.get('temperature')) if request.form.get('temperature') else None,
                oxygen_saturation=int(request.form.get('oxygen_saturation')) if request.form.get('oxygen_saturation') else None,
                notes=request.form.get('notes')
            )
            db.session.add(vitals)
            db.session.commit()

            # Award points for vitals entry
            award_points(current_user.id, 10, "Vital signs recorded")
            update_streak(current_user.id, "vitals")

            flash('Vital signs recorded successfully!', 'success')
            return redirect(url_for('monitoring'))
        except Exception as e:
            flash(f'Error recording vital signs: {str(e)}', 'error')

    return render_template('add_vitals.html')

@app.route('/medication/log', methods=['GET', 'POST'])
@login_required
def log_medication():
    """Log medication intake"""
    if request.method == 'POST':
        try:
            med_log = MedicationLog(
                user_id=current_user.id,
                medication_name=request.form.get('medication_name'),
                dosage=request.form.get('dosage'),
                frequency=request.form.get('frequency'),
                time_taken=datetime.strptime(request.form.get('time_taken'), '%H:%M').time() if request.form.get('time_taken') else None,
                taken=request.form.get('taken') == 'on',
                side_effects=request.form.get('side_effects'),
                notes=request.form.get('notes')
            )
            db.session.add(med_log)
            db.session.commit()
            flash('Medication logged successfully!', 'success')
            return redirect(url_for('monitoring'))
        except Exception as e:
            flash(f'Error logging medication: {str(e)}', 'error')

    return render_template('log_medication.html')

@app.route('/goals/set', methods=['GET', 'POST'])
@login_required
def set_goals():
    """Set health goals"""
    if request.method == 'POST':
        try:
            goal = HealthGoals(
                user_id=current_user.id,
                goal_type=request.form.get('goal_type'),
                target_value=float(request.form.get('target_value')),
                unit=request.form.get('unit'),
                target_date=datetime.strptime(request.form.get('target_date'), '%Y-%m-%d').date() if request.form.get('target_date') else None,
                notes=request.form.get('notes')
            )
            db.session.add(goal)
            db.session.commit()
            flash('Health goal set successfully!', 'success')
            return redirect(url_for('monitoring'))
        except Exception as e:
            flash(f'Error setting goal: {str(e)}', 'error')

    return render_template('set_goals.html')

@app.route('/api/health-data')
@login_required
def get_health_data():
    """API endpoint for real-time health data"""
    today = date.today()

    # Get recent data for charts
    lifestyle_data = LifestyleEntry.query.filter_by(user_id=current_user.id).filter(
        LifestyleEntry.date >= today - timedelta(days=30)
    ).order_by(LifestyleEntry.date).all()

    vitals_data = VitalSigns.query.filter_by(user_id=current_user.id).filter(
        VitalSigns.date >= today - timedelta(days=7)
    ).order_by(VitalSigns.date).all()

    # Format data for charts
    chart_data = {
        'lifestyle': {
            'dates': [entry.date.strftime('%Y-%m-%d') for entry in lifestyle_data],
            'exercise': [entry.exercise_minutes for entry in lifestyle_data],
            'water': [entry.water_glasses for entry in lifestyle_data],
            'sleep': [entry.sleep_hours for entry in lifestyle_data],
            'mood': [entry.mood_rating for entry in lifestyle_data]
        },
        'vitals': {
            'dates': [vital.date.strftime('%Y-%m-%d') for vital in vitals_data],
            'heart_rate': [vital.heart_rate for vital in vitals_data if vital.heart_rate],
            'blood_pressure': [f"{vital.systolic_bp}/{vital.diastolic_bp}" for vital in vitals_data if vital.systolic_bp and vital.diastolic_bp],
            'weight': [vital.weight for vital in vitals_data if vital.weight]
        }
    }

    return jsonify(chart_data)

# ===== REAL-TIME UPDATES WITH SERVER-SENT EVENTS =====

# SSE endpoint removed to replace with HTMX partial updates

@app.route('/monitoring/today-summary')
@login_required
def partial_today_summary():
    today = date.today()
    today_lifestyle = LifestyleEntry.query.filter_by(
        user_id=current_user.id,
        date=today
    ).first()
    return render_template('partials/today_summary.html', today_lifestyle=today_lifestyle)

@app.route('/monitoring/recent-vitals')
@login_required
def partial_recent_vitals():
    today = date.today()
    recent_vitals = VitalSigns.query.filter_by(user_id=current_user.id).filter(
        VitalSigns.date >= today - timedelta(days=7)
    ).order_by(VitalSigns.date.desc()).limit(7).all()
    return render_template('partials/recent_vitals.html', recent_vitals=recent_vitals)

@app.route('/monitoring/active-goals')
@login_required
def partial_active_goals():
    active_goals = HealthGoals.query.filter_by(
        user_id=current_user.id,
        is_active=True
    ).all()
    return render_template('partials/active_goals.html', active_goals=active_goals)

@app.route('/monitoring/today-meds')
@login_required
def partial_today_meds():
    today = date.today()
    today_meds = MedicationLog.query.filter_by(
        user_id=current_user.id,
        date_taken=today
    ).all()
    return render_template('partials/today_meds.html', today_meds=today_meds)

@app.route('/monitoring/health-score')
@login_required
def partial_health_score():
    # For demonstration, calculate a simple health score
    # This can be replaced with actual logic
    score = 85
    score_status = "Excellent"
    return render_template('partials/health_score.html', score=score, score_status=score_status)

# ===== PREDICTION GRAPHS ROUTES =====

@app.route('/predictions/graphs')
@login_required
def predictions_graphs():
    """Display prediction graphs and analytics"""
    return render_template('predictions_graphs.html')

@app.route('/api/predictions-data')
@login_required
def get_predictions_data():
    """API endpoint for prediction data used in graphs"""
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()

        # Format data for charts
        chart_data = []
        for pred in predictions:
            chart_data.append({
                'date': pred.timestamp.strftime('%Y-%m-%d'),
                'glucose': pred.glucose,
                'bloodpressure': pred.bloodpressure,
                'bmi': pred.bmi,
                'age': pred.age,
                'insulin': pred.insulin,
                'skinthickness': pred.skinthickness,
                'dpf': pred.dpf,
                'result': pred.result
            })

        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== GAMIFICATION ROUTES =====

@app.route('/gamification/badges')
@login_required
def badges():
    """Display user's earned badges"""
    user_badges = UserBadge.query.filter_by(user_id=current_user.id).all()
    user_points = UserPoints.query.filter_by(user_id=current_user.id).first()
    all_badges = Badge.query.all()

    return render_template('badges.html',
                         user_badges=user_badges,
                         user_points=user_points,
                         all_badges=all_badges)

@app.route('/gamification/leaderboard')
@login_required
def leaderboard():
    """Display user rankings"""
    # Get top 10 users by points
    top_users = UserPoints.query.join(User).order_by(UserPoints.total_points.desc()).limit(10).all()
    current_user_points = UserPoints.query.filter_by(user_id=current_user.id).first()

    return render_template('leaderboard.html',
                         top_users=top_users,
                         current_user_points=current_user_points)

# ===== DATA EXPORT ROUTES =====

@app.route('/export/report', methods=['GET', 'POST'])
@login_required
def export_report():
    """Export health data as PDF or CSV"""
    if request.method == 'POST':
        export_type = request.form.get('export_type')
        date_range = request.form.get('date_range', '30')  # days

        today = date.today()
        start_date = today - timedelta(days=int(date_range))

        # Gather data
        predictions = Prediction.query.filter_by(user_id=current_user.id).filter(
            Prediction.timestamp >= start_date
        ).order_by(Prediction.timestamp).all()

        lifestyle_entries = LifestyleEntry.query.filter_by(user_id=current_user.id).filter(
            LifestyleEntry.date >= start_date
        ).order_by(LifestyleEntry.date).all()

        vitals = VitalSigns.query.filter_by(user_id=current_user.id).filter(
            VitalSigns.date >= start_date
        ).order_by(VitalSigns.date).all()

        medications = MedicationLog.query.filter_by(user_id=current_user.id).filter(
            MedicationLog.date_taken >= start_date
        ).order_by(MedicationLog.date_taken).all()

        goals = HealthGoals.query.filter_by(user_id=current_user.id).all()

        if export_type == 'pdf':
            return generate_pdf_report(predictions, lifestyle_entries, vitals, medications, goals)
        elif export_type == 'csv':
            return generate_csv_report(predictions, lifestyle_entries, vitals, medications, goals)

    return render_template('export_report.html')

def generate_pdf_report(predictions, lifestyle_entries, vitals, medications, goals):
    """Generate PDF health report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("Health Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Summary
    summary = Paragraph(f"Report generated on {date.today()}", styles['Normal'])
    story.append(summary)
    story.append(Spacer(1, 12))

    # Predictions Table
    if predictions:
        pred_data = [['Date', 'Glucose', 'BP', 'BMI', 'Result']]
        for pred in predictions:
            pred_data.append([
                pred.timestamp.strftime('%Y-%m-%d'),
                str(pred.glucose),
                str(pred.bloodpressure),
                str(pred.bmi),
                pred.result
            ])
        pred_table = Table(pred_data)
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#f0f0f0'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#ffffff'),
            ('GRID', (0, 0), (-1, -1), 1, '#000000')
        ]))
        story.append(Paragraph("Diabetes Predictions", styles['Heading2']))
        story.append(pred_table)
        story.append(Spacer(1, 12))

    # Lifestyle Table
    if lifestyle_entries:
        life_data = [['Date', 'Exercise (min)', 'Water (glasses)', 'Sleep (hrs)', 'Mood']]
        for entry in lifestyle_entries:
            life_data.append([
                entry.date.strftime('%Y-%m-%d'),
                str(entry.exercise_minutes),
                str(entry.water_glasses),
                str(entry.sleep_hours),
                str(entry.mood_rating)
            ])
        life_table = Table(life_data)
        life_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#f0f0f0'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#ffffff'),
            ('GRID', (0, 0), (-1, -1), 1, '#000000')
        ]))
        story.append(Paragraph("Lifestyle Entries", styles['Heading2']))
        story.append(life_table)
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='health_report.pdf', mimetype='application/pdf')

def generate_csv_report(predictions, lifestyle_entries, vitals, medications, goals):
    """Generate CSV health report"""
    data = {
        'Predictions': pd.DataFrame([{
            'Date': p.timestamp.strftime('%Y-%m-%d'),
            'Glucose': p.glucose,
            'BloodPressure': p.bloodpressure,
            'BMI': p.bmi,
            'Result': p.result
        } for p in predictions]),
        'Lifestyle': pd.DataFrame([{
            'Date': l.date.strftime('%Y-%m-%d'),
            'Exercise_Minutes': l.exercise_minutes,
            'Water_Glasses': l.water_glasses,
            'Sleep_Hours': l.sleep_hours,
            'Mood_Rating': l.mood_rating
        } for l in lifestyle_entries]),
        'Vitals': pd.DataFrame([{
            'Date': v.date.strftime('%Y-%m-%d'),
            'Heart_Rate': v.heart_rate,
            'Systolic_BP': v.systolic_bp,
            'Diastolic_BP': v.diastolic_bp,
            'Weight': v.weight
        } for v in vitals]),
        'Medications': pd.DataFrame([{
            'Date': m.date_taken.strftime('%Y-%m-%d'),
            'Medication': m.medication_name,
            'Dosage': m.dosage,
            'Taken': m.taken
        } for m in medications]),
        'Goals': pd.DataFrame([{
            'Goal_Type': g.goal_type,
            'Target_Value': g.target_value,
            'Current_Value': g.current_value,
            'Progress_Percentage': g.progress_percentage
        } for g in goals])
    }

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='health_report.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ===== APPOINTMENT BOOKING ROUTES =====

@app.route('/doctors')
@login_required
def doctors():
    """Display list of available doctors"""
    doctors = Doctor.query.all()
    return render_template('doctor_list.html', doctors=doctors)

@app.route('/book')
@login_required
def book():
    """Standalone booking page - show all doctors with booking forms"""
    doctors = Doctor.query.all()
    # Calculate minimum date (tomorrow)
    min_date = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    return render_template('book.html', doctors=doctors, min_date=min_date)

@app.route('/appointments')
@login_required
def appointments():
    """Display user's appointments"""
    user_appointments = Appointment.query.filter_by(user_id=current_user.id).order_by(Appointment.appointment_date.desc()).all()
    return render_template('appointments.html', appointments=user_appointments)

@app.route('/appointments/book/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def book_appointment(doctor_id):
    """Book an appointment with a doctor"""
    doctor = Doctor.query.get_or_404(doctor_id)

    # Calculate minimum date (tomorrow)
    min_date = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')

    if request.method == 'POST':
        try:
            appointment_date = datetime.strptime(request.form.get('appointment_date'), '%Y-%m-%d').date()
            appointment_time = datetime.strptime(request.form.get('appointment_time'), '%H:%M').time()
            notes = request.form.get('notes')

            # Check if the slot is available (simple check - no overlapping appointments for the same doctor at the same time)
            existing_appointment = Appointment.query.filter_by(
                doctor_id=doctor_id,
                appointment_date=appointment_date,
                appointment_time=appointment_time
            ).first()

            if existing_appointment:
                flash('This time slot is already booked. Please choose a different time.', 'error')
                return render_template('book_appointment.html', doctor=doctor)

            appointment = Appointment(
                user_id=current_user.id,
                doctor_id=doctor_id,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                notes=notes
            )
            db.session.add(appointment)
            db.session.commit()

            # Award points for booking appointment
            award_points(current_user.id, 25, "Appointment booked")

            flash('Appointment booked successfully!', 'success')
            return redirect(url_for('appointments'))

        except Exception as e:
            flash(f'Error booking appointment: {str(e)}', 'error')

    return render_template('book_appointment.html', doctor=doctor)

@app.route('/appointments/cancel/<int:appointment_id>', methods=['POST'])
@login_required
def cancel_appointment(appointment_id):
    """Cancel an appointment"""
    appointment = Appointment.query.filter_by(id=appointment_id, user_id=current_user.id).first()

    if not appointment:
        flash('Appointment not found.', 'error')
        return redirect(url_for('appointments'))

    if appointment.status != 'Scheduled':
        flash('Cannot cancel a completed appointment.', 'error')
        return redirect(url_for('appointments'))

    appointment.status = 'Cancelled'
    db.session.commit()

    flash('Appointment cancelled successfully.', 'info')
    return redirect(url_for('appointments'))

def initialize_doctors():
    """Initialize sample doctors"""
    doctors = [
        Doctor(
            name="Dr. Sarah Johnson",
            specialty="Endocrinologist",
            contact_info="sarah.johnson@clinic.com",
            address="123 Health Street, Medical District",
            available_days="Mon-Fri",
            available_hours="9AM-5PM",
            experience_years=15,
            languages="English, Spanish",
            rating=4.9,
            bio="Dr. Sarah Johnson is a board-certified Endocrinologist with over 15 years of experience in treating diabetes and hormonal disorders. She specializes in Type 1 and Type 2 diabetes management.",
            consultation_fee=150.00
        ),
        Doctor(
            name="Dr. Michael Chen",
            specialty="General Physician",
            contact_info="michael.chen@clinic.com",
            address="456 Wellness Avenue, Downtown",
            available_days="Tue-Sat",
            available_hours="10AM-6PM",
            experience_years=10,
            languages="English, Mandarin",
            rating=4.8,
            bio="Dr. Michael Chen is a compassionate General Physician with a focus on preventive care and chronic disease management. He has been serving the community for over 10 years.",
            consultation_fee=100.00
        ),
        Doctor(
            name="Dr. Emily Rodriguez",
            specialty="Diabetologist",
            contact_info="emily.rodriguez@clinic.com",
            address="789 Diabetes Center, Uptown",
            available_days="Mon-Wed-Fri",
            available_hours="8AM-4PM",
            experience_years=12,
            languages="English, Spanish, Portuguese",
            rating=4.7,
            bio="Dr. Emily Rodriguez specializes exclusively in diabetes care and education. She is passionate about helping patients achieve optimal glucose control through personalized treatment plans.",
            consultation_fee=120.00
        ),
        Doctor(
            name="Dr. David Kim",
            specialty="Internal Medicine",
            contact_info="david.kim@clinic.com",
            address="321 Medical Plaza, Midtown",
            available_days="Mon-Thu",
            available_hours="9AM-5PM",
            experience_years=8,
            languages="English, Korean",
            rating=4.6,
            bio="Dr. David Kim is an Internal Medicine specialist with expertise in managing complex medical conditions, including diabetes-related complications.",
            consultation_fee=90.00
        ),
        Doctor(
            name="Dr. Priya Patel",
            specialty="Endocrinologist",
            contact_info="priya.patel@clinic.com",
            address="555 Sugar Care Center, Westside",
            available_days="Mon-Fri",
            available_hours="9AM-5PM",
            experience_years=18,
            languages="English, Hindi, Gujarati",
            rating=4.9,
            bio="Dr. Priya Patel is a renowned Endocrinologist with extensive experience in diabetes, thyroid disorders, and metabolic conditions. She has helped thousands of patients manage their diabetes effectively.",
            consultation_fee=175.00
        ),
        Doctor(
            name="Dr. James Wilson",
            specialty="Cardiologist",
            contact_info="james.wilson@clinic.com",
            address="888 Heart Health Building, Eastside",
            available_days="Mon-Wed-Fri",
            available_hours="8AM-4PM",
            experience_years=20,
            languages="English, French",
            rating=4.8,
            bio="Dr. James Wilson is a Cardiologist who specializes in the intersection of heart health and diabetes. He helps patients manage cardiovascular risks associated with diabetes.",
            consultation_fee=200.00
        ),
        Doctor(
            name="Dr. Lisa Thompson",
            specialty="Nutritionist",
            contact_info="lisa.thompson@clinic.com",
            address="222 Nutrition Plaza, Northside",
            available_days="Mon-Fri",
            available_hours="10AM-6PM",
            experience_years=7,
            languages="English, German",
            rating=4.7,
            bio="Dr. Lisa Thompson is a Registered Dietitian and Nutritionist who specializes in creating personalized meal plans for diabetes management and overall wellness.",
            consultation_fee=80.00
        ),
        Doctor(
            name="Dr. Robert Martinez",
            specialty="Diabetologist",
            contact_info="robert.martinez@clinic.com",
            address="777 Diabetes Excellence Center, Southside",
            available_days="Tue-Sat",
            available_hours="9AM-5PM",
            experience_years=14,
            languages="English, Spanish",
            rating=4.6,
            bio="Dr. Robert Martinez is a Diabetologist focused on innovative diabetes treatments and patient education. He believes in empowering patients with knowledge to manage their condition.",
            consultation_fee=130.00
        ),
        Doctor(
            name="Dr. Amanda Lee",
            specialty="General Physician",
            contact_info="amanda.lee@clinic.com",
            address="444 Family Care Clinic, Central",
            available_days="Mon-Fri",
            available_hours="8AM-4PM",
            experience_years=6,
            languages="English, Cantonese",
            rating=4.5,
            bio="Dr. Amanda Lee is a dedicated Family Medicine physician with a special interest in diabetes prevention and early intervention. She provides comprehensive primary care services.",
            consultation_fee=75.00
        ),
        Doctor(
            name="Dr. Christopher Brown",
            specialty="Internal Medicine",
            contact_info="chris.brown@clinic.com",
            address="999 Medical Tower, Downtown",
            available_days="Mon-Thu",
            available_hours="9AM-5PM",
            experience_years=11,
            languages="English",
            rating=4.7,
            bio="Dr. Christopher Brown is an Internal Medicine specialist with expertise in adult diabetes management and preventive healthcare. He takes a holistic approach to patient care.",
            consultation_fee=95.00
        )
    ]

    for doctor in doctors:
        if not Doctor.query.filter_by(name=doctor.name).first():
            db.session.add(doctor)
    db.session.commit()

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    with app.app_context():
        db.create_all()
        initialize_gamification()
        initialize_doctors()
    app.run(debug=True)
