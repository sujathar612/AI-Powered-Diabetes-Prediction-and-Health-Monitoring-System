from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, Response
from flask_login import LoginManager, login_user, login_required, logout_user, current_user

from config import Config
from models import db, User, Prediction, LifestyleEntry, VitalSigns, MedicationLog, HealthGoals, Doctor, Appointment
import joblib
import numpy as np
from datetime import datetime, date, timedelta
from fpdf import FPDF
import json
import time

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

        return render_template('dashboard.html',
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
        return render_template('checkup.html', prediction_text=f'Prediction: {prediction_display}')
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

    return jsonify(chart_data) # pyright: ignore[reportUndefinedVariable]

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
                'result': pred.result,
                'diabetes_type': pred.diabetes_type
            })

        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

            # Check if the slot is available
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

            flash('Appointment booked successfully!', 'success')
            return redirect(url_for('appointments'))

        except Exception as e:
            flash(f'Error booking appointment: {str(e)}', 'error')

    return render_template('book_appointment.html', doctor=doctor, min_date=min_date)

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

# ===== EXPORT REPORT ROUTES =====

@app.route('/export/report', methods=['GET', 'POST'])
@login_required
def export_report():
    """Export health report as PDF or CSV"""
    if request.method == 'POST':
        export_type = request.form.get('export_type')
        date_range = int(request.form.get('date_range', 30))
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=date_range)
        
        # Fetch user data within date range
        predictions = Prediction.query.filter(
            Prediction.user_id == current_user.id,
            Prediction.timestamp >= cutoff_date
        ).order_by(Prediction.timestamp.desc()).all()
        
        lifestyle_entries = LifestyleEntry.query.filter(
            LifestyleEntry.user_id == current_user.id,
            LifestyleEntry.date >= cutoff_date.date()
        ).order_by(LifestyleEntry.date.desc()).all()
        
        vitals = VitalSigns.query.filter(
            VitalSigns.user_id == current_user.id,
            VitalSigns.date >= cutoff_date.date()
        ).order_by(VitalSigns.date.desc()).all()
        
        medications = MedicationLog.query.filter(
            MedicationLog.user_id == current_user.id,
            MedicationLog.date_taken >= cutoff_date.date()
        ).order_by(MedicationLog.date_taken.desc()).all()
        
        goals = HealthGoals.query.filter(
            HealthGoals.user_id == current_user.id
        ).order_by(HealthGoals.timestamp.desc()).all()
        
        if export_type == 'csv':
            # Generate CSV
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write predictions
            writer.writerow(['=== Diabetes Predictions ==='])
            writer.writerow(['Date', 'Glucose', 'Blood Pressure', 'BMI', 'Insulin', 'Age', 'Result'])
            for pred in predictions:
                writer.writerow([
                    pred.timestamp.strftime('%Y-%m-%d'),
                    pred.glucose,
                    pred.bloodpressure,
                    pred.bmi,
                    pred.insulin,
                    pred.age,
                    pred.result
                ])
            
            # Write lifestyle
            writer.writerow([])
            writer.writerow(['=== Lifestyle Entries ==='])
            writer.writerow(['Date', 'Exercise (min)', 'Water (glasses)', 'Sleep (hours)', 'Mood', 'Stress'])
            for entry in lifestyle_entries:
                writer.writerow([
                    entry.date.strftime('%Y-%m-%d'),
                    entry.exercise_minutes,
                    entry.water_glasses,
                    entry.sleep_hours,
                    entry.mood_rating,
                    entry.stress_level
                ])
            
            # Write vitals
            writer.writerow([])
            writer.writerow(['=== Vital Signs ==='])
            writer.writerow(['Date', 'Heart Rate', 'Blood Pressure', 'Weight', 'Temperature', 'Oxygen'])
            for vital in vitals:
                writer.writerow([
                    vital.date.strftime('%Y-%m-%d'),
                    vital.heart_rate,
                    f"{vital.systolic_bp}/{vital.diastolic_bp}" if vital.systolic_bp else '',
                    vital.weight,
                    vital.temperature,
                    vital.oxygen_saturation
                ])
            
            # Write medications
            writer.writerow([])
            writer.writerow(['=== Medication Logs ==='])
            writer.writerow(['Date', 'Medication', 'Dosage', 'Frequency', 'Taken'])
            for med in medications:
                writer.writerow([
                    med.date_taken.strftime('%Y-%m-%d'),
                    med.medication_name,
                    med.dosage,
                    med.frequency,
                    'Yes' if med.taken else 'No'
                ])
            
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename=health_report_{datetime.now().strftime('%Y%m%d')}.csv"}
            )
        
        else:
            # Generate PDF using FPDF
            class PDF(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 15)
                    self.cell(0, 10, 'Diabetes Prediction Health Report', 0, 1, 'C')
                    self.set_font('Arial', 'I', 10)
                    self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                    self.ln(5)

                def section_title(self, title):
                    self.set_font('Arial', 'B', 12)
                    self.set_fill_color(200, 220, 255)
                    self.cell(0, 10, title, 0, 1, 'L', 1)
                    self.ln(2)

                def add_table_header(self, headers):
                    self.set_font('Arial', 'B', 9)
                    for header in headers:
                        self.cell(27, 7, header, 1, 0, 'C')
                    self.ln()

                def add_table_row(self, values):
                    self.set_font('Arial', '', 8)
                    for value in values:
                        self.cell(27, 6, str(value), 1, 0, 'C')
                    self.ln()

            pdf = PDF()
            pdf.add_page()
            
            # User Information
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, f'User: {current_user.username} ({current_user.email})', 0, 1)
            pdf.cell(0, 10, f'Report Period: Last {date_range} days', 0, 1)
            pdf.ln(5)

            # Diabetes Predictions Section
            if predictions:
                pdf.section_title('Diabetes Predictions')
                pdf.add_table_header(['Date', 'Glucose', 'BP', 'BMI', 'Insulin', 'Age', 'Result'])
                for pred in predictions[:20]:
                    pdf.add_table_row([
                        pred.timestamp.strftime('%Y-%m-%d'),
                        f'{pred.glucose:.0f}',
                        f'{pred.bloodpressure:.0f}',
                        f'{pred.bmi:.1f}',
                        f'{pred.insulin:.0f}',
                        f'{pred.age:.0f}',
                        pred.result[:8]
                    ])
                pdf.ln(5)
            else:
                pdf.section_title('Diabetes Predictions')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'No prediction records found for this period.', 0, 1)
                pdf.ln(5)

            # Lifestyle Entries Section
            if lifestyle_entries:
                pdf.add_page()
                pdf.section_title('Lifestyle Entries')
                pdf.add_table_header(['Date', 'Exercise', 'Water', 'Sleep', 'Mood', 'Stress'])
                for entry in lifestyle_entries[:20]:
                    pdf.add_table_row([
                        entry.date.strftime('%Y-%m-%d'),
                        f'{entry.exercise_minutes}',
                        f'{entry.water_glasses}',
                        f'{entry.sleep_hours:.1f}',
                        f'{entry.mood_rating}',
                        f'{entry.stress_level}'
                    ])
                pdf.ln(5)
            else:
                pdf.add_page()
                pdf.section_title('Lifestyle Entries')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'No lifestyle entries found for this period.', 0, 1)
                pdf.ln(5)

            # Vital Signs Section
            if vitals:
                pdf.add_page()
                pdf.section_title('Vital Signs')
                pdf.add_table_header(['Date', 'Heart Rate', 'BP', 'Weight', 'Temp', 'O2'])
                for vital in vitals[:20]:
                    bp = f'{vital.systolic_bp}/{vital.diastolic_bp}' if vital.systolic_bp else '-'
                    pdf.add_table_row([
                        vital.date.strftime('%Y-%m-%d'),
                        f'{vital.heart_rate}' if vital.heart_rate else '-',
                        bp,
                        f'{vital.weight:.1f}' if vital.weight else '-',
                        f'{vital.temperature:.1f}' if vital.temperature else '-',
                        f'{vital.oxygen_saturation}' if vital.oxygen_saturation else '-'
                    ])
                pdf.ln(5)
            else:
                pdf.add_page()
                pdf.section_title('Vital Signs')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'No vital signs records found for this period.', 0, 1)
                pdf.ln(5)

            # Medication Logs Section
            if medications:
                pdf.add_page()
                pdf.section_title('Medication Logs')
                pdf.add_table_header(['Date', 'Medication', 'Dosage', 'Frequency', 'Taken'])
                for med in medications[:20]:
                    pdf.add_table_row([
                        med.date_taken.strftime('%Y-%m-%d'),
                        med.medication_name[:10],
                        med.dosage[:6] if med.dosage else '-',
                        med.frequency[:8] if med.frequency else '-',
                        'Yes' if med.taken else 'No'
                    ])
                pdf.ln(5)
            else:
                pdf.add_page()
                pdf.section_title('Medication Logs')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'No medication logs found for this period.', 0, 1)
                pdf.ln(5)

            # Health Goals Section
            if goals:
                pdf.add_page()
                pdf.section_title('Health Goals')
                pdf.add_table_header(['Type', 'Target', 'Current', 'Unit', 'Progress', 'Status'])
                for goal in goals[:20]:
                    pdf.add_table_row([
                        goal.goal_type[:10],
                        f'{goal.target_value:.1f}',
                        f'{goal.current_value:.1f}',
                        goal.unit[:5] if goal.unit else '-',
                        f'{goal.progress_percentage:.0f}%',
                        'Active' if goal.is_active else 'Inactive'
                    ])
            else:
                pdf.add_page()
                pdf.section_title('Health Goals')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'No health goals found.', 0, 1)

            # Output PDF
            pdf_output = pdf.output(dest='S').encode('latin-1')
            
            return Response(
                pdf_output,
                mimetype='application/pdf',
                headers={'Content-disposition': f'attachment; filename=health_report_{datetime.now().strftime("%Y%m%d")}.pdf'}
            )
    
    return render_template('export_report.html')


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
            bio="Dr. Sarah Johnson is a board-certified Endocrinologist with over 15 years of experience in treating diabetes and hormonal disorders.",
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
            bio="Dr. Michael Chen is a compassionate General Physician with a focus on preventive care and chronic disease management.",
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
            bio="Dr. Emily Rodriguez specializes exclusively in diabetes care and education.",
            consultation_fee=120.00
        ),
        Doctor(
            name="Dr. James Williams",
            specialty="Internal Medicine",
            contact_info="james.williams@clinic.com",
            address="321 Medical Plaza, Midtown",
            available_days="Mon-Fri",
            available_hours="9AM-5PM",
            experience_years=18,
            languages="English, French",
            rating=4.9,
            bio="Dr. James Williams is a senior Internal Medicine specialist with expertise in complex medical conditions.",
            consultation_fee=140.00
        ),
        Doctor(
            name="Dr. Anita Patel",
            specialty="Cardiologist",
            contact_info="anita.patel@clinic.com",
            address="555 Heart Care Center, Westside",
            available_days="Mon-Thu",
            available_hours="10AM-6PM",
            experience_years=14,
            languages="English, Hindi, Gujarati",
            rating=4.8,
            bio="Dr. Anita Patel specializes in cardiovascular health for diabetic patients.",
            consultation_fee=175.00
        ),
        Doctor(
            name="Dr. Robert Kim",
            specialty="Nutritionist",
            contact_info="robert.kim@clinic.com",
            address="888 Wellness Blvd, Eastside",
            available_days="Mon-Sat",
            available_hours="8AM-3PM",
            experience_years=8,
            languages="English, Korean",
            rating=4.6,
            bio="Dr. Robert Kim is a certified nutritionist specializing in diabetic meal planning and lifestyle management.",
            consultation_fee=90.00
        ),
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
        initialize_doctors()
    app.run(debug=True)
