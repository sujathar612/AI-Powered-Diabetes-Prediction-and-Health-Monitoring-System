from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    lifestyle_entries = db.relationship('LifestyleEntry', backref='user', lazy=True)
    vital_signs = db.relationship('VitalSigns', backref='user', lazy=True)
    medication_logs = db.relationship('MedicationLog', backref='user', lazy=True)
    health_goals = db.relationship('HealthGoals', backref='user', lazy=True)
    user_badges = db.relationship('UserBadge', backref='user', lazy=True)
    user_points = db.relationship('UserPoints', backref='user', lazy=True)
    streaks = db.relationship('Streak', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pregnancies = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    bloodpressure = db.Column(db.Float, nullable=False)
    skinthickness = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    dpf = db.Column(db.Float, nullable=False)
    age = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    diabetes_type = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class LifestyleEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    exercise_minutes = db.Column(db.Integer, default=0)
    exercise_type = db.Column(db.String(100))
    diet_description = db.Column(db.Text)
    water_glasses = db.Column(db.Integer, default=0)
    sleep_hours = db.Column(db.Float, default=0)
    mood_rating = db.Column(db.Integer)  # 1-10 scale
    stress_level = db.Column(db.Integer)  # 1-10 scale
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class VitalSigns(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    heart_rate = db.Column(db.Integer)
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    weight = db.Column(db.Float)
    temperature = db.Column(db.Float)
    oxygen_saturation = db.Column(db.Integer)
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class MedicationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    medication_name = db.Column(db.String(200), nullable=False)
    dosage = db.Column(db.String(50))
    frequency = db.Column(db.String(100))  # e.g., "twice daily", "as needed"
    time_taken = db.Column(db.Time)
    date_taken = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    taken = db.Column(db.Boolean, default=False)
    side_effects = db.Column(db.Text)
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class HealthGoals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    goal_type = db.Column(db.String(50), nullable=False)  # weight, exercise, water, etc.
    target_value = db.Column(db.Float, nullable=False)
    current_value = db.Column(db.Float, default=0)
    unit = db.Column(db.String(20))  # kg, minutes, glasses, etc.
    start_date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    target_date = db.Column(db.Date)
    is_active = db.Column(db.Boolean, default=True)
    progress_percentage = db.Column(db.Float, default=0)
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    icon = db.Column(db.String(100))  # e.g., 'fas fa-trophy'
    points_required = db.Column(db.Integer, default=0)
    category = db.Column(db.String(50))  # e.g., 'consistency', 'achievement'
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class UserBadge(db.Model):
    __tablename__ = 'user_badge'
    
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    badge_id = db.Column(db.Integer(), db.ForeignKey('badge.id', ondelete='CASCADE'), nullable=False)
    earned_date = db.Column(db.DateTime(), default=db.func.current_timestamp())

class UserPoints(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    total_points = db.Column(db.Integer, default=0)
    current_streak = db.Column(db.Integer, default=0)
    longest_streak = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=db.func.current_timestamp())

class Streak(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    streak_type = db.Column(db.String(50), nullable=False)  # e.g., 'medication', 'exercise', 'lifestyle'
    current_count = db.Column(db.Integer, default=0)
    longest_count = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=db.func.current_timestamp())

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    specialty = db.Column(db.String(100), nullable=False)  # e.g., 'Endocrinologist', 'General Physician'
    contact_info = db.Column(db.String(200))  # phone or email
    address = db.Column(db.Text)
    available_days = db.Column(db.String(100))  # e.g., 'Mon-Fri'
    available_hours = db.Column(db.String(100))  # e.g., '9AM-5PM'
    experience_years = db.Column(db.Integer, default=0)  # Years of experience
    languages = db.Column(db.String(200))  # e.g., 'English, Hindi, Spanish'
    rating = db.Column(db.Float, default=4.5)  # Doctor rating
    bio = db.Column(db.Text)  # Doctor biography
    consultation_fee = db.Column(db.Float, default=50.0)  # Consultation fee
    appointments = db.relationship('Appointment', backref='doctor', lazy=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    appointment_date = db.Column(db.Date, nullable=False)
    appointment_time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(50), default='Scheduled')  # Scheduled, Completed, Cancelled
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
