"""
Check the doctor table schema
"""
from app import app
from models import db

with app.app_context():
    inspector = db.inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('doctor')]
    print('Current doctor table columns:')
    for col in columns:
        print(f'  - {col}')
