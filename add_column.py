"""
Script to add missing columns to the doctor table in the existing database.
Run this script once to update the database schema.
"""
import sys
import os

# Add the parent directory to the path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from models import db

def add_missing_columns():
    """Add missing columns to the doctor table"""
    with app.app_context():
        # Get the existing table info
        inspector = db.inspect(db.engine)
        existing_columns = [col['name'] for col in inspector.get_columns('doctor')]
        
        print(f"Existing columns: {existing_columns}")
        
        # Define the columns to add (based on the Doctor model)
        columns_to_add = {
            'experience_years': 'INTEGER DEFAULT 0',
            'languages': 'VARCHAR(200)',
            'rating': 'FLOAT DEFAULT 4.5',
            'bio': 'TEXT',
            'consultation_fee': 'FLOAT DEFAULT 50.0',
            'timestamp': 'TIMESTAMP'
        }
        
        # Add each missing column
        with db.engine.connect() as conn:
            for column_name, column_def in columns_to_add.items():
                if column_name not in existing_columns:
                    try:
                        sql = f"ALTER TABLE doctor ADD COLUMN {column_name} {column_def}"
                        conn.execute(db.text(sql))
                        conn.commit()
                        print(f"Added column: {column_name}")
                    except Exception as e:
                        print(f"Error adding column {column_name}: {e}")
                else:
                    print(f"Column already exists: {column_name}")
        
        print("Database update complete!")

if __name__ == '__main__':
    add_missing_columns()
