import mysql.connector
from mysql.connector import Error

def view_users():
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',  # Default XAMPP user
            password='',  # Default XAMPP password (empty)
            database='diabetes_app'
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Query to get all users
            cursor.execute("SELECT id, username, email FROM user")

            users = cursor.fetchall()

            print("=== USER REGISTRATION ENTRIES ===")
            print(f"Total users: {len(users)}")
            print("-" * 60)

            if users:
                print("<10")
                print("-" * 60)
                for user in users:
                    user_id, username, email = user
                    print("<10")
            else:
                print("No user entries found.")

            cursor.close()
            connection.close()

    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        print("Make sure XAMPP MySQL is running and the database 'diabetes_app' exists.")

if __name__ == "__main__":
    view_users()
