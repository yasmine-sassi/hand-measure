import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    """Create and return a MySQL database connection"""
    connection = pymysql.connect(
        host=os.getenv('DB_HOST', '127.0.0.1'),
        port=int(os.getenv('DB_PORT', 3306)),
        user=os.getenv('DB_USERNAME', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_DATABASE', 'dashboard'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

def execute_query(query, params=None):
    """Execute a query and return results"""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
        connection.commit()
        return result
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        connection.close()

def execute_insert(query, params=None):
    """Execute an insert query and return the last inserted id"""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            last_id = cursor.lastrowid
        connection.commit()
        return last_id
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        connection.close()
