# Shared API dependencies
# - Database session providers (MongoDB, MySQL)
# - Authentication/authorization (if needed)
# - Functions: get_db_session(), verify_api_key()


# src/api/dependencies.py

from fastapi import Depends, HTTPException, Header
from src.core.database import get_mysql_conn, get_mongo_client, get_neo4j_conn

# Dependency for MySQL connection
def get_mysql():
    """
    Yields a MySQL connection for the duration of the request.
    Use as db=Depends(get_mysql).
    """
    conn = get_mysql_conn()
    try:
        yield conn
    finally:
        conn.close()

# Dependency for MongoDB client
def get_mongo():
    """
    Yields a MongoDB client for the duration of the request.
    Use as mongo=Depends(get_mongo).
    """
    client = get_mongo_client()
    try:
        yield client
    finally:
        client.close()

# Dependency for Neo4j driver
def get_neo4j():
    """
    Yields a Neo4j driver for the duration of the request.
    Use as driver=Depends(get_neo4j).
    """
    driver = get_neo4j_conn()
    try:
        yield driver
    finally:
        driver.close()


# Optional: API Key Authentication Dependency Example
def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    # Replace 'my-super-secret-key' with your secret or read from a config/env
    expected_key = "my-super-secret-key"
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
