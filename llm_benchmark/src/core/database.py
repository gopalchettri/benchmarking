# Database connections and session management
# - MongoDB: Stores raw JSON outputs from LLM (pymongo/motor)
# - MySQL: Stores structured metrics, test executions (pymysql)
# - Neo4j: Graphs control hierarchies (neo4j driver)
# - Functions: init_databases(), get_mongo_client(), get_mysql_conn()
"""
Manages connections and CRUD for MongoDB, MySQL, Neo4j.
"""


# src/core/database.py
import json
import os
from pymongo import MongoClient
import pymysql
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Load config from JSON file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/database.json")
with open(CONFIG_PATH) as f:
    db_cfg = json.load(f)


# --- MongoDB Section ---

#def get_mongo_client():
    cfg = db_cfg["mongodb"]
    #uri = f"mongodb://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/?ssl=true&tls="
    #uri += str(cfg.get("tls", True)).lower()
    #return MongoClient(uri, tls=cfg.get("tls", True))

def get_mongo_client():
    cfg = db_cfg["mongodb"]
    connection_string = cfg["connection_string"]
    return MongoClient(connection_string)

def mongo_insert(collection, doc):
    """Insert doc in given MongoDB collection."""
    client = get_mongo_client()
    db = client[db_cfg["mongodb"]["db"]]
    result = db[collection].insert_one(doc)
    client.close()
    return result.inserted_id

def mongo_find(collection, query):
    """Find docs in given MongoDB collection."""
    client = get_mongo_client()
    db = client[db_cfg["mongodb"]["db"]]
    results = list(db[collection].find(query))
    client.close()
    return results

def mongo_update(collection, query, update):
    """Update docs in given MongoDB collection."""
    client = get_mongo_client()
    db = client[db_cfg["mongodb"]["db"]]
    result = db[collection].update_many(query, {'$set': update})
    client.close()
    return result.modified_count

def mongo_delete(collection, query):
    """Delete docs from MongoDB collection."""
    client = get_mongo_client()
    db = client[db_cfg["mongodb"]["db"]]
    result = db[collection].delete_many(query)
    client.close()
    return result.deleted_count

# Save raw output specific function
def mongo_save_raw_output(model_name, use_case, test_case_id, params, prompt, output):
    data = {
        "model": model_name,
        "use_case": use_case,
        "test_case_id": test_case_id,
        "params": params,
        "prompt": prompt,
        "output": output,
        "status": "completed"
    }
    inserted_id = mongo_insert("criteria_raw", data)
    return inserted_id


# --- MongoDB Section Ends ---

# --- MySQL Section ---
def get_mysql_conn():
    cfg = db_cfg["mysql"]
    ssl_params = {'ssl': {}}
    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["username"],
        password=cfg["password"],
        database=cfg["db"],
        cursorclass=pymysql.cursors.DictCursor,
        ssl=ssl_params if cfg.get("ssl", False) else None
    )
    return conn

def mysql_insert(query, params):
    """Insert into MySQL."""
    conn = get_mysql_conn()
    with conn.cursor() as cursor:
        cursor.execute(query, params)
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

def mysql_fetch(query, params=None):
    """Fetch all rows from MySQL."""
    conn = get_mysql_conn()
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        results = cursor.fetchall()
    conn.close()
    return results

def mysql_update(query, params):
    """Update MySQL rows."""
    conn = get_mysql_conn()
    with conn.cursor() as cursor:
        cursor.execute(query, params)
    conn.commit()
    conn.close()

def mysql_delete(query, params):
    """Delete MySQL rows."""
    conn = get_mysql_conn()
    with conn.cursor() as cursor:
        cursor.execute(query, params)
    conn.commit()
    conn.close()

# Save metrics specific function
def mysql_save_metrics(execution_id, model_name, use_case, test_case_id, metrics):
    sql_query = """INSERT INTO evaluation_metrics (
        execution_id, model_name, use_case, test_case_id,
        relevance_score, specificity_score
    ) VALUES (%s, %s, %s, %s, %s, %s)"""
    params = (
        execution_id, model_name, use_case, test_case_id,
        metrics["relevance"], metrics["specificity"]
    )
    mysql_insert(sql_query, params)


# --- MySQL Section Ends ---

# --- Neo4j Section ---

def get_neo4j_conn():
    cfg = db_cfg["neo4j"]
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["username"], cfg["password"]))
    return driver

def neo4j_create(query, params=None):
    """Run Neo4j CREATE query."""
    driver = get_neo4j_conn()
    with driver.session() as session:
        result = session.run(query, params or {})
    driver.close()
    return result.data()

def neo4j_match(query, params=None):
    """Run Neo4j MATCH query."""
    driver = get_neo4j_conn()
    with driver.session() as session:
        result = session.run(query, params or {})
        data = [dict(record) for record in result]
    driver.close()
    return data

def neo4j_update(query, params=None):
    """Run Neo4j SET query."""
    driver = get_neo4j_conn()
    with driver.session() as session:
        result = session.run(query, params or {})
    driver.close()
    return result.data()

def neo4j_delete(query, params=None):
    """Run Neo4j DELETE query."""
    driver = get_neo4j_conn()
    with driver.session() as session:
        result = session.run(query, params or {})
    driver.close()
    return result.data()


# --- Neo4j Section Ends ---


# --- Initialization Function ---

def init_databases():
    """
    Creates initial tables/collections (stubs here, replace with schema SQL/Cypher for prod).
    """
    # Example: Create MySQL tables
    conn = get_mysql_conn()
    with conn.cursor() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS test_executions (
                        execution_id VARCHAR(36) PRIMARY KEY,
                        test_case_id VARCHAR(20),
                        model_name VARCHAR(50),
                        execution_timestamp DATETIME,
                        total_execution_time_ms INT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS model_parameters (
                        param_id INT AUTO_INCREMENT PRIMARY KEY,
                        execution_id VARCHAR(36),
                        temperature FLOAT,
                        top_p FLOAT,
                        top_k INT,
                        min_p FLOAT,
                        repetition_penalty FLOAT,
                        frequency_penalty FLOAT,
                        presence_penalty FLOAT,
                        context_window INT,
                        max_token INT
                    )""")
        c.execute("""CREATE TABLE IF NOT EXISTS evaluation_metrics (
                        metric_id INT AUTO_INCREMENT PRIMARY KEY,
                        execution_id VARCHAR(36),
                        relevance_score FLOAT,
                        specificity_score FLOAT,
                        policy_level_focus FLOAT,
                        completeness_score FLOAT,
                        clarity_score FLOAT,
                        measurability_score FLOAT,
                        compliance_alignment FLOAT,
                        criteria_count INT,
                        token_efficiency FLOAT
                    )""")
    conn.commit()
    conn.close()
    
    # MongoDB collection creation is dynamic, no actions needed
    # Example: Neo4j graph setup (stub)
    driver = get_neo4j_conn()
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS ON (t:TestCase) ASSERT t.control_id IS UNIQUE")
    driver.close()
    print("Databases initialized.")