from fastapi import APIRouter, Depends, HTTPException
from src.api.dependencies import get_mysql, get_mongo, get_neo4j

router = APIRouter()

# MongoDB CRUD Test
@router.post("/mongo-insert")
def mongo_insert_test(document: dict, mongo=Depends(get_mongo)):
    db = mongo["your-db"]
    result = db["test_collection"].insert_one(document)
    return {"inserted_id": str(result.inserted_id)}

@router.get("/mongo-find")
def mongo_find_test(mongo=Depends(get_mongo)):
    db = mongo["your-db"]
    docs = list(db["test_collection"].find({}))
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return {"documents": docs}

@router.put("/mongo-update")
def mongo_update_test(query: dict, update: dict, mongo=Depends(get_mongo)):
    db = mongo["your-db"]
    result = db["test_collection"].update_many(query, {"$set": update})
    return {"modified_count": result.modified_count}

@router.delete("/mongo-delete")
def mongo_delete_test(query: dict, mongo=Depends(get_mongo)):
    db = mongo["your-db"]
    result = db["test_collection"].delete_many(query)
    return {"deleted_count": result.deleted_count}

# MySQL CRUD Test
@router.post("/mysql-insert")
def mysql_insert_test(row: dict, db=Depends(get_mysql)):
    try:
        with db.cursor() as cursor:
            cursor.execute(
                "INSERT INTO test_table (col1, col2) VALUES (%s, %s)",
                (row["col1"], row["col2"])
            )
        db.commit()
        return {"status": "inserted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mysql-find")
def mysql_find_test(db=Depends(get_mysql)):
    with db.cursor() as cursor:
        cursor.execute("SELECT * FROM test_table")
        rows = cursor.fetchall()
    return {"rows": rows}

@router.put("/mysql-update")
def mysql_update_test(row: dict, db=Depends(get_mysql)):
    with db.cursor() as cursor:
        cursor.execute(
            "UPDATE test_table SET col2=%s WHERE col1=%s",
            (row["col2"], row["col1"])
        )
    db.commit()
    return {"status": "updated"}

@router.delete("/mysql-delete")
def mysql_delete_test(row: dict, db=Depends(get_mysql)):
    with db.cursor() as cursor:
        cursor.execute("DELETE FROM test_table WHERE col1=%s", (row["col1"],))
    db.commit()
    return {"status": "deleted"}

# Neo4j CRUD Test
@router.post("/neo4j-create")
def neo4j_create_test(props: dict, driver=Depends(get_neo4j)):
    with driver.session() as session:
        result = session.run(
            "CREATE (n:TestNode {props}) RETURN n", props=props
        )
        nodes = [record["n"] for record in result]
    return {"created_nodes": nodes}

@router.get("/neo4j-find")
def neo4j_find_test(driver=Depends(get_neo4j)):
    with driver.session() as session:
        result = session.run(
            "MATCH (n:TestNode) RETURN n LIMIT 10"
        )
        nodes = [dict(record["n"]) for record in result]
    return {"nodes": nodes}

@router.delete("/neo4j-delete")
def neo4j_delete_test(driver=Depends(get_neo4j)):
    with driver.session() as session:
        result = session.run(
            "MATCH (n:TestNode) DELETE n"
        )
    return {"status": "deleted"}
