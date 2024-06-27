"""API to generate tags for each token in a given EHR"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
import json
from predict import get_ner_predictions, get_re_predictions
from neo4j import GraphDatabase
from utils import find_specific_word,clean_string,display_ehr, get_long_relation_table, display_knowledge_graph, get_relation_table


class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def create_node_and_relationship(self, entity1_text, entity2_text, relation_type):
        with self._driver.session() as session:
            # Create nodes for entity1 and entity2 if they don't exist, and establish the relationship
            session.write_transaction(self._create_nodes_and_relationship, entity1_text, entity2_text, relation_type)

    @staticmethod
    def _create_nodes_and_relationship(tx, entity1_text, entity2_text, relation_type):
        # Cypher query with parameters
        query = (
            "MERGE (entity1:Entity {name: $entity1_text}) "
            "MERGE (entity2:Entity {name: $entity2_text}) "
            "MERGE (entity1)-[r:RELATION {type: $relation_type}]->(entity2)"
        )
        # Pass parameters to the query
        tx.run(query, entity1_text=entity1_text, entity2_text=entity2_text,relation_type = relation_type)


class QuestionRequest(BaseModel):
    question: str

class NERTask(BaseModel):
    ehr_text: str
    model_choice: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("static\input2.txt") as f:
    SAMPLE_EHR = f.read()


@app.post("/")
def get_ehr_predictions(ner_input: NERTask):
    """Request EHR text data and the model choice for NER Task"""

    ner_predictions = get_ner_predictions(
        ehr_record=ner_input.ehr_text,
        model_name=ner_input.model_choice)
    
    print(ner_predictions.entities)

    re_predictions = get_re_predictions(ner_predictions)
    relation_table = get_long_relation_table(re_predictions.relations)

    ner_entity_mapping = {}
    for entity in ner_predictions.entities:
        entity_id = entity.ann_id
        entity_text = clean_string(entity.ann_text)
        ner_entity_mapping[entity_id] = entity_text

    idx = 1
    rel_preds = []
    for relation_info in re_predictions.relations:
        entity2_id = relation_info.arg2.ann_id
        if entity2_id:
            entity2_text = ner_entity_mapping.get(entity2_id)
            if entity2_text:
                relation_type = relation_info.name
                # Create a relationship based on entity2_text, relation_type, etc.
                # You can adjust this part based on how you want to structure your knowledge graph
                relationship = {
                    "Entity2 text": entity2_text,
                    "Relation type": relation_type,
                    "Entity1 text": relation_info.arg1.ann_text,  # You may need to identify Entity1 based on your data
                }
                rel_preds.append(relationship)
            else:
                print(f"Entity text not found for entity ID: {entity2_id}")
        else:
            print("Entity ID not found in the RE output.")
    print(rel_preds)

    db = Neo4jDatabase("YOUR_NEO4J_URI", "neo4j", "YOUR_NEO4J_PASSWORD")

    for record in rel_preds:
        entity1_text = record['Entity1 text']
        entity2_text = record['Entity2 text']
        relation_type = record['Relation type']
    
        # Create nodes and relationship in the Neo4j database
        db.create_node_and_relationship(entity1_text, entity2_text, relation_type)

    # Close the database connection
    db.close()

    html_ner = display_ehr(
        text=ner_input.ehr_text,
        entities=ner_predictions.get_entities(),
        relations=re_predictions.relations,
        return_html=True)

    graph_img = display_knowledge_graph(relation_table, return_html=True)
    
    if len(relation_table) > 0:
        relation_table_html = get_relation_table(relation_table)
    else:
        relation_table_html = "<p>No relations found</p>"

    if graph_img is None:
        graph_img = "<p>No Relation found!</p>"

    return {'tagged_text': html_ner, 're_table': relation_table_html, 'graph': graph_img}


@app.get("/sample/")
def get_sample_ehr():
    """Returns a sample EHR record"""
    return {"data": SAMPLE_EHR}


@app.post("/query/")
def getAnswer(question_request: QuestionRequest):
    question = question_request.question
    print(question)
    ner_predictions = get_ner_predictions(
        ehr_record=question,
        model_name="biobert")
    
    print(ner_predictions.entities)
    
    relationship = ""
    entity1 = ""
    entity2 = ""
    answer = ""

    db = Neo4jDatabase("YOUR_NEO4J_URI", "neo4j", "YOUR_NEO4J_PASSWORD")


    if find_specific_word(question) == "ade" or find_specific_word(question) == "sideeffects":
        relationship = "ADE-Drug"
    elif find_specific_word(question) == "frequency":
        relationship = "Frequency-Drug"
    elif find_specific_word(question) == "reason" or find_specific_word(question) == "reasons" or find_specific_word(question) == "suffering":
        relationship = "Reason-Drug"
    elif find_specific_word(question) == "dosage":
        relationship = "Dosage-Drug"
    elif find_specific_word(question) == "strength":
        relationship = "Strength-Drug"
    elif find_specific_word(question) == "route" or find_specific_word(question) == "routes":
        relationship = "Route-Drug"
    
    print(relationship)

    for entity in ner_predictions.entities:
        if(entity.name == "Drug"):
            entity1 = entity.ann_text
            print(entity1)
            if relationship != "" :
                with db._driver.session() as session:
                    query= ("MATCH p=(n:Entity{name: $e1})-[r:RELATION{type: $r1}]->() RETURN p;")
                    # Execute the query with parameters
                    result = session.run(
                        query,
                        e1=entity1,
                        r1=relationship
                    )

                    for record in result:
                        # Extract the end node from the result and get its name property
                        end_node_name = record['p'].end_node['name']

                        # Append it to the answer string
                        answer += end_node_name + "\n"

                    print(answer)
                    db.close()
            break    
        else:
            entity2 = entity.ann_text
            print(entity2)
            if relationship != "" :
               with db._driver.session() as session:
                    query= ("MATCH p=()-[r:RELATION{type: $r1}]->(n:Entity{name: $e2}) RETURN p;")
                    # Execute the query with parameters
                    result = session.run(
                        query,
                        e2=entity2,
                        r1=relationship
                    )

                    for record in result:
                        # Extract the end node from the result and get its name property
                        start_node_name = record['p'].start_node['name']

                        # Append it to the answer string
                        answer += start_node_name + "\n"

                    print(answer)
                    db.close() 
            break
    return {"answer": answer}