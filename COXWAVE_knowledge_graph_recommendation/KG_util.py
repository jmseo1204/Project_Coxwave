from langchain.graphs import Neo4jGraph
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from openai import OpenAI

import pickle
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import json
from tqdm import tqdm

load_dotenv()


def get_KG():
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    return graph


def get_words_to_embeddings(word: list[str] | str) -> dict:
    if isinstance(word, str):
        words = [word]
    else:
        words = word
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    word_embeddings = {}
    model = "text-embedding-3-small"
    for w in tqdm(words):
        response = client.embeddings.create(model=model, input=w)
        embedding = response.data[0].embedding
        word_embeddings[w] = embedding
    return word_embeddings[word] if isinstance(word, str) else word_embeddings


def get_json_from_prompt(messages: list[dict]):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def get_unique_target_people(graph, minimum_neighbors_for_target=5) -> list[str]:
    people = graph.query(
        "MATCH (p:Person) WHERE p.id =~ '[\d]{5}' AND COUNT {(p)-[]->()} >= "
        + str(minimum_neighbors_for_target)
        + " RETURN p.id AS person_id"
    )
    person_list = [person["person_id"] for person in people]
    # print(type(person_list[0]))
    person_series = pd.Series(person_list)
    return list(person_series.unique())


def get_unique_nodes(graph) -> list[str]:
    adj_nodes = graph.query("MATCH (p:Person)-[r]->(n) RETURN n.id as adj_node")
    adj_node_list = [dic["adj_node"] for dic in adj_nodes]
    adj_node_series = pd.Series(adj_node_list)
    # allowed_nodes = ["Person", "Job", "Location", "Event", "Hobby", "Food"]
    return list(adj_node_series.unique())


def get_adj_nodes(user_id: str, graph) -> list[str]:
    adj_nodes = graph.query(
        f"MATCH (p:Person)-[r]->(n) WHERE p.id = '{user_id}' RETURN n.id as node"
    )
    adj_node_list = [dic["node"] for dic in adj_nodes]
    adj_node_series = pd.Series(adj_node_list)
    return list(adj_node_series.unique())


def get_adj_rel_to_node(
    user_id: str | list[str], graph
) -> dict[str, list[dict[str, str]]]:
    if isinstance(user_id, str):
        user_ids = [user_id]
    else:
        user_ids = user_id

    rel_to_node_list = []
    for id in user_ids:
        adj_nodes_and_relations = graph.query(
            f"MATCH (p:Person)-[r]->(n) WHERE p.id = '{id}' RETURN n.id as node, r as relation"
        )
        rel_to_node = [
            {node_relation["relation"][1]: node_relation["node"]}
            for node_relation in adj_nodes_and_relations
        ]
        rel_to_node_list.append(rel_to_node)

    if isinstance(user_id, str):
        return rel_to_node_list[0]

    return dict(zip(user_ids, rel_to_node_list))


def get_unique_relationships(graph) -> list[str]:
    relations = graph.query("MATCH (p:Person)-[r]->(n) RETURN r")
    relations_list = [dic["r"][1] for dic in relations]
    relations_series = pd.Series(relations_list)
    # allowed_rels = ["like", "hate", "see", "hoby", "participated_in", "knows", "work_as", "visit", "interested_in"]
    # 이거 말고 다른 rel도 생기네..?
    return list(relations_series.unique())


def get_IDF_of_node(node_id: str, graph) -> dict[str, float]:
    if isinstance(node_id, str):
        node_ids = [node_id]
    else:
        node_ids = node_id

    node_to_IDF = {}
    for id in node_ids:
        indegree_from_nodes = graph.query(
            "MATCH (p:Person)-[r]->(n) WHERE n.id = '"
            + id
            + "' RETURN COUNT { (p)-[r]->(n) } AS indegree"
        )
        indegree = 0
        for node in indegree_from_nodes:
            indegree += node["indegree"]
        node_to_IDF[id] = 1 / np.log1p(indegree)
    return node_to_IDF[node_id] if isinstance(node_id, str) else node_to_IDF


def cos_mapping(vector1: list, vector2: list, mapping_func=lambda x: x) -> float:
    vector1 = np.array(vector1)
    vector1_normalized = vector1 / (np.linalg.norm(vector1) + 1e-8)
    vector2 = np.array(vector2)
    vector2_normalized = vector2 / (np.linalg.norm(vector2) + 1e-8)
    return mapping_func((vector1_normalized * vector2_normalized).sum())


def calculate_simularity(
    rels_to_nodes: list[dict[str, str]],
    rel_to_interest_score: dict[str, float],
    nodes_to_IDF: dict[str, float],
    nodes_to_embeddings: dict[str, list[float]],
    items_to_embeddings: dict[str, list[float]],
    # item_KG: dict[str, list],
) -> dict[str, float]:
    recommendation_items_score = {}
    for item in items_to_embeddings.keys():
        score_for_item = 0
        for rel_to_node in rels_to_nodes:
            rel = list(rel_to_node.keys())[0]
            node = list(rel_to_node.values())[0]

            # print(f"rel:{rel}, node:{node}, item:{item}")

            score_for_item += (
                rel_to_interest_score[rel]
                * nodes_to_IDF[node]
                * cos_mapping(
                    nodes_to_embeddings[node], items_to_embeddings[item], np.exp
                )
            )  # mapping_func = e^cos
        recommendation_items_score[item] = score_for_item
    return recommendation_items_score


def get_item_KG(item: str):
    if isinstance(item, str):
        items = [item]
    else:
        items = item

    items_to_KG = {}

    for i in items:
        prompt_to_make_item_KG = (
            "Make knowledege graph of properties of "
            + i
            + " which are the reasons expected customers consume "
            + i
            + ". Here is an example of the knowledge graph of flower delivery service."
            + '{"Freshness": ["Same-day delivery", "Locally sourced flowers", "Quality guarantee"], "Customization": ["Personalized arrangements", "Custom messages", "Various bouquet sizes"], "Convenience": ["Online ordering", "Scheduled deliveries", "Subscription plans"], "Special Occasions": ["Weddings", "Anniversaries", "Birthdays"], "Customer Service": ["24/7 support", "Satisfaction guarantee", "Easy returns"], "Unique Offerings": ["Exotic flowers", "Eco-friendly options", "Add-on gifts"]}'
            + '[detailed explanation of the example]  Output dict of FlowerDeliveryService is {"main property which is appealed to potential customers" : ["specific instance 1", "specific instance 2", "specific instance 3"]}'
        )

        messages = [
            {
                "role": "system",
                "content": "Make Knowledge graph of json format",
            },
            {"role": "user", "content": prompt_to_make_item_KG},
        ]

        items_to_KG[i] = get_json_from_prompt(messages)

    return items_to_KG[item] if isinstance(item, str) else items_to_KG


def store_graph_to_db(graph, made_KG):
    for recommended_item, properties in made_KG.items():
        graph.query("MERGE (i:RecommendedItem {id: '" + recommended_item + "'})")
        for property, values in properties.items():
            graph.query("MERGE (p:RecommendedItemProperty {id: '" + property + "'})")
            graph.query(
                f"""MATCH (i:RecommendedItem), (p:RecommendedItemProperty) 
                        WHERE i.id = '{recommended_item}' AND p.id = '{property}'
                        MERGE (i)-[r:HAS_A_PROPERTIY_OF]->(p) 
                        """
            )
            for value in values:
                graph.query("MERGE (i:RecommendedItemInstance {id: '" + value + "'})")
                graph.query(
                    f"""MATCH (p:RecommendedItemProperty), (i:RecommendedItemInstance) 
                        WHERE p.id = '{property}' AND i.id = '{value}'
                        MERGE (p)-[r:HAS_A_INSTANCE_OF]->(i) 
                        """
                )


class Word:
    def __init__(self, word, embedding=None):
        self.word = word
        self.embedding = (
            get_words_to_embeddings(word) if embedding == None else embedding
        )

    def set_weight_from_itemKG(
        self, itemKG, threshold=0, store_on_db=False, user_id=None, graph=None
    ):
        self.weight = {}

        for item, props_to_values in itemKG.items():
            properties = list(props_to_values.keys())
            similarity_of_properties = []
            for prop in properties:
                similarity_of_properties.append(self.similarity(prop))
            max_prop_idx = np.argmax(similarity_of_properties)
            max_prop = properties[max_prop_idx]

            similarity_of_values = []
            for value in props_to_values[max_prop]:
                similarity_of_values.append(self.similarity(value))
            max_val_idx = np.argmax(similarity_of_values)

            sim_score = (
                similarity_of_properties[max_prop_idx]
                * similarity_of_values[max_val_idx]
            )

            if sim_score > threshold:
                print(
                    f"< {item} >   ::   {self} --- {max_prop} --- {props_to_values[max_prop][max_val_idx]}  :  {sim_score}"
                )
                if store_on_db:
                    graph.query(
                        f"""MATCH (p: Person)-[r]->(n), (prop:RecommendedItemProperty) 
                                WHERE p.id = '{user_id}' AND prop.id = '{max_prop}' AND n.id = '{self.word.split()[1]}'
                                MERGE (n)-[rr:IS_SIMILAR_TO]->(prop) 
                                """
                    )

            self.weight[item] = sim_score

        return self.weight

    def get_weight(self):
        if hasattr(self, "weight"):
            return self.weight
        print("set weight before do this")
        return None

    def similarity(self, another_word) -> float:
        return cos_mapping(self.embedding, another_word.embedding)

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.word


def KG_to_embeddings(KG):
    items_to_properties = {}
    for item, properties in KG.items():
        props_to_values = {}
        for prop, values in properties.items():
            prop_word = Word(prop)
            value_words = []
            for value in values:
                value_word = Word(value)
                value_words.append(value_word)
            props_to_values[prop_word] = value_words

        items_to_properties[item] = props_to_values
    return items_to_properties


def score_of_items(
    relnodes: list[Word],
    itemKG: dict[str, dict[Word, list[Word]]],
    threshold=0,
    store_on_db=False,
    user_id=None,
    graph=None,
):
    print(
        "< Recommended Item >   ::   [Node Name]  ---  [Best Related Property]  ---  [Best Matched Instance]\n"
    )
    item_scores = {}
    items = itemKG.keys()
    for relnode in relnodes:
        relnode.set_weight_from_itemKG(itemKG, threshold, store_on_db, user_id, graph)

    for item in items:
        item_scores[item] = 0
        for relnode in relnodes:
            weight_of_item = relnode.get_weight()[item]
            if weight_of_item > threshold:
                item_scores[item] += weight_of_item

    print(item_scores)
    return item_scores


def get_relnode_embedding(
    user_rels_to_nodes: dict[str, list[dict[str, str]]]
) -> dict[str, list[dict[str, str]]]:
    if isinstance(user_rels_to_nodes, list):
        users_rels_to_nodes = {"11111": user_rels_to_nodes}
    else:
        users_rels_to_nodes = user_rels_to_nodes
    users_to_relnodes = {}
    for user, rels_to_nodes in users_rels_to_nodes.items():
        # print(rels_to_nodes)
        relnode_embeddings = []
        for rel_to_node in rels_to_nodes:
            rel = list(rel_to_node.keys())[0]
            node = list(rel_to_node.values())[0]
            relnode = rel + " " + node
            relnode_embeddings.append(Word(relnode))
        users_to_relnodes[user] = relnode_embeddings

    return (
        users_to_relnodes["11111"]
        if isinstance(user_rels_to_nodes, list)
        else users_to_relnodes
    )
