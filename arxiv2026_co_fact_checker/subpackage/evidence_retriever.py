from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import torch
import spacy
import os
import warnings
from typing import List
import numpy as np
import time
import json
import ast
import re
from langchain_openai import ChatOpenAI

## Evidence retrieval is based on factcheckgpt. Source: https://github.com/yuxiaw/Factcheck-GPT

QUESTION_GENERATION_PROMPT = """You are an expert fact-checker.

## Task
Given a claim, write web search queries that gather evidence to verify the claim.

## Instructions
1. Break the claim into atomic verifiable sub-claims.
2. Write 3-5 search queries.
3. Queries must be diverse, non-redundant, and concise.
4. Focus on the entities, dates, places and numbers mentioned in the claim.
5. Do not provide explanations, only the numbered query list.
6. Only return a python list of strings, each being a search query. Do not give any explanation or description.


## Example 1
Claim: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.
Search Queries: ['who sings "The Time of My Life" Dirty Dancing soundtrack', '"The Time of My Life" original recording artists', 'what movie features "The Time of My Life"', '"The Time of My Life" producer']

## Example 2
Claim: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
Search Queries: ['does nose switch between nostrils?', 'how often do our nostrils switch', 'why does our nostril switch', 'what is nasal cycle']

## Example 3
Claim: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
Search Queries: ['where was the Stanford Prison Experiment conducted', 'what building was the Stanford Prison Experiment conducted in']

Now, generate search queries for the following claim:
Claim: {claim}
""".strip()


class EvidenceRetriever:
    """
    Class representing the evidence retriever.

    Methods
    -------
    __init__():
        Initializes a new instance of the EvidenceRetriever.
    """

    _qdrant_embedding_model = None

    def __init__(self, qdrant_url, qdrant_api_key, qdrant_collection_name, qdrant_num_searches=3, qdrant_score_threshold=0.7):
        self.QDRANT_URL = qdrant_url
        self.QDRANT_API_KEY = qdrant_api_key
        self.QDRANT_COLLECTION_NAME = qdrant_collection_name
        self.QDRANT_NUM_SEARCHES = qdrant_num_searches
        self.QDRANT_SCORE_THRESHOLD = qdrant_score_threshold
        self.QDRANT_FETCHED_IDS = []

    @staticmethod
    def _get_qdrant_embedding_model():
        if EvidenceRetriever._qdrant_embedding_model is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            EvidenceRetriever._qdrant_embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5").to(device)
        return EvidenceRetriever._qdrant_embedding_model

    def parse_api_response(self, api_response: str) -> List[str]:
        """Extract search queries from a Python list of strings in the API response.

        The model is instructed to return a Python list, but it may add extra text.
        This function searches for a valid Python list literal and returns its string items.

        Args:
            api_response: Question generation response from model.
        Returns:
            questions: A list of parsed search queries.
        """
        candidates = [api_response.strip()]
        candidates.extend(match.group(0) for match in re.finditer(r"\[[\s\S]*?\]", api_response))

        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue

            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return [item.strip() for item in parsed if item.strip()]

        return []

    def run_question_generation(self, prompt, model, temperature, num_rounds, num_retries=5):
        questions = set()
        for _ in range(num_rounds):
            qgen_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=None,
                timeout=None,
                max_retries=num_retries
            )

            messages = [
                ("human", prompt),
            ]
            response = qgen_llm.invoke(messages).content.strip()
            cur_round_questions = self.parse_api_response(
                response 
            )
                
            questions.update(cur_round_questions)

        questions = list(sorted(questions))
        return questions

    def remove_duplicate_questions(self, model, all_questions):
        qset = [all_questions[0]]
        for question in all_questions[1:]:
            q_list = [(q, question) for q in qset]
            scores = model.predict(q_list)
            if np.max(scores) < 0.60:
                qset.append(question)
        return qset

    def get_queries_for_claim(self, claim: str) -> list:
        question_duplicate_model = CrossEncoder(
            "navteca/quora-roberta-base",
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        # give a claim, sometimes return the empty question list
        questions = []
        num_tries = 0
        while len(questions) <= 10 and num_tries < 8:
            questions += self.run_question_generation(
                prompt=QUESTION_GENERATION_PROMPT.format(claim=claim),
                model="gpt-4o-mini",
                temperature=0.7,
                num_rounds=2,
            )
            num_tries += 1
        questions = list(set(questions))

        if len(questions) > 0:
            questions = self.remove_duplicate_questions(question_duplicate_model, questions)
        questions = list(questions)

        return questions

    def fetch_evidence(self, query: str) -> list:

        # client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        client = QdrantClient(url="https://195c23ef-1d4c-4814-87ec-cde9d1c0d3cb.eu-west-2-0.aws.cloud.qdrant.io", api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTFjNTVhZmUtMTYyNi00NjRkLWExODktNWUwZjkzNGVmMTJhIn0.I7PsGtw_CdmHaG0fHZjMd2PJtzvmDycMj09SEo99N9k")

        qdrant_embedding_model = EvidenceRetriever._get_qdrant_embedding_model()
        query_embedding = qdrant_embedding_model.encode("Represent this sentence for searching relevant passages: " + query).tolist()
        qdrant_search_result = client.search(
            collection_name=self.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=self.QDRANT_NUM_SEARCHES,
            with_payload=True,
            score_threshold=self.QDRANT_SCORE_THRESHOLD,
            query_filter=models.Filter(
                must_not=[
                    models.HasIdCondition(has_id=self.QDRANT_FETCHED_IDS)
                ]
            )
        )

        self.QDRANT_FETCHED_IDS += [point.id for point in qdrant_search_result]
        self.QDRANT_FETCHED_IDS = list(set(self.QDRANT_FETCHED_IDS))
        qdrant_search_result = [point.payload for point in qdrant_search_result]

        return qdrant_search_result

    def run_evidence_retrieval(self, claim):
        queries = self.get_queries_for_claim(claim = claim)
        if queries is not None:
            search_results = []
            for num,query in enumerate(queries):
                search_result = self.fetch_evidence(query=query)#, qdrant_collection_name="exclaim")
                search_results += search_result

        else:
            search_results = []

        return search_results