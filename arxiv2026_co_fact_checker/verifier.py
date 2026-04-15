# given a claim, return a list of related evidence
import json
import re
from dotenv import load_dotenv
from tqdm import tqdm
import ollama
import warnings
load_dotenv()

# Verifier prompt adapted from FIRE: https://github.com/mbzuai-nlp/fire

class Verifier:
    def __init__(self, verifier_model: str = 'deepseek-r1:14b'):
        self.verifier_model = verifier_model
        self._CLAIM_PLACEHOLDER = '[CLAIM]'
        self._EVIDENCE_PLACEHOLDER = '[EVIDENCE]'

        self.INSTRUCTION_PROMPT = """You are an expert fact-checker. 

## Task
You must interact with the user and fact-check the claims provided by the user according to their preferences.

## Task Instructions
1. You will be provided with a CLAIM and relevant EVIDENCE. Based on the EVIDENCE, you must assess the factual accuracy of the CLAIM. Before presenting your conclusion, think carefully through the process using key points from the EVIDENCE. 
2. If the EVIDENCE allows you to confidently make a decision, output the final answer as a JSON object in the following format:
{{
    "final_answer": <claim veracity label>,
    "explanation": <your explanation for the claim's veracity>
}}
3. You are given a spectrum of veracity labels to choose the <claim veracity label> from, along with a description of what each label means. Based on your reasoning, the <claim veracity label> should strictly be one of the labels from the following spectrum of labels: {}.
4. In case the evidence does not allow you to confidently reach a verdict, choose the 'not-enough-information' veracity label from the spectrum of labels. 
5. The explanation must be a detailed justification of how the EVIDENCE supports the claim's veracity label. The explanation should cite URLs from the EVIDENCE and give an overview of how the EVIDENCE argues the veracity assigned to the claim. 
6. Only give the JSON object as the output.

## Perform the task as instructed --
"""

        self.VERIFY_PROMPT = f"""Fact-check the following CLAIM using the given EVIDENCE.

EVIDENCE:
{self._EVIDENCE_PLACEHOLDER}

CLAIM: "{self._CLAIM_PLACEHOLDER}"
"""

    def create_prompt(self, claim: str = '', knowledge: str = '', label_space: str = '') -> str:
        inst_prompt = self.INSTRUCTION_PROMPT.format(' or '.join(label_space))
        prompt = self.VERIFY_PROMPT.replace(self._CLAIM_PLACEHOLDER, claim)
        prompt = prompt.replace(self._EVIDENCE_PLACEHOLDER, knowledge)
        prompt = inst_prompt + '\n' + prompt
        return prompt

    def generate_responses(self, model: str, prompt: str, num_generations: int = 1) -> str:
        responses = []
        for i in tqdm(range(num_generations), desc='Generating responses', total=num_generations):
            responses += [ollama.generate(model=model, prompt=prompt, options={"temperature": 0.6}).response]
        return responses

    def run_verify(self, claim: str, label_set: list, search_results: list, num_generations: int = 1):
        dedup_search_results = []
        for search_item in search_results:
            if search_item['evidence'] not in [item['evidence'] for item in dedup_search_results]:
                dedup_search_results.append(search_item)

        knowledge = '- ' + '\n- '.join([f"URL: {item['url']}" + ((f"\n{'WEBPAGE TITLE: '+item['title']}") if item.get('title', False) else '') + f"\nRELEVANT TEXT: {item['evidence']}" for item in dedup_search_results])

        prompt = self.create_prompt(claim=claim, knowledge=knowledge, label_space=label_set)
        responses = self.generate_responses(model=self.verifier_model, prompt=prompt, num_generations=num_generations)

        return responses