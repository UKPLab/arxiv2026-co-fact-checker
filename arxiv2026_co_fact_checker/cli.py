"""CLI interface for co_factchecker project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""
import os
import json
from .verifier import Verifier
from .subpackage import EvidenceRetriever, Oracle, TraceEditor
import argparse

def main():  # pragma: no cover

    parser = argparse.ArgumentParser(description="Co-FactChecker CLI")
    parser.add_argument("--file", type=str, default="example_data/data.json", help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, default="example_data/output.json", help="Path to output JSON file")
    parser.add_argument("--verifier_model", type=str, default="deepseek-r1:32b", help="Model to use for verification")
    parser.add_argument("--editor_model", type=str, default="llama3.2:3b", help="Model to use for trace editing")
    parser.add_argument("--oracle_model", type=str, default="gpt-4o-mini", help="Model to use for oracle feedback generation. Only ChatOpenAI models supported for now.")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333", help="URL of the Qdrant database")
    parser.add_argument("--qdrant_evidence_collection", type=str, default="ambiguous_snopes", help="Name of the Qdrant collection to use")
    parser.add_argument("--qdrant_num_searches", type=int, default=3, help="Number of search results to retrieve from Qdrant")
    parser.add_argument("--qdrant_score_threshold", type=float, default=0.7, help="Score threshold for filtering Qdrant search results")
    parser.add_argument("--qdrant_api_key", type=str, default="", help="API key for Qdrant database (if required)")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)

    evidence_retriever = EvidenceRetriever(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection_name=args.qdrant_evidence_collection,
        qdrant_num_searches=args.qdrant_num_searches,
        qdrant_score_threshold=args.qdrant_score_threshold
    )
    verifier = Verifier(verifier_model=args.verifier_model)
    oracle = Oracle(oracle_model=args.oracle_model)
    trace_editor = TraceEditor(verifier_model=args.verifier_model, editor_model=args.editor_model)

    verification_responses = []
    for claim_item in data:
        claim = claim_item['claim']
        label_set = claim_item['label_set']
        ref_veracity = claim_item['label']['rating']
        ref_explanation = claim_item['label']['context']
        ref_article = claim_item['report_content']

        search_results = evidence_retriever.run_evidence_retrieval(claim=claim)
        verifier_response = verifier.run_verify(claim=claim, label_set=label_set, search_results=search_results)[0]
        oracle_feedback_nl = oracle.generate_nl_feedback_item(claim=claim, label_set=label_set, ref_veracity=ref_veracity, ref_explanation=ref_explanation, ref_article=ref_article, verifier_response=verifier_response)
        trace_edited_response = trace_editor.trace_edit_item(claim=claim, label_set=label_set, search_results=search_results, verifier_response=verifier_response, nl_feedback=oracle_feedback_nl)

        verification_responses.append({
            "claim_item": claim_item,
            "search_results": search_results,
            "verifier_response": verifier_response,
            "oracle_feedback_nl": oracle_feedback_nl,
            "trace_edited_response": trace_edited_response
        })

    with open(args.output_file, "w") as f:
        json.dump(verification_responses, f, indent=4)