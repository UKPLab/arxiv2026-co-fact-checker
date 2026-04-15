import re
import torch
import torch
import json
from tqdm import tqdm
import re
import os
from copy import deepcopy
from .criteria_objective import extrinsic_criterion, intrinsic_overall_criterion, argument_level_criterion
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


class Oracle:
    """
    Oracle class containing functions to run the Oracle feedback generation on a given model response.
    """

    def __init__(self, oracle_model):
        self.judge_llm = ChatOpenAI(
            model=oracle_model,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

    def extract_json_from_output(self, model_output) -> dict | None:
        """Extracts JSON from model output string.

        Args:
            model_output (str): The output string from the model that may contain JSON.

        Returns:
            dict or None: Parsed JSON as a dictionary if found; None if not found or parsing fails.
        """

        if "</think>" in model_output:
            model_output = model_output.split("</think>")[-1].strip()

        model_output = model_output.replace('\n', ' ').replace('\\n', ' ').replace('```json', '').replace('```', '').strip()

        # Step 1: Use regex to find the JSON block in the output
        json_match = re.search(r'{.*}', model_output, re.DOTALL)

        # Step 2: If JSON is found, attempt to parse it
        if json_match:
            try:
                # Extract JSON string
                json_str = json_match.group(0)
                # Parse the JSON string
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError:
                print("Error parsing JSON")
                return None
        else:
            print("No JSON found in the output")
            return None

    def generate_intrinsic_evaluation_report_item(self, claim, label_set, response, response_json):
        intrinsic_report = deepcopy(intrinsic_overall_criterion)

        for eval_item_id in range(len(intrinsic_report)):
            eval_item = intrinsic_report[eval_item_id]

            if eval_item['name'] == "label_in_set":
                formatted_label_set = [i.split("' (")[0].strip("'").lower() for i in label_set]
                if response_json.get('final_answer', "").strip("'").lower() not in formatted_label_set:
                    eval_item['verdict'] = "INCORRECT. The label provided must be from the input label set."
                else:
                    eval_item['verdict'] = "CORRECT. The label provided is from the input label set."

            else:
                eval_prompt_prefix = f"""{eval_item["desc"]}"""
                eval_prompt = None

                if eval_item["name"] == "consistency_veracity":
                    icl_eg_prompt = ""
                    for icl_eg in eval_item["icl_egs"]:
                        icl_eg_json = self.extract_json_from_output(icl_eg["model_response"])
                        icl_eg_prompt += f"""<user>\nCLAIM: "{icl_eg["claim"]}\nSTUDENT THINKING PROCESS: "{icl_eg["model_response"].split('</think>')[0].strip()}"\nSTUDENT-ISSUED VERACITY LABEL: "{icl_eg_json.get('final_answer', "None")}"\n</user>\n\n<assistant>{icl_eg["eval_response"]}\n</assistant>"""
                    eval_prompt = f"""{eval_prompt_prefix}\n\n{icl_eg_prompt}\n\n<user>CLAIM: "{claim}"\nSTUDENT THINKING PROCESS: "{response.split('</think>')[0].strip()}"\nSTUDENT-ISSUED VERACITY LABEL: "{response_json.get('final_answer', "None")}"\n</user>\n\n<assistant>"""
                
                elif eval_item["name"] == "consistency_explanation":
                    icl_eg_prompt = ""
                    for icl_eg in eval_item["icl_egs"]:
                        icl_eg_json = self.extract_json_from_output(icl_eg["model_response"])
                        icl_eg_prompt += f"""<user>\nCLAIM: "{icl_eg["claim"]}\nSTUDENT THINKING PROCESS: "{icl_eg["model_response"].split('</think>')[0].strip()}"\nSTUDENT-ISSUED EXPLANATION: "{icl_eg_json.get('explanation', "")}"\n</user>\n\n<assistant>{icl_eg["eval_response"]}\n</assistant>"""
                    eval_prompt = f"""{eval_prompt_prefix}\n\n{icl_eg_prompt}\n\n<user>CLAIM: "{claim}"\nSTUDENT THINKING PROCESS: "{response.split('</think>')[0].strip()}"\nSTUDENT-ISSUED EXPLANATION: "{response_json.get('explanation', "")}"\n</user>\n\n<assistant>"""
                elif eval_item["name"] == "focus_on_claim_wording":
                    icl_eg_prompt = ""
                    for icl_eg in eval_item["icl_egs"]:
                        icl_eg_json = self.extract_json_from_output(icl_eg["model_response"])
                        icl_eg_prompt += f"""<user>\nCLAIM: "{icl_eg["claim"]}\nSTUDENT-ISSUED VERACITY LABEL: "{icl_eg_json.get('final_answer', "None")}"\nSTUDENT-ISSUED EXPLANATION: "{icl_eg_json.get('explanation', "")}"\nSTUDENT THINKING PROCESS: "{icl_eg["model_response"].split('</think>')[0].strip()}"\n</user>\n\n<assistant>{icl_eg["eval_response"]}\n</assistant>"""
                    eval_prompt = f"""{eval_prompt_prefix}\n\n{icl_eg_prompt}\n\n<user>CLAIM: "{claim}"\nSTUDENT-ISSUED VERACITY LABEL: "{response_json.get('final_answer', "None")}"\nSTUDENT-ISSUED EXPLANATION: "{response_json.get('explanation', "")}"\nSTUDENT THINKING PROCESS: "{response.split('</think>')[0].strip()}"\n</user>\n\n<assistant>"""

                eval_output = None
                if eval_prompt is not None:
                    messages = [
                        ("human", eval_prompt),
                    ]
                    eval_output = self.judge_llm.invoke(messages).content.strip()

                eval_item["evaluator_response"] = eval_output
            intrinsic_report[eval_item_id] = eval_item

        return intrinsic_report
    
    def generate_extrinsic_evaluation_report_item(self, claim, response, response_json, ref_veracity, ref_explanation, ref_article):
        extrinsic_report = deepcopy(extrinsic_criterion)

        for eval_item_id in range(len(extrinsic_report)):
            eval_item = extrinsic_report[eval_item_id]

            if eval_item['name'] == "correctness_veracity":
                if response_json.get('final_answer', "").lower() == ref_veracity.lower():
                    eval_item['verdict'] = "CORRECT. The veracity label provided matches the human preferred label."
                else:
                    eval_item['verdict'] = "INCORRECT. The veracity label provided does not match the human preferred label. You should think about adjusting the label a little bit."

            else:
                eval_prompt_prefix = f"""{eval_item["desc"]}"""
                eval_prompt = None
                if eval_item['name'] == "correctness_explanation":
                    icl_eg_prompt = ""
                    for icl_eg in eval_item["icl_egs"]:
                        icl_eg_prompt += f"""<user>\nCLAIM: "{icl_eg["claim"]}\nSTUDENT-ISSUED EXPLANATION: "{icl_eg["model_response"]}"\nEXPERT-WRITTEN REFERENCE EXPLANATION: "{icl_eg["reference"]}"</user>\n\n<assistant>{icl_eg["eval_response"]}\n</assistant>"""
                    eval_prompt = f"""{eval_prompt_prefix}\n\n{icl_eg_prompt}\n\n<user>CLAIM: "{claim}"\nSTUDENT-ISSUED EXPLANATION: "{response_json.get('explanation', "")}"\nEXPERT-WRITTEN REFERENCE EXPLANATION: "{ref_explanation}"</user>\n\n<assistant>"""
                elif eval_item['name'] == "identification_harmful_part":
                    icl_eg_prompt = ""
                    for icl_eg in eval_item["icl_egs"]:
                        icl_eg_json = self.extract_json_from_output(icl_eg["model_response"])
                        icl_eg_prompt += f"""<user>\nCLAIM: "{icl_eg["claim"]}\nSTUDENT THINKING PROCESS: "{icl_eg["model_response"].split('</think>')[0].strip()}"\nSTUDENT-ISSUED VERACITY LABEL: "{icl_eg_json.get('final_answer', "None")}"\nSTUDENT-ISSUED EXPLANATION: "{icl_eg_json.get('explanation', "")}"\nEXPERT-WRITTEN REFERENCE EXPLANATION: "{icl_eg["reference"]}"\n</user>\n\n<assistant>{icl_eg["eval_response"]}\n</assistant>"""
                    eval_prompt = f"""{eval_prompt_prefix}\n\n{icl_eg_prompt}\n\n<user>CLAIM: "{claim}"\nSTUDENT THINKING PROCESS: "{response.split('</think>')[0].strip()}\nSTUDENT-ISSUED VERACITY LABEL: "{response_json.get('final_answer', "None")}"\nSTUDENT-ISSUED EXPLANATION: "{response_json.get('explanation', "")}"\nEXPERT-WRITTEN REFERENCE EXPLANATION: "{ref_explanation}"\n</user>\n\n<assistant>"""
                elif eval_item['name'] == "missing_arguments":
                    icl_eg_prompt = ""
                    for icl_eg in eval_item["icl_egs"]:
                        icl_eg_json = self.extract_json_from_output(icl_eg["model_response"])
                        icl_eg_prompt += f"""<user>\nCLAIM: "{icl_eg["claim"]}\nSTUDENT THINKING PROCESS: "{icl_eg["model_response"].split('</think>')[0].strip()}"\nEXPERT-WRITTEN FACT-CHECK: "{icl_eg["reference"]}"\n</user>\n\n<assistant>{icl_eg["eval_response"]}\n</assistant>"""
                    eval_prompt = f"""{eval_prompt_prefix}\n\n{icl_eg_prompt}\n\n<user>CLAIM: "{claim}"\nSTUDENT THINKING PROCESS: "{response.split('</think>')[0].strip()}\nEXPERT-WRITTEN FACT-CHECK: "{ref_article}"\n</user>\n\n<assistant>"""

                eval_output = None
                if eval_prompt is not None:
                    messages = [
                        ("human", eval_prompt),
                    ]
                    eval_output = self.judge_llm.invoke(messages).content.strip()

                eval_item["evaluator_response"] = eval_output
            extrinsic_report[eval_item_id] = eval_item

        return extrinsic_report

    def generate_argument_level_report_item(self, claim, response, ref_article):
        argument_level_report = deepcopy(argument_level_criterion)

        response_thinking_process = response.split('</think>')[0].strip("<think>").strip()
        response_thinking_process = response_thinking_process.split("\n\n")

        for eval_item_id in range(len(argument_level_report)):
            eval_item = argument_level_report[eval_item_id]

            if eval_item['name'] == "correctness_arguments":

                eval_prompt_prefix = f"""{eval_item["desc"]}"""

                icl_eg_prompt = ""
                for icl_eg in eval_item["icl_egs"]:
                    icl_eg_prompt += f"""<user>\nCLAIM: "{icl_eg["claim"]}\nSTUDENT ARGUMENT: "{icl_eg["argument"]}"\nSTUDENT ARGUMENT CONTEXT: "{icl_eg["context"]}"\nEXPERT-WRITTEN REFERENCE FACT-CHECK: "{icl_eg['reference']}"</user>\n\n<assistant>{icl_eg["eval_response"]}\n</assistant>"""

                for stud_arg_id, stud_arg in enumerate(response_thinking_process):
                    stud_arg_context = '\n'.join(response_thinking_process[:stud_arg_id])
                    eval_prompt = f"""{eval_prompt_prefix}\n\n{icl_eg_prompt}\n\n<user>CLAIM: "{claim}"\nSTUDENT ARGUMENT: "{stud_arg}"\nSTUDENT ARGUMENT CONTEXT: "{stud_arg_context}"\nEXPERT-WRITTEN REFERENCE FACT-CHECK: "{ref_article}"</user>\n\n<assistant>"""

                    eval_output = None
                    messages = [
                        ("human", eval_prompt),
                    ]
                    eval_output = self.judge_llm.invoke(messages).content.strip()

                    eval_item["evaluator_response"] = eval_item.get("evaluator_response", []) + [{"argument": stud_arg, "argument_context": stud_arg_context, "eval_output": eval_output}]

            argument_level_report[eval_item_id] = eval_item
        
        return argument_level_report

    def generate_evaluation_report_item(self, claim, label_set, ref_veracity, ref_explanation, ref_article, verifier_response):

        response = verifier_response
        response_json = self.extract_json_from_output(response)

        if response_json is not None:
            intrinsic_report = self.generate_intrinsic_evaluation_report_item(claim, label_set, response, response_json)
            extrinsic_report = self.generate_extrinsic_evaluation_report_item(claim, response, response_json, ref_veracity, ref_explanation, ref_article)
            argument_level_report = self.generate_argument_level_report_item(claim, response, ref_article)

            return intrinsic_report, extrinsic_report, argument_level_report

        return None, None, None

    def generate_nl_feedback_item(self, claim, label_set, ref_veracity, ref_explanation, ref_article, verifier_response):

        intrinsic_report, extrinsic_report, argument_level_report = self.generate_evaluation_report_item(claim, label_set, ref_veracity, ref_explanation, ref_article, verifier_response)

        response = verifier_response
        response_json = self.extract_json_from_output(response)

        if intrinsic_report is None or extrinsic_report is None or argument_level_report is None:
            print("Could not generate evaluation report due to missing or unparsable model response.")
            return ""
        
        if response_json is not None and 'final_answer' in response_json and 'explanation' in response_json:

            collated_eval_report = []

            for crit_id, criterion_item in enumerate(intrinsic_report):
                to_add = f"""Criterion {crit_id+1}: {criterion_item['name']}\nEvaluation: {criterion_item.get("evaluator_response", criterion_item.get("verdict", ""))}"""
                collated_eval_report.append(to_add)

            for crit_id, criterion_item in enumerate(extrinsic_report):
                to_add = f"""Criterion {crit_id+1 +len(intrinsic_report)}: {criterion_item['name']}\nEvaluation: {criterion_item.get("evaluator_response", criterion_item.get("verdict", ""))}"""
                collated_eval_report.append(to_add)

            for crit_id, criterion_item in enumerate(argument_level_report):
                to_add = f"""Criterion {crit_id+1 +len(intrinsic_report) + len(extrinsic_report)}: {criterion_item['name']}\n"""
                collated_eval_report.append(to_add)
                arg_to_add = []
                for eval_arg_id, eval_arg in enumerate(criterion_item.get('evaluator_response', [])):
                    arg_to_add.append(f'Argument {eval_arg_id + 1} from thinking process: {eval_arg["argument"]}\nEvaluation: {eval_arg["eval_output"]}')
                if len(arg_to_add) > 0:
                    collated_eval_report.append("\n".join(arg_to_add))

            collated_eval_report = "\n".join(collated_eval_report)

            gen_feedback_prompt = """You are an expert at giving guided feedback instructions to students. A student training to be a fact-checker recently completed a fact-checking task, and produced a verdict for a given claim. His verdict includes the veracity of the claim, and a explanation for the assigned veracity label. The student has also documented his thinking process, detailing the steps taken and deductions made using the evidence to arrive at the verdict. Since then, an expert has evaluated the student's work based on several criteria, providing a detailed evaluation report. You will be provided with the claim, the student's thinking process and verdict, and the expert's evaluation report. 
            
            ## Task
            Your task is to translate this evaluation report into a set of feedback instructions for the student. 
            
            ## Task Instructions
            1. The feedback should explicitly instruct the student to edit specific arguments in his thinking process; remove some specific arguments that are deemed irrelevant or incorrect; or guide the student to focus on a specific aspect of the claim using arguments that are deemed missing by the evaluation report but necessary for a comprehensive analysis. 
            2. Ensure that the feedback is constructive and actionable, enabling the student to enhance his fact-checking skills effectively. 
            3. Only your feedback instructions will be provided to the student to improve his solution, and not the evaluation report! So, make sure your instructions are complete.
            4. Structure your output as a python list containing 5 specific instructions. Be direct in what you want the student to do, and provide verbatim what you want to edit, remove, or focus on.\n\n
            
            ## Perform the task as instructed --"""

            gen_feedback_prompt += f"""**CLAIM**: \"{claim}\"\n**STUDENT THINKING PROCESS**: \"{response.split('</think>')[0].strip('<think>').strip()}\"\n**STUDENT-ISSUED VERACITY LABEL**: \"{response_json['final_answer']}\"\n**STUDENT-ISSUED EXPLANATION**: \"{response_json['explanation']}\"\n**EVALUATION REPORT**\n{collated_eval_report}\n\n"""

            messages = [
                ("human", gen_feedback_prompt),
            ]
            feedback = self.judge_llm.invoke(messages).content.strip()

        else:
            feedback = ""

        return feedback