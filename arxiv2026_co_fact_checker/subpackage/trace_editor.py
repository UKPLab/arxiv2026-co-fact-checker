import json
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
from tqdm import tqdm
import re
import torch
import ollama

class TraceEditor:
    """
    Class representing a trace editor.
    """

    def __init__(self, verifier_model, editor_model):
        self.VERIFIER_MODEL = verifier_model
        self.EDITOR_MODEL = editor_model
        self.RM_MODEL = "BBQGOD/DeepSeek-GRM-16B"
        self.rm_pipe = pipeline(
            "text-generation",
            tokenizer=AutoTokenizer.from_pretrained(self.RM_MODEL, trust_remote_code=True),
            model=self.RM_MODEL,
            # model_kwargs={"load_in_8bit": True},
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",  # replace with "mps" to run on a Mac device
            trust_remote_code=True,
        )

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
6. Only give the JSON object as the output without extra reasoning.

## Perform the task as instructed --
"""

        self.VERIFY_PROMPT = f"""Fact-check the following CLAIM using the given EVIDENCE.

EVIDENCE:
{self._EVIDENCE_PLACEHOLDER}

CLAIM: "{self._CLAIM_PLACEHOLDER}"
"""

        self.STEP_EVAL_INSTRUCTION_PROMPT = """You are an expert fact-checker.

## Task
You must interact with the user and fact-check the claims provided by the user according to their preferences.

## Task Instructions
1. You will be provided with a CLAIM and relevant EVIDENCE. Based on the EVIDENCE, your task is to assess the factual accuracy of the CLAIM. Before presenting your conclusion, think step-by-step through the process using key points from the EVIDENCE.
2. You are given a spectrum of veracity labels to choose the claim veracity label from, along with a description of what each label means. Based on your reasoning, the claim veracity label should strictly be one of the labels from the following spectrum of labels: {}.
3. In case the evidence does not allow you to confidently reach a verdict, choose the 'not-enough-information' veracity label from the spectrum of labels.
4. Here, you are given a partial sequence of REASONING STEPS, a USER FEEDBACK INSTRUCTION for your reasoning, and you must produce the next step in the reasoning process.

## Perform the task as instructed --

Fact-check the following CLAIM using the given EVIDENCE.

EVIDENCE:
{}

CLAIM: "{}"

REASONING STEPS: {}

USER FEEDBACK INSTRUCTION: {}
"""

        self.STEP_EDITEVAL_USER_PROMPT = """You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them.
Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.
#### General Evaluation Criteria ####
1. Focus on the claim (Weight: 15%): The argument must focus on verifying the exact claim, or another interpretation of the claim that is equally or more threatful. The arguments and deductions made must be directly related to the claim, and not to other context or interpretations which are less important.
    - Highly Focused (9-10 points)
    - Partially Focused (6-8 points)
    - Somewhat Unfocused (3-5 points)
    - Unfocused (1-2 points)
2. Correctness of the argument (Weight: 25%): The argument must be logically sound, and be directly supported by the evidence provided.
    - Mostly Correct (9-10 points)
    - Partially Correct (6-8 points)
    - Somewhat Problematic (3-5 points)
    - Incorrect (1-2 points)
3. Speculation (Weight: 25%): The argument must not include any speculations or assumptions that are not directly supported by the evidence provided.
    - No Speculation (9-10 points)
    - Minor Speculation (6-8 points)
    - Some Speculation, or extended assumptions using the evidence (3-5 points)
    - Major Speculation, or no support from the evidence (1-2 points)
4. Adherence to the feedback instruction (Weight: 35%): The argument must adhere to the given feedback instruction, and follow the user's specific preferences and requirements.
    - Fully Consistent (9-10 points)
    - Mostly Consistent (6-8 points)
    - Partially Consistent (3-5 points)
    - Not Consistent (1-2 points)

#### Conversation Context ####
{conversation_context}

#### Responses to be Scored ####
{responses_scored}

#### Output Format Requirements ####

Output with three lines
Specific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.
Analysis: <Compare different responses based on given Criteria>.
Scores: <the overall comprehensive score of all resposnes in order, seperate by comma in the boxed, e.g., \\boxed{{x, y}} if there exists 2 responses>.""" 

        self.edit_rm_template = [{
            "role": "user",
            "content": self.STEP_EDITEVAL_USER_PROMPT
        }]

        self.STEP_GUIDEEVAL_USER_PROMPT = """You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them.
Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.
#### General Evaluation Criteria ####
1. Focus on the claim (Weight: 50%): The reasoning step must focus on verifying the exact claim, or another interpretation of the claim that is equally or more threatful. The reasoning step must be directly relevant to the claim, and not to other context or interpretations which are less important.
    - Highly Focused (9-10 points)
    - Partially Focused (6-8 points)
    - Somewhat Unfocused (3-5 points)
    - Unfocused (1-2 points)
2. Adherence to the feedback instruction (Weight: 50%): The reasoning step must adhere to the given feedback instruction, and follow the user's specific preferences and requirements.
    - Fully Consistent (9-10 points)
    - Mostly Consistent (6-8 points)
    - Partially Consistent (3-5 points)
    - Not Consistent (1-2 points)

#### Conversation Context ####
{conversation_context}

#### Responses to be Scored ####
{responses_scored}

#### Output Format Requirements ####

Output with three lines
Specific Criteria: <Other potential criteria specific to the query and the context, and the weights of each criteria>.
Analysis: <Compare different responses based on given Criteria>.
Scores: <the overall comprehensive score of all resposnes in order, seperate by comma in the boxed, e.g., \\boxed{{x, y}} if there exists 2 responses>.""" 

        self.guide_rm_template = [{
            "role": "user",
            "content": self.STEP_GUIDEEVAL_USER_PROMPT
        }]


    def extract_last_floats(self, text: str) -> list[float]:
        """
        Extracts the last sequence of floats from a given text string from the Judge Model response. The floats will be enclosed in either curly braces {} or square brackets [].
        Args:
            text (str): The input text string from which to extract the last sequence of floats.
        Returns:
            list[float]: A list of floats extracted from the last sequence in the text. If no valid sequence of floats is found, returns an empty list.
        """

        pattern = re.compile(
            r'(?:\\{1,2}boxed\{|\[)'
            r'\s*([^\]\}]+?)\s*'
            r'(?:\}|\])'
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        last_content = matches[-1].group(1)
        parts = re.split(r'\s*,\s*', last_content.strip())
        floats = []
        for p in parts:
            try:
                floats.append(float(p))
            except ValueError:
                pass
        return floats

    def create_evalinst_prompt(self, claim: str = '', search_results: list = [], label_space: list = [], reasoning_steps: str = "", user_instruction: str = "") -> str:
        """
        Creates an evaluation instruction prompt for the trace editor.
        Args:
            claim (str): The claim to be evaluated.
            search_results (list): A list of search results relevant to the claim.
            label_space (list): A list of labels to choose the claim's veracity from.
            reasoning_steps (str): The current reasoning steps taken for the claim.
            user_instruction (str): The user's feedback instruction on the reasoning process.
        Returns:
            str: The formatted evaluation instruction prompt.
        """

        EVIDENCE = '- ' + '\n- '.join([f"URL: {item['url']}\nRELEVANT TEXT: {item['evidence']}" for item in search_results])

        prompt = self.STEP_EVAL_INSTRUCTION_PROMPT.format(' or '.join(label_space), EVIDENCE, claim, reasoning_steps, user_instruction)
        return prompt

    def extract_json_from_output(self, model_output) -> dict | None:
        """Extracts JSON from model output string.

        Args:
            model_output (str): The output string from the model that may contain JSON.

        Returns:
            dict or None: Parsed JSON as a dictionary if found; None if not found or parsing fails.
        """

        if "</think>" in model_output:
            model_output = model_output.split("</think>")[-1].strip()

        json_match = re.search(r'{.*}', model_output, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError:
                print("Error parsing JSON")
                return None
        else:
            print("No JSON found in the output")
            return None

    def extract_list_from_nlfeedback(self, nl_feedback) -> list:
        """Extracts a list of points from natural language feedback.

        Args:
            nl_feedback (str): The natural language feedback string.

        Returns:
            list: A list of extracted points.
        """

        if "</think>" in nl_feedback:
            nl_feedback = nl_feedback.split("</think>")[-1].strip()

        list_match = re.search(r'\[.*\]', nl_feedback, re.DOTALL)

        if list_match:
            try:
                list_str = list_match.group(0)
                points_list = eval(list_str)
                points_list = [str(inst) for inst in points_list]
                return points_list
            except SyntaxError:
                print("Error evaluating list from feedback")
                return []
            except json.JSONDecodeError:
                print("Error parsing list from feedback")
                return []
        else:
            print("No list found in the feedback")
            return []

    def create_prompt_tt(self, claim: str = '', search_results: list = [], label_space: list = [], thinking_trace: str = "") -> str:
        """
        Creates a prompt for completing the thinking trace and model response generation, given a prefix thinking trace.
        Args:
            claim (str): The claim to be evaluated.
            search_results (list): A list of search results relevant to the claim.
            label_space (list): A list of labels to choose the claim's veracity from.
            thinking_trace (str): The current prefix of the thinking trace for the claim.
        Returns:
            str: The formatted prompt for thinking trace completion and model response generation.
        """

        EVIDENCE = '- ' + '\n- '.join([f"URL: {item['url']}\nRELEVANT TEXT: {item['evidence']}" for item in search_results])

        inst_prompt = self.INSTRUCTION_PROMPT.format(' or '.join(label_space))
        prompt = self.VERIFY_PROMPT.replace(self._CLAIM_PLACEHOLDER, claim)
        prompt = prompt.replace(self._EVIDENCE_PLACEHOLDER, EVIDENCE)
        prompt = inst_prompt + '\n' + prompt
        prompt = "<｜User｜>" + prompt + "<｜Assistant｜><think>\\n" + thinking_trace
        return prompt

    def generate_response_raw(self, model: str, prompt: str) -> str:
        """
        Helper function for completing the model response using the prompt as a prefix.
        Args:
            model (str): The name of the model to use for generation.
            prompt (str): The input prompt to guide the generation.
        Returns:
            str: The raw generated response from the model.
        """

        response = ollama.generate(model=model, prompt=prompt, raw=True, options={"temperature": 0.6}).response
        return response

    def generate_response(self, model: str, prompt: str) -> str:
        """
        Helper function to generate a model's chat response given a prompt.
        Args:
            model (str): The name of the model to use for generation.
            prompt (str): The input prompt to guide the generation.
        Returns:
            str: The generated response from the model.
        """

        response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.6}).response
        return response

    def assemble_responses(self, resp_list):
        """
        Helper function to assemble multiple responses into a single string for the Judge Model (Deepseek GRM).
        Args:
            resp_list (list): A list of individual responses to assemble.
        Returns:
            str: The assembled response string.
        """

        return "".join([f"[The Begin of Response {i+1}]\nNEXT STEP: {resp}\n[The End of Response {i+1}]\n\n" for i, resp in enumerate(resp_list)])

    def classify_feedback_instruction_type(self, claim, thinking_trace, response_json, feedback_instruction):
        """
        Helper function to classify the type of a feedback instruction as either 'EDIT', 'REMOVE', or 'GUIDE' using the editor model.
        Args:
            claim (str): The claim being evaluated.
            thinking_trace (str): The current thinking trace for the claim.
            response_json (dict): The JSON object containing the student's response.
            feedback_instruction (str): The feedback instruction to classify.
        Returns:
            str: The classified instruction type, which can be 'EDIT', 'REMOVE', or 'GUIDE'.
        """

        ask_prompt = f"""You are a helpful assistant experienced in fact-checking. A student training to become a fact-checker recently completed a fact-checking task, and produced a verdict for a given claim. His verdict includes the veracity of the claim, and an explanation for the assigned veracity label. He has also documented his thinking process, detailing the steps taken and arguments made using the evidence to arrive at the verdict. An expert fact-checker has evaluated the student's work based on several criteria, and provided some feedback, which instruct the student to make some changes in his thinking process. The feedback may ask the student to edit specific arguments in his thinking process, or to remove specific arguments from his thinking process, or to adjust the veracity or explanation of the claim in some way, or to focus on a specific aspect of the claim in his thinking process. You will be given the claim, the student's thinking process, the student issued veracity, its explanation, and the expert's feedback instructions one-by-one.

## Task 
Help the student by assessing whether the given feedback instruction asks the student to EDIT any specific arguments from his thinking process to correct them, or if the instruction asks the student to REMOVE any specific arguments from his thinking process, or if the instruction GUIDES the student to focus on a specific aspect of the claim in his thinking process. Be careful and classify the instruction as 'EDIT' or 'REMOVE' if the feedback instruction clearly indicates that specific arguments need to be edited or removed. If the feedback instruction primarily guides the student to focus on a specific aspect of the claim in his thinking process without explicitly asking for edits or removals, classify it as 'GUIDE'.

## Output Format
Structure your output as a json object with one key: "instruction_type", which should be either one of 'EDIT', 'REMOVE', or 'GUIDE' based on your analysis. Only return the JSON object as the output, and do not include any other reasoning. Only provide the json object as the output.

## Perform the task for the given claim, the student's thinking process, and the expert's feedback instruction --

CLAIM: "{claim}"
STUDENT'S THINKING PROCESS: "{thinking_trace}"
STUDENT-ISSUED VERACITY: "{response_json.get("final_answer", "other")}"
STUDENT-ISSUED EXPLANATION: "{response_json.get("explanation", "")}"
FEEDBACK INSTRUCTION: "{feedback_instruction}"
"""

        ask_output = self.generate_response(model=self.EDITOR_MODEL, prompt=ask_prompt)
        ask_output_json = self.extract_json_from_output(ask_output)

        if ask_output_json is not None:
            return ask_output_json.get("instruction_type", "")
        else:
            return ""

        
    def remove_step_based_on_instruction(self, claim, thinking_trace, step, response_json, feedback_instructions):
        """
        Helper function to remove a step from the thinking trace based on the feedback instructions, using the editor model to identify whether a given step needs to be removed.
        Args:
            claim (str): The claim being evaluated.
            thinking_trace (str): The current thinking trace for the claim.
            step (str): The specific step in the thinking trace being evaluated for potential removal.
            response_json (dict): The JSON object containing the student's response.
            feedback_instructions (list): A list of feedback instructions from the expert.
        Returns:
            bool: True if the step needs to be removed according to the feedback instructions, False otherwise. The function should return False if the step is a filler string, or if the feedback instructions do not clearly indicate that the step needs to be removed.
        
        """
        
        remove_prompt = f"""You are a helpful assistant experienced in fact-checking. A student training to become a fact-checker recently completed a fact-checking task, and produced a verdict for a given claim. His verdict includes the veracity of the claim, and an explanation for the assigned veracity label. He has also documented his thinking process, detailing the steps taken and arguments made using the evidence to arrive at the verdict. An expert fact-checker has evaluated the student's work based on several criteria, and provided some feedback, which instruct the student to remove certain specific arguments from his thinking process. You will be given the claim, the arguments in the student's thinking process one-by-one, the student's complete thinking process for context, the student issued veracity, its explanation, and the expert's feedback instructions.

## Task
Help the student by assessing whether the given feedback instructions ask the student to remove the ARGUMENT IN QUESTION from his thinking process based on the feedback instructions and considering the complete thinking process as context. Be strict and only ask for removals if the feedback instructions clearly indicate so.
        
## Output Format
Structure your output as a json object with one key: "to_remove" which must be either 'yes' if the ARGUMENT IN QUESTION must be deleted altogether from the student's thinking process, or 'no' if the ARGUMENT IN QUESTION need not be deleted. You should also output 'no' if the ARGUMENT IN QUESTION is a filler string. Only provide the json object as the output.

## Perform the task for the given claim, the student's thinking process, the argument in question, and the expert's feedback instructions --
CLAIM: "{claim}"
STUDENT'S COMPLETE THINKING PROCESS: "{thinking_trace}"
STUDENT-ISSUED VERACITY: "{response_json.get("final_answer", "other")}"
STUDENT-ISSUED EXPLANATION: "{response_json.get("explanation", "")}"

ARGUMENT IN QUESTION: "{step}"
FEEDBACK INSTRUCTIONS: "{" ".join(feedback_instructions)}"
"""

        remove_output = self.generate_response(model=self.EDITOR_MODEL, prompt=remove_prompt)
        remove_output_json = self.extract_json_from_output(remove_output)

        if remove_output_json is not None and remove_output_json.get("to_remove", "").lower() == "yes":
            print("Removing argument based on remove instructions:", step)
            return True

        return False

    def edit_step_based_on_instruction(self, claim, search_results, label_set, thinking_trace, partial_thinking_trace, step, response_json, feedback_instructions, sample_steps=4):
        """
        Helper function to edit a step from the thinking trace based on the feedback instructions, using the editor model to identify whether a given step needs to be edited, and to propose a corrected version of the step if it needs to be edited.
        Args:
            claim (str): The claim being evaluated.
            search_results (list): A list of search results relevant to the claim.
            label_set (list): A list of labels to choose the claim's veracity from.
            thinking_trace (str): The current thinking trace for the claim.
            step (str): The specific step in the thinking trace being evaluated for potential editing.
            response_json (dict): The JSON object containing the student's response.
            feedback_instructions (list): A list of feedback instructions from the expert.
            sample_steps (int): The number of edited samples to generate if the step needs to be edited, for the purpose of selecting the best edit using the RM.
        Returns:
            str or None: The edited version of the step if it needs to be edited, or None if it does not need to be edited or if the argument is a filler string. If multiple edits are generated, the best one is selected using the RM and returned.

        """

        to_edit_prompt = f"""You are a helpful assistant experienced in fact-checking. A student training to become a fact-checker recently completed a fact-checking task, and produced a verdict for a given claim. His verdict includes the veracity of the claim, and an explanation for the assigned veracity label. He has also documented his thinking process, detailing the steps taken and arguments made using the evidence to arrive at the verdict. An expert fact-checker has evaluated the student's work based on several criteria, and provided some feedback, which instruct the student to edit certain specific arguments in his thinking process. You will be given the claim, the arguments in the student's thinking process one-by-one, the student's complete thinking process for context, the student issued veracity, its explanation, and the expert's feedback instructions.

## Task
Help the student by assessing whether the given feedback instructions ask the student to edit the ARGUMENT IN QUESTION in his thinking process based on the feedback instructions and considering the complete thinking process as context. If the feedback instructions indicate that the ARGUMENT IN QUESTION needs to be edited say "yes", and if the feedback instructions do not indicate that the ARGUMENT IN QUESTION needs to be edited say "no". Be strict and only ask for edits if the feedback instructions clearly indicate so. You should also say "no" if the ARGUMENT IN QUESTION is a filler string.

## Output Format
Structure your output as a json object with one key: "to_edit", which must be either "yes" or "no" based on whether the ARGUMENT IN QUESTION needs to be edited according to the feedback instructions and considering the complete thinking process as context. You should output "no" if the ARGUMENT IN QUESTION is a filler string. Only provide the json object as the output.

## Perform the task for the given claim, the student's thinking process, the argument in question, and the expert's feedback instructions --
CLAIM: "{claim}"
STUDENT'S COMPLETE THINKING PROCESS: "{thinking_trace}"
STUDENT-ISSUED VERACITY: "{response_json.get("final_answer", "other")}"
STUDENT-ISSUED EXPLANATION: "{response_json.get("explanation", "")}"

ARGUMENT IN QUESTION: "{step}"
FEEDBACK INSTRUCTIONS: "{" ".join(feedback_instructions)}"
"""
        edit_output = self.generate_response(model=self.EDITOR_MODEL, prompt=to_edit_prompt)
        edit_output_json = self.extract_json_from_output(edit_output)
        if edit_output_json is not None and edit_output_json.get("to_edit", "").lower() == "yes":
            print("Editing step based on edit instructions:", step)

            sample_edit_prompt = f"""You are a helpful assistant experienced in fact-checking. A student training to become a fact-checker recently completed a fact-checking task, and produced a verdict for a given claim. His verdict includes the veracity of the claim, and an explanation for the assigned veracity label. He has also documented his thinking process, detailing the steps taken and arguments made using the evidence to arrive at the verdict. An expert fact-checker has evaluated the student's work based on several criteria, and provided some feedback, which instruct the student to edit certain specific arguments in his thinking process. You will be given the claim, the arguments in the student's thinking process one-by-one, the student's complete thinking process for context, the student issued veracity, its explanation, and the expert's feedback instructions.

## Task
Help the student by proposing a corrected version of the ARGUMENT IN QUESTION in his thinking process based on the feedback instructions and considering the complete thinking process as context.

## Output Format
Structure your output as a json object with one key: "correction", where you must provide the corrected version of the ARGUMENT IN QUESTION based on the feedback instructions, and considering the complete thinking process as context. Remember to maintain the writing style of the original thinking process for the corrected version of the argument. Only provide the json object as the output.

## Perform the task for the given claim, the student's thinking process, the argument in question, and the expert's feedback instructions --
CLAIM: "{claim}"
STUDENT'S COMPLETE THINKING PROCESS: "{thinking_trace}"
STUDENT-ISSUED VERACITY: "{response_json.get("final_answer", "other")}"
STUDENT-ISSUED EXPLANATION: "{response_json.get("explanation", "")}"

ARGUMENT IN QUESTION: "{step}"
FEEDBACK INSTRUCTIONS: "{" ".join(feedback_instructions)}"
"""

            sampled_edits = []

            for i in range(sample_steps):
                edit_output = self.generate_response(model=self.EDITOR_MODEL, prompt=sample_edit_prompt)
                edit_output_json = self.extract_json_from_output(edit_output)
                
                if edit_output_json is not None and len(edit_output_json.get("correction", "").lower()) > 0:
                    sampled_edits.append(edit_output_json.get("correction", "").strip())

            if len(sampled_edits) == 0:
                return None
            elif len(sampled_edits) == 1:
                return sampled_edits[0]
            else:
                best_sampled_edit = self.run_sampled_actions_through_RM(claim=claim, search_results=search_results, label_set=label_set, partial_thinking_trace=partial_thinking_trace, feedback_instructions=feedback_instructions, sampled_actions=sampled_edits, rm_prompt_template=self.edit_rm_template)

                return best_sampled_edit    
        
        return None

    def gen_guidance_string_based_on_instruction(self, claim, search_results, label_set, thinking_trace, response_json, feedback_instructions, sample_strings = 4):
        """
        Helper function to generate a guidance string based on the feedback instructions, using the editor model to identify the specific aspect of the claim that the student needs to focus on in his thinking process, and generating a guidance string to guide the student to focus on that aspect in his thinking process.
        
        Args:
            claim (str): The claim being evaluated.
            thinking_trace (str): The current thinking trace for the claim.
            response_json (dict): The JSON object containing the student's response.
            feedback_instructions (list): A list of feedback instructions from the expert.

        Returns:
            str: The generated guidance string to guide the student to focus on the specific aspect of the claim in his thinking process based on the feedback instructions.
        """

        guidance_string_prompt = f"""You are a helpful assistant experienced in fact-checking. A student training to become a fact-checker recently completed a fact-checking task, and produced a verdict for a given claim. His verdict includes the veracity of the claim, and an explanation for the assigned veracity label. He has also documented his thinking process, detailing the steps taken and arguments made using the evidence to arrive at the verdict. An expert fact-checker has evaluated the student's work based on several criteria, and provided some feedback, which instruct the student to extend his focus toward a specific aspect of the claim in his thinking process. You will be given the claim, the student's thinking process, the student issued veracity, its explanation, and the expert's feedback instructions.

## Task
Help the student by proposing a guidance string to append to the student's thinking process, to guide the student to focus on a specific aspect of the claim in his thinking process based on the feedback instructions and considering the complete thinking process as context. You must summarise the given feedback instructions into a concise string of the form "Wait, I should also consider X, Y, and Z" to guide the student to focus on specific aspects as mentioned in the feedback instructions. 

## Output Format
Structure your output as a json object with one key: "guidance_string", which must contain the proposed guidance string to append to the student's thinking process, to guide the student. Remember to maintain the writing style of the original thinking process for the guidance string. Only provide the json object as the output.

## Perform the task for the given claim, the student's thinking process, the argument in question, and the expert's feedback instructions --
CLAIM: "{claim}"
STUDENT'S COMPLETE THINKING PROCESS: "{thinking_trace}"
STUDENT-ISSUED VERACITY: "{response_json.get("final_answer", "other")}"
STUDENT-ISSUED EXPLANATION: "{response_json.get("explanation", "")}"

FEEDBACK INSTRUCTIONS: "{" ".join(feedback_instructions)}"
"""
        
        sampled_guidance_strings = []

        for i in range(sample_strings):
            guidance_output = self.generate_response(model=self.EDITOR_MODEL, prompt=guidance_string_prompt)
            guidance_output_json = self.extract_json_from_output(guidance_output)
            
            if guidance_output_json is not None and len(guidance_output_json.get("guidance_string", "").lower()) > 0:
                sampled_guidance_strings.append(guidance_output_json.get("guidance_string", "").strip())

        if len(sampled_guidance_strings) == 0:
            return ""

        elif len(sampled_guidance_strings) == 1:
            return sampled_guidance_strings[0]

        else:
            best_sampled_guidance_string = self.run_sampled_actions_through_RM(claim=claim, search_results=search_results, label_set=label_set, partial_thinking_trace=thinking_trace, feedback_instructions=feedback_instructions, sampled_actions=sampled_guidance_strings, rm_prompt_template=self.guide_rm_template)

            return best_sampled_guidance_string

        
    def run_sampled_actions_through_RM(self, claim, search_results, label_set, partial_thinking_trace, feedback_instructions, sampled_actions, rm_prompt_template):
        eval_instruction = self.create_evalinst_prompt(claim=claim, search_results=search_results, label_space=label_set, reasoning_steps=partial_thinking_trace, user_instruction=" ".join(feedback_instructions))

        tournament = deepcopy(sampled_actions)

        while len(tournament) != 1:
            round_winners = []
            for batch_start in range(0, len(tournament), 2):
                participants = tournament[batch_start:batch_start + 2]

                if len(participants) == 1:
                    winner = participants[0]
                    round_winners.append(winner)
                    continue

                rm_messages = deepcopy(rm_prompt_template)
                rm_messages[0]["content"] = rm_messages[0]["content"].format(conversation_context=eval_instruction, responses_scored=self.assemble_responses(participants))

                try:
                    rm_outputs = self.rm_pipe(rm_messages, max_new_tokens=2048, temperature=1.0, do_sample=True)
                    judgement = rm_outputs[0]['generated_text'][-1]["content"].strip()
                    rewards = self.extract_last_floats(judgement)

                    if len(rewards) >= 1:
                        best_output_idx = rewards.index(max(rewards))
                        best_correction = participants[best_output_idx]
                        print("Selected best correction from judge:", best_correction)

                    else:
                        best_correction = participants[0]
                        print("Selected first correction due to no rewards:", best_correction)

                except torch.OutOfMemoryError as e:
                    print(str(e))
                    best_correction = participants[0]
                    print("Selected first correction due to OOM:", best_correction)

                round_winners.append(best_correction)
            
            tournament = round_winners
        
        return tournament[0]


    def trace_edit_item(self, claim, label_set, search_results, verifier_response, nl_feedback):

        response = verifier_response
        response_json = self.extract_json_from_output(response)

        if not response:
            print("Invalid verifier response for the given claim:", claim)
            return verifier_response

        if not response_json:
            print("Verifier response does not contain a verdict for the given claim:", claim)
            return verifier_response

        response_thinking_trace = response.split("</think>")[0].strip("<think>").strip()
        reasoning_steps = [step.strip() for step in response_thinking_trace.split("\n\n")]
        new_reasoning_steps = deepcopy(reasoning_steps)

        feedback_list = self.extract_list_from_nlfeedback(nl_feedback)

        if len(feedback_list) != 0:

            guiding_insts = []
            edit_insts = []
            remove_insts = []

            for user_inst in feedback_list:

                instruction_type = self.classify_feedback_instruction_type(claim=claim, thinking_trace=response_thinking_trace, response_json=response_json, feedback_instruction=user_inst)
                
                if instruction_type == "GUIDE":
                    guiding_insts.append(user_inst)
                elif instruction_type == "EDIT":
                    edit_insts.append(user_inst)
                elif instruction_type == "REMOVE":
                    remove_insts.append(user_inst)

            # Remove concluding step
            for step_id, step in enumerate(new_reasoning_steps):
                
                processed_step = step.strip().lower()

                if processed_step.startswith("putting it together") or processed_step.startswith("putting it all together") or processed_step.startswith("putting together") or processed_step.startswith("in conclusion") or processed_step.startswith("to conclude") or processed_step.startswith("together"):
                    print("Identified potential concluding step:", step)
                    new_reasoning_steps[step_id] = ""
                    break
            
            # Perform removal trace-edits
            if len(remove_insts) > 0:
                for step_id, step in enumerate(new_reasoning_steps):
                    
                    processed_step = step.strip()

                    remove_step = self.remove_step_based_on_instruction(claim=claim, thinking_trace=response_thinking_trace, step=processed_step, response_json=response_json, feedback_instructions=remove_insts)

                    if remove_step:
                        new_reasoning_steps[step_id] = ""

            # Perform modification trace-edits
            if len(edit_insts) > 0:
                for step_id, step in enumerate(new_reasoning_steps):
                    
                    processed_step = step.strip()
                    partial_thinking_trace = "\n\n".join(new_reasoning_steps[:step_id])

                    edit_step = self.edit_step_based_on_instruction(claim=claim, search_results=search_results, label_set=label_set, thinking_trace=response_thinking_trace, partial_thinking_trace=partial_thinking_trace, step=processed_step, response_json=response_json, feedback_instructions=edit_insts)

                    if edit_step is not None and edit_step.strip() != "":
                        new_reasoning_steps[step_id] = edit_step.strip()

            # Perform guiding trace-edits by appending the guidance string to the end of the thinking trace
            if len(guiding_insts) > 0:
                guiding_insts_joined = " ".join(guiding_insts)
                
                guidance_string = self.gen_guidance_string_based_on_instruction(claim=claim, search_results=search_results, label_set=label_set, thinking_trace=response_thinking_trace, response_json=response_json, feedback_instructions=guiding_insts)

                if guidance_string is not None and guidance_string.strip() != "":
                    new_reasoning_steps.append(guidance_string.strip())

            new_reasoning_steps = [step for step in new_reasoning_steps if step.strip() != ""]
            edited_thinking_trace = "\n\n".join(new_reasoning_steps)

            continue_gen_prompt = self.create_prompt_tt(claim=claim, search_results=search_results, label_space=label_set, thinking_trace=edited_thinking_trace)
            new_response = edited_thinking_trace + "\n\n" + self.generate_response_raw(model=self.VERIFIER_MODEL, prompt=continue_gen_prompt).strip()

        else:
            new_response = response

        return new_response