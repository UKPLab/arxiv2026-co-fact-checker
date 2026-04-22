# Co-FactChecker
[![Arxiv](https://img.shields.io/badge/Arxiv-2604.13706-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.13706)
[![License](https://img.shields.io/github/license/UKPLab/arxiv2026-co-fact-checker)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This is the official repository of the paper "Co-FactChecker: A Framework for Human-AI Collaborative Claim Verification Using Large Reasoning Models" currently under review. It contains scripts for our implementation of the Co-FactChecker framework.

It should help you start your project and give you continuous status updates on the development through [GitHub Actions](https://docs.github.com/en/actions).

> **Abstract:** Professional fact-checkers rely on domain knowledge and deep contextual understanding to verify claims. Large language models~(LLMs) and large reasoning models (LRMs) lack such grounding and primarily reason from available evidence alone, creating a mismatch between expert-led and fully automated claim verification. To mitigate this gap, we posit human--AI collaboration as a more promising path forward, where expert feedback, grounded in real-world knowledge and domain expertise, guides the model's reasoning. However, existing LRMs are hard to calibrate to natural language feedback, particularly in a multi-turn interaction setup. We propose **Co-FactChecker**, a framework for human--AI collaborative claim verification. We introduce a new interaction paradigm that treats the model's thinking trace as a shared scratchpad. Co-FactChecker translates expert feedback into _trace-edits_ that introduce targeted modifications to the trace, sidestepping the shortcomings of dialogue-based interaction. We provide theoretical results showing that trace-editing offers advantages over multi-turn dialogue, and our automatic evaluations demonstrate that Co-FactChecker outperforms existing autonomous and human--AI collaboration approaches. Human evaluations further show that Co-FactChecker is preferred over multi-turn dialogue, producing higher quality reasoning and verdicts along with relatively easier to interpret and more useful thinking traces.

Contact person: [Dhruv Sahnan](mailto:dhruv.sahnan@mbzuai.ac.ae) 

[MBZUAI](https://mbzuai.ac.ae/)
[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

## Getting Started

Simply clone the repository, create a virtual environment and install dependencies:

  ```bash
  git clone git@github.com:UKPLab/arxiv2026-co-fact-checker.git
  cd arxiv2026-co-fact-checker/
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements-dev.txt
  ```

You also need to export your OPENAI_API_KEY as an environment variable:

  ```bash
  export OPENAI_API_KEY=<your-api-key>
  ```

Additionally, you need to start a local Ollama server to run local models (Verifier and Editor):

  ```bash
  ollama serve
  ```

Finally, you will need to set up a vector database for evidence retrieval. We use Qdrant, and the scripts support this database. Qdrant allows a free cluster that you may use for setting up your vector database. In the paper, we ran experiments on the [ExClaim](https://github.com/znhy1024/JustiLM) dataset, which is built on top of the [WatClaimCheck](https://github.com/nxii/watclaimcheck). We suggest you to set up the ExClaim dataset in the same format as in [example_data/data.json](example_data/data.json) to run Co-FactChecker on the full dataset. You may set up the Qdrant vector database for the ExClaim dataset using evidence documents provided with the WatClaimCheck dataset.

## Usage

We offer the Co-FactChecker framework as a module. We also provide an example datapoint to run the framework on for testing it for yourself.

  ```bash
  python -m arxiv2026_co_fact_checker --file example_data/data.json --output_file example_data/output.json
  ```

### Supported parameters

- `--file <filename>`: path to the input data. Expects a json file. Use [example_data/data.json](example_data/data.json) to check data format for preparing your own dataset.
- `--output_file <filename>`: path to save the output from the Co-FactChecker framework for the claims in the input dataset.
- `--verifier_model <model_name>`: name of the Ollama supported model for the verifier in the Co-FactChecker framework. Make sure you use a model that supports a thinking mode, and also exposes the thinking tokens as a thinking trace. Default = `deepseek-r1:32b`.
- `--editor_model <model_name>`: name of the Ollama supported model for the editor in the Co-FactChecker framework. Default = `llama3.2:3b`.
- `--oracle_model <openai_model_name>`: name of the OpenAI model for the oracle in the Co-FactChecker framework. Only OpenAI models are supported for now. Default = `gpt-4o-mini`.
- `--qdrant_url <url>`: URL to the qdrant vector database cluster.
- `--qdrant_evidence_collection <collection_name>`: name of the qdrant collection on the vector database to use. Default = "evidence_collection".
- `--qdrant_num_searches <int>`: number of search results to retrieve per search on the qdrant vector database. Default = 3.
- `--qdrant_score_threshold <float>`: score threshold for filtering qdrant search results by cosine similarity between text embeddings. Default = 0.7.
- `--qdrant_api_key <key>`: you API key to access the qdrant vector database.

## Cite

Citation releasing soon.

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective research publication. 
