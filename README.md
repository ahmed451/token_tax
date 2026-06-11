# Token Tax: Tokenizer Fertility Rate Analysis

A tool for analyzing and comparing tokenizer fertility rates across multiple languages and large language models (LLMs). This project evaluates how efficiently different models tokenize text in various languages, which directly impacts inference costs and performance.

### Overview

Token fertility rate measures how many tokens a model requires to represent a given text. Lower fertility rates indicate more efficient tokenization, leading to:
- Reduced inference costs
- Faster processing times
- Lower memory usage

This tool analyzes the UN Parallel Corpus across 6 languages to compare tokenizer efficiency across multiple LLM providers.

Languages Supported

| Code | Language |
|------|----------|
| ar | Arabic |
| en | English |
| es | Spanish |
| fr | French |
| ru | Russian |
| zh | Chinese |

Directory Structure


bash
token_tax/
├── README.md
├── fertility_rate.py      # Main analysis script
└── data/
    ├── sample_UNv1.0.6way.ar
    ├── sample_UNv1.0.6way.en
    ├── sample_UNv1.0.6way.es
    ├── sample_UNv1.0.6way.fr
    ├── sample_UNv1.0.6way.ru
    └── sample_UNv1.0.6way.zh


### Quick Start

1. Download the UN Parallel Corpus

The dataset is split into multiple parts. Download all parts:

```bash
wget https://www.un.org/dgacm/sites/www.un.org.dgacm/files/files/UNCORPUS/UNv1.0.6way.tar.gz.00
wget https://www.un.org/dgacm/sites/www.un.org.dgacm/files/files/UNCORPUS/UNv1.0.6way.tar.gz.01
wget https://www.un.org/dgacm/sites/www.un.org.dgacm/files/files/UNCORPUS/UNv1.0.6way.tar.gz.02
wget https://www.un.org/dgacm/sites/www.un.org.dgacm/files/files/UNCORPUS/UNv1.0.6way.tar.gz.03
```

Alternative source: UN Corpus Download Page

2. Extract the Dataset

```bash
cat UNv1.0.6way.tar.gz.* | tar -xzf -
```

3. Create Sample Files (Optional)

To work with a manageable subset of the data:

```bash
ls UNv1.0.6way.* | while read -r l; do
    head -1365709 $l | tail -565709 > sample_$l
done
```

This extracts a 565,709-line sample from each language file.

4. Run the Analysis

```bash
for l in ar es ru en fr zh; do
    echo "Processing ${l}"
    python fertility_rate.py \
        --input /path/to/token_tax/data/sample_UNv1.0.6way.$l \
        --models \
            inceptionai/jais-13b-chat \
            humain-ai/ALLaM-7B-Instruct-preview \
            tiiuae/Falcon-H1R-7B \
            QCRI/Fanar-2-27B-Instruct \
            google/flan-t5-base \
            google/gemma-4-26B-A4B \
            mistralai/Mistral-7B-Instruct-v0.3 \
            Qwen/Qwen3.6-27B \
            01-ai/Yi-34B-Chat \
        --output results_UN_${l}.txt
done
```

### Languages and Models Analyzed


| Language Code | Language       |
|---------------|----------------|
| ar            | Arabic         |
| en            | English         |
| es            | Arabic         |
| fr            | Arabic         |
| ru            | Arabic         |
| zh            | Arabic         |



| Model | Organization | Notes |
|-------|--------------|-------|
| jais-13b-chat | Inception AI | Arabic-focused |
| ALLaM-7B-Instruct-preview | Humain AI | Arabic-focused |
| Falcon-H1R-7B | TII UAE | Multilingual |
| Fanar-2-27B-Instruct | QCRI | Arabic-focused |
| flan-t5-base | Google | General purpose |
| gemma-4-26B-A4B | Google | General purpose |
| Mistral-7B-Instruct-v0.3 | Mistral AI | General purpose |
| Qwen3.6-27B | Qwen | Chinese/General purpose |
| Yi-34B-Chat | 01.AI | Chinese/General purpose |

### Prerequisites

- Python 3.8+
- Hugging Face transformers library
- Hugging Face account (for gated models)


### Installation

```bash
pip install transformers torch accelerate sentencepiece
```

Authentication

Some models require Hugging Face authentication:

```bash
Login to Hugging Face
huggingface-cli login
```
Or set environment variable
```bash
export HF_TOKEN=your_huggingface_token_here
```

Gated models that may require approval:
- google/gemma-*
- mistralai/Mistral-*
- Qwen/Qwen-*

Output Format

Results are saved to results_UN_<language>.txt with the following structure:


Language: <language_code>
Model: <model_name>
Total Characters: <count>
Total Tokens: <count>
Fertility Rate: <tokens per character>
...


Interpretation

- Lower fertility rate = More efficient tokenizer (fewer tokens for same text)
- Higher fertility rate = Less efficient tokenizer (more tokens for same text)
- Compare rates across languages to identify models with better multilingual support

### License

This project uses the UN Parallel Corpus. Please review the UN Corpus license terms before use.


### Contributing

Contributions welcome! Areas for improvement:
- Visualization tools for results
- Batch processing optimizations