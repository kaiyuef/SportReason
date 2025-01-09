
# Dataset Processing Workflow

This guide provides a step-by-step workflow for processing a dataset and related Wikipedia HTML files. The workflow includes preparing the dataset, integrating evidence, expanding Wikipedia data, and chunking JSONL files for efficient processing.

---

## Step 1: Prepare Dataset and Related Wikipedia HTML Files

### Dataset Format
The dataset should follow the structure of the `Sample` class:

```python
class Sample:
    id: str  # e.g., "sample_1"
    seed_question: str  # Original question from the source dataset
    seed_answers: list  # Original answers from the source dataset
    seed_dataset: str  # Original dataset name
    seed_split: str  # Original split (e.g., train, dev, test)
    seed_id: str  # Original question ID
    extended_question: str  # Proposed question
    answers: list  # Correct answers (e.g., ["Biden", "Joe Biden", "President Biden"])
    gold_evidence_ids: list  # List of gold evidence IDs (e.g., ["text_1", "table_2"])
    gold_evidence_type: dict  # Evidence types and counts (e.g., {"text": 2, "table": 1})
    gold_evidences: list  # List of evidence objects
    temporal_reasoning: bool  # Whether temporal reasoning is involved
    numerical_operation_program: str  # Program for numerical operations
    difficulty: str  # Difficulty level (low, medium, high)
    meta: dict  # Metadata with original sample info
```

### Gold Evidence Format
The gold evidence should follow one of the formats below:

#### Text Evidence
```json
{
    "id": "text_1",
    "title": "Document Title",
    "content": "String content of the text evidence"
}
```

#### Table Evidence
```json
{
    "id": "table_1",
    "title": "Table Title",
    "content": {
        "columns": ["Column1", "Column2"],
        "rows": [["Value1", "Value2"], ["Value3", "Value4"]]
    }
}
```

#### Infobox Evidence
```json
{
    "id": "infobox_1",
    "title": "Infobox Title",
    "content": {
        "category": "Category Name",
        "caption": "Caption Text",
        "etc.": "follow the original format"
    }
}
```

### HTML Format
Raw HTML should be downloaded directly from Wikipedia and stored for integration.



## Step 2: Process the Dataset with `corpus_form.py`
Integrate the raw HTML files into the dataset to form a complete corpus.



## Step 3: Verify Evidence with `evidence_exist.py`
Ensure that all gold evidence listed in the dataset exists within the corpus.



## Step 4: Expand Wikipedia Data with `wiki_expand.py`
Download expanded HTML files from Wikipedia to enrich the dataset.



## Step 5: Integrate Expanded Data with `expanded_corpus_form.py`
Incorporate the expanded Wikipedia data into the corpus and convert it to JSONL format.



## Step 6: Chunk Text-Related JSONL Files
Run chunking scripts on text-related JSONL files for efficient storage and processing.



## Step 7: Collect All Processed JSONL Files
Gather the following JSONL files for downstream tasks:
1. `chunked_data.jsonl`
2. `expanded_chunked_data.jsonl`
3. `expanded_infobox_content.jsonl`
4. `expanded_tables_content.jsonl`
5. `infobox_content_updated.jsonl`
6. `tables_content_updated.jsonl`



## File Overview

| **File Name**                  | **Description**                                      |
|--------------------------------|------------------------------------------------------|
| `chunked_data.jsonl`           | Chunked version of the primary corpus.              |
| `expanded_chunked_data.jsonl`  | Chunked version of the expanded corpus.             |
| `expanded_infobox_content.jsonl` | Expanded infobox content in JSONL format.          |
| `expanded_tables_content.jsonl` | Expanded tables content in JSONL format.           |
| `infobox_content_updated.jsonl` | Updated infobox content after integration.         |
| `tables_content_updated.jsonl`  | Updated tables content after integration.          |

---

## Additional Notes
- Ensure the dataset adheres to the specified formats before processing.
- Use appropriate tools and scripts for chunking and JSONL conversion.
- Validate the final corpus to ensure completeness and consistency.

---
