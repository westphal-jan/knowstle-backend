# Sentence Embedding Model for MS MARCO Passage Retrieval


This a `distilroberta-base` model from the [sentence-transformers](https://github.com/UKPLab/sentence-transformers)-repository. It was trained on the [MS MARCO Passage Retrieval dataset](https://github.com/microsoft/MSMARCO-Passage-Ranking): Given a search query, it finds the relevant passages.

You can use this model for semantic search. Details can be found on: [SBERT.net - Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html).

This model was optimized to be used with **cosine-similarity** as similarity function between queries and documents.


## Training

Details about the training of the models can be found here: [SBERT.net - MS MARCO](https://www.sbert.net/examples/training/ms_marco/README.html)

## Performance

For performance details, see: [SBERT.net - Pre-Trained Models - MS MARCO](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)

## Usage (HuggingFace Models Repository)

You can use the model directly from the model repository to compute sentence embeddings:
```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Queries we want embeddings for
queries = ['What is the capital of France?', 'How many people live in New York City?']

# Passages that provide answers
passages = ['Paris is the capital of France', 'New York City is the most populous city in the United States, with an estimated 8,336,817 people living in the city, according to U.S. Census estimates dating July 1, 2019']

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModel.from_pretrained("model_name")

def compute_embeddings(sentences):
	#Tokenize sentences
	encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

	#Compute query embeddings
	with torch.no_grad():
		model_output = model(**encoded_input)

	#Perform pooling. In this case, mean pooling
	return mean_pooling(model_output, encoded_input['attention_mask'])

query_embeddings = compute_embeddings(queries)
passage_embeddings = compute_embeddings(passages)
```

## Usage (Sentence-Transformers)
Using this model becomes more convenient when you have [sentence-transformers](https://github.com/UKPLab/sentence-transformers) installed:
```
pip install -U sentence-transformers
```

Then you can use the model like this:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')

# Queries we want embeddings for
queries = ['What is the capital of France?', 'How many people live in New York City?']

# Passages that provide answers
passages = ['Paris is the capital of France', 'New York City is the most populous city in the United States, with an estimated 8,336,817 people living in the city, according to U.S. Census estimates dating July 1, 2019']

query_embeddings = model.encode(queries)
passage_embeddings = model.encode(passages)
```

## Changes in v3
The models from v2 have been used for find for all training queries similar passages. An [MS MARCO Cross-Encoder](ce-msmarco.md) based on the electra-base-model has been then used to classify if these retrieved passages answer the question.

If they received a low score by the cross-encoder, we saved them as hard negatives: They got a high score from the bi-encoder, but a low-score from the (better) cross-encoder.

We then trained the v2 models with these new hard negatives.

## Citing & Authors
If you find this model helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084):
``` 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```