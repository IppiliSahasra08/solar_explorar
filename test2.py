from ragas import evaluate
from datasets import Dataset
from ragas.metrics._faithfulness import Faithfulness

dataset = Dataset.from_dict({
    "user_input": ["What is solar power?"],
    "response": ["Solar power converts sunlight into electricity."],
    "retrieved_contexts": [["Solar panels convert sunlight into electricity."]],
    "reference": ["Solar power converts sunlight into electricity."]
})

result = evaluate(
    dataset,
    metrics=[Faithfulness()]
)

print(result)