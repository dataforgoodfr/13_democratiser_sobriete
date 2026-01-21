import dspy
from dspy.teleprompt import MIPROv2

import json
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import torch
from datetime import datetime


# Load environment variables
load_dotenv()

# 1. Configuration
### Scaleway  API
"""
lm = dspy.LM(
        model="mistral/mistral-small-3.2-24b-instruct-2506:fp8",
        api_key=os.getenv('SCALEWAY_API_KEY'),
        api_base="https://c1b66caa-347e-448c-a54c-d3fb43889a62.ifr.fr-par.scaleway.com"
        )
"""
### OpenAI  API
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))


dspy.configure(lm=lm)
model_used = lm.model.replace("/","_")
# 2. Data Loading
golden_dataset = []

if os.path.exists('model_training_data/conclusions&pollitiques_gold.jsonl'):
    with open('model_training_data/conclusions&pollitiques_gold.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            example = dspy.Example(question=data['question'], response=data['response'])
            golden_dataset.append(example.with_inputs('question'))
else:
    exit ("Data file 'model_training_data/conclusions&pollitiques_gold.jsonl' not found.")

syntetic_dataset = []
if os.path.exists('model_training_data/conclusions&pollitiques_synthetiques_diversifies.jsonl'):
    with open('model_training_data/conclusions&pollitiques_synthetiques_diversifies.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            example = dspy.Example(question=data['question'], response=data['response'])
            syntetic_dataset.append(example.with_inputs('question'))
else:
    exit ("Data file 'model_training_data/conclusions&pollitiques_synthetiques_diversifies.jsonl' not found.")

# Meilleur score avec dataset synthetique petit  
trainset = syntetic_dataset
devset = golden_dataset

print(f"Training examples: {len(trainset)}, Validation examples: {len(devset)}")

# 3. Signature
class GenerationReponse(dspy.Signature):
    """Extract policy recommendations from research paper conclusions."""

    question = dspy.InputField(
        desc="A paragraph taken from the conclusion section of a research paper."
    )

    response = dspy.OutputField(
        desc=(
        "Extract all policy recommendations explicitly or implicitly mentioned "
        "in the paragraph. Return ONLY a semicolon-separated list of policies."
        "Do not include numbering, explanations, or extra text "
        "Format: 'policy1; policy2; policy3'"
    ),format="json"
)

# 4. Module
class MonProgramme(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerationReponse)
    
    def forward(self, question):
        return self.generate(question=question)


class CrossEncoderMetric:
    def __init__(self):
        ## Bon model pour textes uniquement en anglais cross-encoder/ms-marco-MiniLM-L-6-v2
        ## Bon model pour textes multilingues cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def __call__(self,example, pred, trace=None):
        """Metric function using a cross-encoder for semantic similarity evaluation"""
        
        gold_answer = example.response.strip()
        pred_answer = pred.response.strip()
        
        sentence_pairs = [(gold_answer, pred_answer)]
        logits = self.model.predict(sentence_pairs)
        
        score = torch.sigmoid(torch.tensor(logits))

        return  float(score[0])

metric_fn = CrossEncoderMetric()

print("Starting optimization...")

optimizer = MIPROv2(metric=metric_fn
                    #,auto="heavy"
                    )

compiled_program = optimizer.compile(MonProgramme(), trainset=trainset)

print("Evaluating optimized model on Dev Set...")
optimized_evaluator = dspy.Evaluate(
    devset=devset,
    metric=metric_fn,
    display_progress=False,
    display_table=True
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


optimized_score = optimized_evaluator(compiled_program,save_as_json=f"saved_dspy_model/policy/{model_used}{timestamp}.json")
print(optimized_score)

score_str = f"{round(optimized_score.score,2)}".replace(".", "_") 


optimized_evaluator(
    compiled_program
)

print(f"Final Score on Validation Set (optimized): {optimized_score}%")

# --- Saving the optimized model ---
model_path = f"saved_dspy_model/policy/{score_str}"
compiled_program.save(model_path,save_program=True)
print(f"Optimized model saved to {model_path}")
