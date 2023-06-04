from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from flask import Flask, jsonify, request

model = "tiiuae/falcon-7b-instruct"

print("Loading pipeline...")
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
print("Pipeline loaded")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prompt = data['inputs']
    #parameters = data['parameters']
    sequences = pipeline(prompt,
                        max_length=200,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        )
    return jsonify({"result": sequences})

if __name__ == "__main__":
    app.run()