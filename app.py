from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers, torch
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

parameters = {
    'max_length':1500,
    'do_sample':True,
    'top_k':10,
    'num_return_sequences':1,
    'eos_token_id': tokenizer.eos_token_id,
}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prompt = data['inputs']
    try:
        params = data['parameters']
        parameters.update((k, v) for k, v in params.items() if k in parameters)
    except:
        pass
    try:
        sequences = pipeline(prompt, **parameters)
        return jsonify({"result": sequences})
    except Exception as e:
        return jsonify({"result": e})

if __name__ == "__main__":
    app.run()