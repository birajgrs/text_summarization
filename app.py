from flask import Flask, render_template, request
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

app = Flask(__name__)

# Initialize the pretrained model and tokenizer
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        inputtext = request.form["inputtext_"]
        input_text = "summarize: " + inputtext

        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ids = model.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template("output.html", data={"input_text": inputtext, "summary": summary})

if __name__ == '__main__':
    app.run()
