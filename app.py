# Import necessary libraries and modules
from flask import Flask, render_template, request
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

# Create a Flask web application instance
app = Flask(__name__)

# Initialize the pretrained model and tokenizer for text summarization
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a route for the home page ("/")
@app.route('/')
def home():
    # Render an HTML template named 'index.html' when accessing the home page
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    # Check if the incoming request is a POST request
    if request.method == "POST":
        # Get the input text from the HTML form named "inputtext_"
        inputtext = request.form["inputtext_"]
        
        # Prepend the input text with "summarize: " as required by the Pegasus model
        input_text = "summarize: " + inputtext

        # Tokenize the input text and move it to the appropriate device (GPU or CPU)
        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        
        # Generate a summary using the Pegasus model with a shorter max_length
        summary_ids = model.generate(tokenized_text, min_length=30, max_length=200)  
        
        # Decode the summary from token IDs to human-readable text
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Render an HTML template named 'output.html' and pass the input text and generated summary as data
    return render_template("output.html", data={"input_text": inputtext, "summary": summary})

# Start the Flask web application if this script is executed
if __name__ == '__main__':
    app.run()


# python app.py
