from flask import Flask ,render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask_cors import CORS, cross_origin
from html import entities
from nltk.corpus import stopwords
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizer, BertModel, BertForTokenClassification
import torch
import re

# Loading the saved BioBERT model
model_path = "D:\\Knowledge-Graph-using-NER---Relationship-Extraction-from-EHR\\trained-model"
tokenizer = BertTokenizer.from_pretrained(model_path, do_basic_tokenize=False)
model = BertForTokenClassification.from_pretrained(model_path)
label_map = model.config.id2label


def prescription(text):
    # Split ting the text into smaller chunks
    res=""
    chunk_size = 64
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            end = len(text)
        else:
            end = text.rfind(' ', start, end)
            if end == -1:
                end = start + chunk_size
        chunks.append(text[start:end])
        start = end + 1

    # Process each chunk separately
    entities = []
    for chunk in chunks:
        inputs = tokenizer.encode_plus(chunk, return_tensors='pt', max_length=512, truncation=True)

        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        predictions = torch.argmax(outputs.logits, dim=2)

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        entity = ''
        prev_label = ''
        for i, token in enumerate(predictions[0]):
            label = label_map[token.item()]
            if label.startswith('B-'):
                if entity:
                    entities.append((entity.replace('##', ''), prev_label))
                    entity = ''
                entity = tokenizer.decode([inputs['input_ids'][0][i]])
                prev_label = label[2:]
            elif label.startswith('I-'):
                entity += ' ' + tokenizer.decode([inputs['input_ids'][0][i]])
            else:
                if entity:
                    entities.append((entity.replace('##', ''), prev_label))
                    entity = ''
        if entity:
            entities.append((entity.replace('##', ''), prev_label))

        entities = [(entity.replace(' ', ''), label) for entity, label in entities]
    for entity, label in entities:
        entity = re.sub(r'\s+', '', entity) # remove spaces between subwords
        res += f"Entity '{entity}' has label '{label}'\n"
    return(res)

app = Flask(__name__,template_folder='template')
cors = CORS(app)

@app.route("/")
def main():
    return render_template("upload.html")


@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        fname=secure_filename(f.filename);
        f.save("static/"+secure_filename(f.filename))
        print(f.filename)
        new_path="static/"+fname;
        print("FILE PATH:  ")
        print(new_path)
        with open(new_path, "r") as f:
            text = f.read().strip()
        final=prescription(text);        
    return  render_template("upload.html",value = final)
     

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)