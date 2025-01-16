from flask import Flask, request, jsonify
import torch
import os
import csv
import random
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################################
# 1) Trieda pre "emotion only" so 6 logitmi
############################################################
class BertForEmotionOnly(nn.Module):
    """
    BERT classifier so 6 logits:
    Indexy: 0->sadness,1->joy,2->love,3->anger,4->fear,5->surprise
    """
    def __init__(self):
        super(BertForEmotionOnly, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        # 6 výstupov pre 6 tried: sadness, joy, love, anger, fear, surprise
        self.ffn = nn.Linear(self.bert.config.hidden_size, 6)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]

        x = self.dropout(pooled_output)
        x = self.ffn(x)  # [batch_size, 6]
        return x

############################################################
# 2) Načítať checkpoint (6 tried)
############################################################
emotion_model_path = r"C:\Users\slavo\Desktop\SpanEmo-master\models\2024-12-11-14-56-47_checkpoint.pt"
if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"Checkpoint {emotion_model_path} neexistuje.")

emotion_model = BertForEmotionOnly()

checkpoint = torch.load(emotion_model_path, map_location=device)
# Očakávame 'ffn.weight' s tvarom [6,768], 'ffn.bias' [6]
emotion_model.load_state_dict(checkpoint['model_state_dict'])  # strict=True
emotion_model.to(device).eval()

# 6 tried v presnom poradí: 
label_names = ["sadness","joy","love","anger","fear","surprise"]

# Tokenizer (uncased)
tokenizer_main = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def classify_emotion_6(text: str) -> str:
    """
    Vráti jednu z: 
      "sadness","joy","love","anger","fear","surprise"
    """
    encoded = tokenizer_main.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        logits = emotion_model(input_ids, attention_mask=attention_mask)
        pred_idx = torch.argmax(logits, dim=1).item()

    if 0 <= pred_idx < len(label_names):
        return label_names[pred_idx]
    return "none"

############################################################
# 3) Načítame 4 cased checkpointy na INTENZITU (sadness,joy,anger,fear)
############################################################
from src.models.bert.bert_cased_classifier import BertForCasedClassification

anger_model_path   = r"C:\Users\slavo\Desktop\tweets-emotion-intensity-classification-master\anger_cased.pt"
fear_model_path    = r"C:\Users\slavo\Desktop\tweets-emotion-intensity-classification-master\fear_cased.pt"
joy_model_path     = r"C:\Users\slavo\Desktop\tweets-emotion-intensity-classification-master\joy_cased.pt"
sadness_model_path = r"C:\Users\slavo\Desktop\tweets-emotion-intensity-classification-master\sadness_cased.pt"

for pth in [anger_model_path, fear_model_path, joy_model_path, sadness_model_path]:
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Checkpoint {pth} neexistuje.")

anger_model   = BertForCasedClassification(dropout=0.1).to(device).eval()
fear_model    = BertForCasedClassification(dropout=0.1).to(device).eval()
joy_model     = BertForCasedClassification(dropout=0.1).to(device).eval()
sadness_model = BertForCasedClassification(dropout=0.1).to(device).eval()

anger_model.load_state_dict(torch.load(anger_model_path,   map_location=device))
fear_model.load_state_dict(torch.load(fear_model_path,     map_location=device))
joy_model.load_state_dict(torch.load(joy_model_path,       map_location=device))
sadness_model.load_state_dict(torch.load(sadness_model_path, map_location=device))

# cased tokenizer
tokenizer_cased = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def get_intensity_prediction(model, text: str) -> int:
    """
    Vráti intensitu 0..3 pre danú emóciu.
    """
    encoded_dict = tokenizer_cased.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_dict["input_ids"].to(device)
    attention_mask = encoded_dict["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, masks=attention_mask)
        pred_intensity = torch.argmax(logits, dim=1).item()  # 0..3
    return pred_intensity

############################################################
# 4) Logika: 
#    - Ak emócia je "sadness","joy","anger","fear" => predikuj intensitu
#    - Ak "love" či "surprise" => intensita = 0 (alebo vynechaj).
############################################################
def predict_emotion_and_intensity(text: str):
    emo = classify_emotion_6(text)  # 6-class model

    if emo == "sadness":
        intens = get_intensity_prediction(sadness_model, text)
    elif emo == "joy":
        intens = get_intensity_prediction(joy_model, text)
    elif emo == "anger":
        intens = get_intensity_prediction(anger_model, text)
    elif emo == "fear":
        intens = get_intensity_prediction(fear_model, text)
    else:
        # "love" or "surprise" or "none"
        intens = 0

    return emo, intens

############################################################
# 5) Flask route
############################################################
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    txt = data.get('text', '')
    if not txt.strip():
        return jsonify({"error": "No text provided"}), 400

    emotion, intensity = predict_emotion_and_intensity(txt)
    return jsonify({
        "emotion": emotion,
        "intensity": intensity
    }), 200


@app.route('/get_random_sentence', methods=['GET'])
def get_random_sentence():
    csv_path = r'C:\Users\slavo\Desktop\SpanEmo-master\data\text.csv'
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            sentences = [row['text'] for row in reader]
            if not sentences:
                return jsonify({"error": "No sentences found in CSV"}), 404
            random_sentence = random.choice(sentences)
            return jsonify({"sentence": random_sentence}), 200
    except FileNotFoundError:
        return jsonify({"error": "CSV file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

############################################################
# 6) Spustenie
############################################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
