import pandas as pd
import torch
from transformers import BertTokenizer
from src.models.bert.bert_cased_classifier import BertForCasedClassification
from src.constants.constant import DEVICE

# Path to your cased models
ANGER_MODEL_PATH   = r"C:\tweets-emotion-intensity-classification-master\anger_cased.pt"
FEAR_MODEL_PATH    = r"C:\tweets-emotion-intensity-classification-master\fear_cased.pt"
JOY_MODEL_PATH     = r"C:\tweets-emotion-intensity-classification-master\joy_cased.pt"
SADNESS_MODEL_PATH = r"C:\tweets-emotion-intensity-classification-master\sadness_cased.pt"

# Load the cased tokenizer
bert_tokenizer_cased = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def load_model(model_path):
    """Loads a saved BertForCasedClassification model from model_path."""
    model = BertForCasedClassification(dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Pre-load each emotion model (you could also load on-demand, but this is simpler)
anger_model   = load_model(ANGER_MODEL_PATH)
fear_model    = load_model(FEAR_MODEL_PATH)
joy_model     = load_model(JOY_MODEL_PATH)
sadness_model = load_model(SADNESS_MODEL_PATH)

def predict_cased():
    # Path to your test samples
    csv_path = r"C:\tweets-emotion-intensity-classification-master\test_samples.csv"
    df = pd.read_csv(csv_path)

    predictions = []
    for _, row in df.iterrows():
        text = row["Clean_Tweet"]  # use "Clean_Tweet" for cased model if you prefer
        emotion = row["Affect Dimension"]  # which model to use?

        # Tokenize
        encoded_dict = bert_tokenizer_cased.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_dict["input_ids"].to(DEVICE)
        attention_mask = encoded_dict["attention_mask"].to(DEVICE)

        # Select the correct model
        if emotion == "anger":
            model = anger_model
        elif emotion == "fear":
            model = fear_model
        elif emotion == "joy":
            model = joy_model
        elif emotion == "sadness":
            model = sadness_model
        else:
            # If the row has an unexpected "Affect Dimension", skip or handle differently
            predictions.append({"Predicted Intensity": None})
            continue

        # Inference
        with torch.no_grad():
            probas = model(input_ids, attention_mask)
            # Predicted class index (0..3 => intensities: NONE, LIGHT, MEDIUM, HIGH)
            pred_intensity = torch.argmax(probas, dim=1).item()

        predictions.append({"Predicted Intensity": pred_intensity})

    # Attach predictions to original DataFrame
    pred_df = pd.concat([df, pd.DataFrame(predictions)], axis=1)

    # Save to a new CSV
    out_csv = r"C:\tweets-emotion-intensity-classification-master\test_samples_predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv}")

if __name__ == "__main__":
    predict_cased()
