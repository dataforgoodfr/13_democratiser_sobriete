import torch
from argparse import ArgumentParser
from transformers import BertTokenizer, BertForSequenceClassification
from train import LitTextClassification


def main(text: str, checkpoint_path: str):

    # Load the model
    if checkpoint_path is not None:
        model = LitTextClassification.load_from_checkpoint(checkpoint_path, map_location="cpu")
    else:
        model = LitTextClassification()
        # Use the pretrained weights from HuggingFace
        model.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    model.eval()

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Encode the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get the model output
    with torch.no_grad():
        outputs = model.model(**inputs)

    # The output logits
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()

    class_labels = {
        0: "negative",
        1: "positive",
    }

    # Print the results
    print("\n\n")
    print(f"Text: {text}")
    print(f"Output probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class}")
    print(f"Predicted class label: {class_labels[predicted_class]}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="Text to classify", 
        default="This is an amazing movie! So much depth and emotion. I loved it!"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the checkpoint file",
        default="lightning_logs/version_0/checkpoints/epoch=2-step=4689.ckpt"
    )
    args = parser.parse_args()
    main(args.input, args.checkpoint)
