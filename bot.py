import random
import json
import torch
from model import LLM
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

with open('LLM.json', 'r') as f:
    intents = json.load(f)

FILE = "results.pth"
results = torch.load(FILE)

INPUT_SIZE = results["input size"]
OUTPUT_SIZE = results["output size"]
HIDDEN_UNITS = results["hidden units"]
all_words = results['all words']
tags = results['tags']

model = LLM(INPUT_SIZE,
            HIDDEN_UNITS,
            OUTPUT_SIZE).to(device)
model.eval()

bot_name = "BabyBlues"
print("Let's Chat!")
while True:
  sentence = input("You: ")
  if sentence == "quit":
    break

  sentence = tokenize(sentence)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)

  y_preds = model(X)
  _, predicted = torch.max(y_preds, dim=1)

  tag = tags[predicted.item()]

  probs = torch.softmax(y_preds, dim=1)
  prob = probs[0][predicted.item()]
  if prob.item() > 0.3:
    for intent in intents['intents']:
      if tag == intent['tag']:
        print(f"{bot_name}: {random.choice(intent['responses'])}")
  else:
    print(f"{bot_name}: Could you elaborate more?")