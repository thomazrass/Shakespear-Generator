from model import BigramLanguageModel
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Recreate the model structure
model = BigramLanguageModel()
state_dict = torch.load("bigram_model_word.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # Make sure dropout is off

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = decode(model.generate(context, max_new_tokens=10000)[0].tolist())
print(output.replace('__NEWLINE__', '\n'))
