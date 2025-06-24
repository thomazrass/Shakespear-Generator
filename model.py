import torch
import torch.nn as nn
from torch.nn import functional as F
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


# hyperparameters
batch_size = 256 # how many independent sequences will we process in parallel?
block_size = 200 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 100
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 186
n_head = 6
n_layer = 5
dropout = 0.3
early_stop_patience = 2500
use_word_tokenization = False
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if use_word_tokenization :
    text = text.replace('\n', '__NEWLINE__')

    # here are all the unique words that occur in this text
    words = word_tokenize(text)
    vocab = sorted(set(words))
    vocab_size = len(vocab)
    # create a mapping from words to integers
    stoi = { w:i for i,w in enumerate(vocab) }
    itos = { i:w for i,w in enumerate(vocab) }
    encode = lambda wlist: [stoi[w] for w in wlist]   # list of words → list of ints
    decode = lambda ilist: ' '.join([itos[i] for i in ilist])  # list of ints → sentence

    # Train and test splits
    data = torch.tensor(encode(words), dtype=torch.long)

else:
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # create 'batch size' random numbers between 0 and len-block_size. These are the starting points of each sample
    # len-block_size because we dont want a sample to not have enough characters when training, so max index we can take the one that is 'block_size' away from the end
    
    #  size=(batch_size,) means 1D. Equivalent to (batch_size, 0, 0)
    ix = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix]) # x.shape == (64, 256)

    # same as x, but shifted 1 position. This represents the next characters
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y.shape == (64, 256)

    x, y = x.to(device), y.to(device)
    return x, y

# averages loss over multiple batches
@torch.no_grad()
def estimate_loss():
    out = {}
    # print("1")

    # put  in eval mode: disables, dropout, batch norm, etc. (don't want randomness in evaluation)
    model.eval()
    # print("2")
    for split in ['train', 'val']:
        print("evaluating ", split)
        losses = torch.zeros(eval_iters)
        
        # test eval on multiple batches to avoid a random "good" or "bad" eval
        for k in range(eval_iters):
            print("k: ", k)
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        # take eval mean of all the batches
        out[split] = losses.mean()

    # put back into train mode
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # “How much does token A (query) care about token B (key)? If a lot, take B's value.”
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # creates a lower triangular matrix of shape (block_size, block_size)
        # Used to prevent tokens from looking ahead in the sequence
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape

        # q[i, t]: what token t in sequence i wants
        # k[i, t']: what each token t' offers
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # compute attention scores ("affinities") 
        # also normalize to avoid peaky softmax. Without normalizing, the bigger the values, 
        # the more softmax converges towards the largest value. This means that it willl only retain information 
        # form the largest number token, but we want it to retain info from every token (Also from the paper)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T), # Scale by √(head_size)

        # Mask out future tokens (causal attention)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        #Now each row is a probability distribution — what each token should attend to.
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        # Each token’s new representation is a weighted average of the values from all tokens it attended to.
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        #lets the model mix the information from all heads in a learnable way.
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-lineary
        FeedForward is for transforming the information at each token.
        Always comes after attention layer
        This is where all of the actual math is done
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # times 4 beacuse it says in the paper
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),

            # projection layer going back into the residual pathway
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation 
        Attention is All You Need
    """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        # how much work each head will do
        head_size = n_embd // n_head

        # creates multi-head attention with n_heads nb of heads all with size head_size
        self.sa = MultiHeadAttention(n_head, head_size)

        # fully connected MLP applied to each token individually
        self.ffwd = FeedForward(n_embd)

        # layer normalization from paper
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # + becasue we are doing skip connections, or residual connections
        #   fork off, do connections, and then add back to x. Good for optimization
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    """
    The whole architecture is absed on the transformers paper. It just does not include the encoder part.
    """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table

        # creates an embedding table of size vocab_size x n_embd
        # every row is a token or word ( so there are vocab_size number of rows)
        # every row has size n_embd
        # turns word IDs → vectors that the model can learn from.
        # REPRESENTS WHAT THE WORD IS
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # create an embedding table for the position of every toke. Since the size of every batch is block size,
        # creates an embedding table of size block_size x n_embd
        # every row is a token or word ( so there are block_size number of rows)
        # every row has size n_embd
        # REPRESENTS WHERE THE WORD IS IN THE SEQUENCE
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # This creates a list of n_layer number of blocks.
        # [Block(...), Block(...), Block(...), Block(...), Block(...), Block(...)]
        # * is unpacking the list of Blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
       
        # idx represents:
        #   a batch of B sequences
        #   each sequence has T word indices

        # idx =
        #     [
        #     [14, 2, 7, 32, ...],   # sequence 1
        #     [9,  6, 4, 0,  ...],   # sequence 2
        #     ...
        #     ]

        # idx and targets are both (B,T) tensor of integers

        # Each element in idx gets replaced with its embedding vector
        # every token of every sequence in the batch is replaces with the token embedding, so it adds a dimension C = n_embd
        tok_emb = self.token_embedding_table(idx) # (B,T,C)

        # T = block_size
        # torch.arange(T, device=device) returns [0, 1, 2, ..., T-1]
        # replace every position with its positional embedding vector
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        # Add up to get a unique embedding for every word.
        # ["The", "cat"] will get a different embedding than ["cat", "The"]
        x = tok_emb + pos_emb # (B, T, C)

        # here is where the embeddings layers learn
        x = self.blocks(x) # (B, T, C)

        # a final layer normalization
        x = self.ln_f(x) # (B, T, C)
        
        # maps from the embedding space (C) to vocabulary logits. nn.Linear is a continuous mapping
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:

            # F.cross_entropy() expects: (check functional cross entropy documentation)
                # logits: shape (N, vocab_size)
                # targets: shape (N,) (with each value being a class index)

            # so reshape
            B, T, C = logits.shape

            # since were reshaping boths in the same way it doesnt matter. They will all be in the right positions

            # .view() is used to reshape a tensor without changing its data.
            # “Hey PyTorch, treat this data as if it had a different shape.”
            # nb of elements must match before and after calling .view()
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets, label_smoothing=0.05)

        return logits, loss

    # "predict function"
    def generate(self, idx, max_new_tokens):
        """returns a new tensor with the original context + max_new_tokens newly generated tokens."""
        # idx is (B, T) array of indices in the current context
        
        # run this max_new_tokens number of times to generate that number of new tokens
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens (otherwise it will break the embedding table)
            # we do this becasue we just want to look at the last block_size number of tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions (run forward)
            logits, loss = self(idx_cond) #logits.shape = (B, T, vocab_size)

            # focus only on the last time step since we're generating the next one
            logits = logits[:, -1, :] # becomes (B, vocab_size) basically (nb_batches x logits for every word in the vocab) for the last time step

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)

            # sample from the distribution
            # Randomly selects one word index from the probability distribution for each sequence in the batch.
            # multinomial adds randomness (better for creativity)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    
if __name__ == "__main__":
    # put your training loop here
    model = BigramLanguageModel()
    # train the model and save it

    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # implement early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    steps = []


    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            print("yo")
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Save the loss values and step
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            steps.append(iter)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement % 100 == 0:
                print(f"--> No improvement. Patience: {epochs_without_improvement}/{early_stop_patience}")

        if epochs_without_improvement >= early_stop_patience:
            print("\nEarly Stopping triggered.\n")
            break

        # sample a batch of data
        xb, yb = get_batch('train')
        #print("1")
        # evaluate the loss (calls forward)
        logits, loss = model(xb, yb)
        #print("2")
        optimizer.zero_grad(set_to_none=True)
        #print("3")
        # compute gradients
        loss.backward()
        #print("4")
        # update weigths
        optimizer.step()
        #print("5")

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(steps, val_losses, label='Validation Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")  # Save to file (optional)
    plt.show()


    print("Saving model...")
    torch.save(model.state_dict(), 'bigram_model_word.pt')


    # Recreate the model structure
    model = BigramLanguageModel()
    model.load_state_dict(torch.load('bigram_model_word.pt'))
    model.to(device)
    model.eval()  # Make sure dropout is off

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
    print(output.replace('__NEWLINE__', '\n'))
