import sys
import os

sys.path.append("/Users/benjawesome/coding/positional-gpt-2")

import pickle
import torch
import torch.optim as optim
from models.gpt_implementation import GPT2Model
from data.byte_pair_encoding import BytePairTokenizer
from data.dataloader import GPTLoader
from datasets import load_dataset

save_dir = os.path.join("..", "files")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "my_object.pkl")

# define hyperparameters

NUM_EPOCHS = 5
LR = 0.0003
BATCH_SIZE = 32
D_MODEL = 256
MAX_LEN = 512
VOCAB_SIZE = 5001
N_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1   
D_FF = D_MODEL * 4
WEIGHT_DECAY = 0.01

def main():
    """
    Run and train the model
    """
    # get device
    device = torch.device("mps")

    # need to import our tokenizer
    tokenizer = BytePairTokenizer.load(save_path)

    print("tokenizer saved!")

    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    # get our data
    train_text_list = raw_datasets["train"]["text"] 
    val_text_list = raw_datasets["validation"]["text"]

    train_text_str = "\n".join(train_text_list)
    val_text_str = "\n".join(val_text_list)

    print("loaded data")
    
    # encode our data to prepare to run through model
    encoded_train = tokenizer.encode(train_text_str)
    encoded_val = tokenizer.encode(val_text_str)

    print("encoded data")

    # save each list as a file
    train_path = '../files/encoded_train.pt'
    val_path = '../files/encoded_val.py'

    train_tensor = torch.tensor(encoded_train, dtype=torch.long)
    val_tensor = torch.tensor(encoded_val, dtype=torch.long)

    torch.save(train_tensor, train_path)
    print("Saved encoded training set")
    torch.save(val_tensor, val_path)
    print("Saved encoded validation set")

    # need to first create an instance of the model
    model = GPT2Model(VOCAB_SIZE, N_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT, device, MAX_LEN)
    model = model.to(device)

    # create an instance of our dataloader
    train_dataset = GPTLoader(train_path, MAX_LEN)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # define our optimizer
    optimizer = optim.AdamW(
    model.parameters(), 
    lr=LR, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999), 
    eps=1e-8             
    )

    print("training begun")

    train(model, 5, train_dataloader, optimizer, device)
    print("model trained)")

def train(model, num_epochs, train_loader, optimizer, device):
    # loop for however many epochs
    for epoch in range(num_epochs):
        model.train()

        # for epoch statistics
        total_loss = 0
        count = 0

        # for step statistics
        running_loss = 0
        step_count = 0

        # first tokenize every input batch
        for input, target in train_loader:
            input = input.to(device)
            target = target.to(device)

            logits, loss = model(input, target)

            # do some statistics calculations
            running_loss += loss.item()
            step_count += 1
            total_loss += loss.item()
            count += 1

            # update gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step_count == 100):
                avg_loss = running_loss / step_count
                print(f"Epoch: {epoch} | Average loss for step {step_count}:  {avg_loss:.2f}")

                # reset
                running_loss = 0
                step_count = 0
        
        epoch_avg_loss = total_loss / count
        print(f"Epoch {epoch + 1} average loss:  {epoch_avg_loss:.2f}")

        # reset 
        total_loss = 0
        count = 0

if __name__ == "__main__":
    main()