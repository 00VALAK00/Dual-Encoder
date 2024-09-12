import tqdm
from config.load_logger import logger, console_logger
import torch
from utils.loss import CrossEntropyLoss
import numpy as np


def train(model,device, epochs=20, criterion=CrossEntropyLoss(), data_loader=None, lr=1e-5):
    num_batches = len(data_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    model.train()
    console_logger.info(f"Training begins for {epochs} epochs")
    for epoch in range(epochs):
        console_logger.info(f'Epoch {epoch + 1}/{epochs}')
        running_loss = []

        for batch, (q, a) in enumerate(tqdm.tqdm(data_loader)):

            q = {k: v.to(device) for k, v in q.items()}
            a = {k: v.to(device) for k, v in a.items()}

            optimizer.zero_grad()

            _, _, similarity_scores = model(q, a)
            if torch.isnan(similarity_scores).any():
                logger.error(ValueError("Nan values detected in similarity_scores."))
                raise ValueError("Nan values detected in similarity_scores.")
            loss = criterion(similarity_scores)

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            if batch % 10 == 0:
                logger.info(f"Epoch {epoch+1}, loss = {np.mean(running_loss)}")