"Trainer module for training and evaluation"
import torch
import numpy as np
from torch import nn
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from config.config_handler import handler

class Trainer:
    "Trainer class with related functions"
    def __init__(self, model, train_data, train_labels, val_data):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.val_data = val_data

        #Move model to appropriate device GPU/CPU
        self.model = self.model.to(self.device)

        #Optimizer to be used for updates
        self.optimizer = AdamW(self.model.parameters(), lr = 1e-3)

        #Dynamically compute the class weights to handle data imbalance
        class_wts = compute_class_weight('balanced', np.unique(self.train_labels),\
         self.train_labels)

		#Convert class weights to tensor
        weights= torch.tensor(class_wts,dtype=torch.float)
        weights = weights.to(self.device)

        #Define Loss function
        self.cross_entropy  = nn.NLLLoss(weight=weights)

    def train(self):
        "function to train the model"
        self.model.train()
        total_loss = 0

        total_preds=[]

        for step,batch in enumerate(self.train_data):
            if step % 50 == 0 and not step == 0:
                print('Batch:',step,'of ',len(self.val_data))
            # push the batch to gpu
            batch = [r.to(self.device) for r in batch]

            sent_id, mask, labels = batch

            # clear previously calculated gradients
            self.optimizer.zero_grad()
			#Predictions for the current batch
            preds = self.model(sent_id, mask)

			#Compute the loss
            loss = self.cross_entropy(preds, labels)

			#Update total loss
            total_loss = total_loss + loss.item()

			#Backward pass to calculate the gradients
            loss.backward()

			#Clip the the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

			#Update parameters
            self.optimizer.step()

			#Move model predictions to CPU
            preds = preds.detach().cpu().numpy()

			#Update all preds
            total_preds.append(preds)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(self.train_data)

        # reshape the predictions for ease of understanding
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def evaluate(self):
        "Function for evaluating the model"
        self.model.eval()
        total_loss = 0
        total_preds = []

        #Iterate over batches for Evaluation
        for step,batch in enumerate(self.val_data):
            #Progress update after every 100 batches
            if step % 100 == 0 and not step == 0:
                print('Batch:',step,'of ',len(self.val_data))
            batch = [t.to(self.device) for t in batch]
            sent_id, mask, labels = batch
            #Deactivate autograd
            with torch.no_grad():
                # model predictions
                preds = self.model(sent_id, mask)
                # compute the validation loss between actual and predicted values
                loss = self.cross_entropy(preds,labels)
                total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)
        #Compute the validation loss of each epoch
        avg_loss = total_loss / len(self.val_data)
        #Reshape the predictions
        total_preds  = np.concatenate(total_preds, axis=0)
        return avg_loss, total_preds

    def execute(self):
        "Train and evaluate for defined epochs"
        best_valid_loss = float('inf')
        train_losses=[]
        valid_losses=[]

        for epoch in range(handler('settings','epochs')):
            print('Training epoch:',epoch, 'of', handler('settings','epochs'))
            #Train
            train_loss, _ = self.train()
            #Evaluate
            valid_loss, _ = self.evaluate()
            #Save Model if better performance seen
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'saved_weights.pt')
            #Record training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

    def save_model(self):
        "load weights of model for later use"
        path = 'saved_weights.pt'
        self.model.load_state_dict(torch.load(path))
