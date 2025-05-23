"Implements layer freezing and Defines network architecture, associated parameters"
from torch import nn

class ModelArch(nn.Module):
    "Class to define the basic network to be utilized for tuning"
    def __init__(self, transformer_model, freeze_percent):
        super().__init__()
        self.model = transformer_model

        "Calculate and freeze layers based on Elsa research"
        layers = len(self.model.encoder.layer)
        frozen_layers = int(layers - ((layers * freeze_percent)/100) - 1)
        print('Froze layers', frozen_layers, 'of', layers)
        for layer in self.model.encoder.layer[:frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False

		# dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,5)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        "Forward pass on the inputs"
        _, cls_hs = self.model(sent_id, attention_mask=mask, return_dict=False)
        output = self.fc1(cls_hs)
        output = self.relu(output)
        output = self.dropout(output)

        #Final Layer
        output = self.fc2(output)
        output = self.softmax(output)
        return output
