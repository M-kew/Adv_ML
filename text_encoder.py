import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Agent(nn.Module):
    def __init__(self, num_actions=10):
        super(Agent, self).__init__()
        
        # Load pre-trained sentence embedding model from Hugging Face
        # Example model: 'sentence-transformers/all-MiniLM-L6-v2'
        self.encoder_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_name)
        self.encoder = AutoModel.from_pretrained(self.encoder_model_name)
        
        # Freeze the encoder weights such that the weights are not updated during training
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Define the classifier
        self.classifier = nn.Sequential(
            # Classifier configuration: follows teh architecture of the report
            # input dim 1024 --> RELU --> intermediate layer dim 1024 --> RELU --> input dim 1024 & output dim is based on the number of actions, which is 10
            nn.Linear(self.encoder.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)  # Output size equals number of actions
        )
    
    def forward(self, input_text):
        # Tokenize input text
        inputs = self.tokenizer(
            input_text, return_tensors='pt', padding=True, truncation=True
        )
        
        with torch.no_grad():
            # Obtain encoder outputs
            outputs = self.encoder(**inputs)
            # Use the [CLS] token's embedding as the sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Pass through the classifier
        logits = self.classifier(cls_embedding)
        
        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Sample an action from the categorical distribution
        action_distribution = torch.distributions.Categorical(probabilities)
        action = action_distribution.sample()
        
        return action, probabilities

# Example usage
if __name__ == "__main__":
    agent = Agent(num_actions=10)
    input_text = ["How to produce a fake check?"]
    action, probabilities = agent(input_text)
    print(f"Selected action: {action.item()}")
    print(f"Action probabilities: {probabilities.detach().numpy()}")
