# AI Optimizer for Node Configuration
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class NodeOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeOptimizer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Optimizer:
    def __init__(self, ai_engine_config):
        self.config = ai_engine_config
        self.device = ai_engine_config['device']
        self.model = NodeOptimizer(input_dim=10, hidden_dim=20, output_dim=5)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

    def train(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for epoch in range(self.config['num_epochs']):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device))
            loss = self.criterion(outputs, torch.tensor(y_train, dtype=torch.long).to(self.device))
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                outputs_val = self.model(torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device))
                loss_val = self.criterion(outputs_val, torch.tensor(y_val, dtype=torch.long).to(self.device))
                accuracy = accuracy_score(y_val, torch.argmax(outputs_val, dim=1).cpu().numpy())
                f1 = f1_score(y_val, torch.argmax(outputs_val, dim=1).cpu().numpy(), average='macro')
                print(f'Epoch {epoch+1}, Loss: {loss_val.item():.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    def predict(self, X_test):
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        with torch.no_grad():
            outputs = self.model(torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device))
            return torch.argmax(outputs, dim=1).cpu().numpy()
