import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Load data from Excel
def load_data_from_excel(file_path, data = 1):
    df = pd.read_excel(file_path, engine='openpyxl')  # Ensure openpyxl is installed
    x_data = df.iloc[2:, 0].values.astype(float)
    y_data = df.iloc[2:, data].values.astype(float)
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)

# Custom Gaussian equation model (adjust this to match your equation)
class ComplexGaussianModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize all parameters 
        # self.a  = torch.nn.Parameter(torch.randint(1, 5, (1,)).float())
        # self.w  = torch.nn.Parameter(torch.randint(60, 70, (1,)).float())
        self.c1 = torch.nn.Parameter(torch.randint(100, 200, (1,)).float())
        self.c2 = torch.nn.Parameter(torch.randint(10, 20, (1,)).float())
        # self.c3 = torch.nn.Parameter(torch.randint(10, 20, (1,)).float())
        # self.c4 = torch.nn.Parameter(torch.randint(10, 20, (1,)).float())
        self.t1 = torch.nn.Parameter(torch.randint(20, 30, (1,)).float())
        self.t2 = torch.nn.Parameter(torch.randint(20, 30, (1,)).float())
        # self.t3 = torch.nn.Parameter(torch.randint(20, 30, (1,)).float())
        # self.t4 = torch.nn.Parameter(torch.randint(20, 30, (1,)).float())
        # self.x0 = torch.nn.Parameter(torch.randint(50, 60, (1,)).float())
        self.y0 = torch.nn.Parameter(torch.randint(1, 5, (1,)).float())
    
    def forward(self, x):
        # Example: Sum of three Gaussian terms (modify to match your equation)
        w=50
        x0=54
        coef1 = self.c1*w*torch.sqrt(torch.tensor(np.pi))/2
        coef2 = self.c2*w*torch.sqrt(torch.tensor(np.pi))/2
        # coef3 = self.c3*self.w*torch.sqrt(torch.tensor(np.pi))/2
        # coef4 = self.c4*self.w*torch.sqrt(torch.tensor(np.pi))/2
        exp1 = torch.exp(torch.clamp((w**2 - 4*self.t1*(x-x0))/(4*self.t1**2), min=-50, max=50))
        exp2 = torch.exp(torch.clamp((w**2 + 4*self.t2*(x-x0))/(4*self.t2**2), min=-50, max=50))
        # exp3 = torch.exp(torch.clamp((self.w**2 - 4*self.t3*(x-x0))/(4*self.t3**2), min=-50, max=50))
        # exp4 = torch.exp(torch.clamp((self.w**2 - 4*self.t4*(x-x0))/(4*self.t4**2), min=-50, max=50))
        erf1 = torch.erf(torch.clamp((w**2 - 2*self.t1*(x-x0))/(2*self.t1*w), min=-5, max=5))
        erf2 = torch.erf(torch.clamp((w**2 + 2*self.t2*(x-x0))/(2*self.t2*w), min=-5, max=5))
        # erf3 = torch.erf(torch.clamp((self.w**2 - 2*self.t3*(x-x0))/(2*self.t3*self.w), min=-5, max=5))
        # erf4 = torch.erf(torch.clamp((self.w**2 - 2*self.t4*(x-x0))/(2*self.t4*self.w), min=-5, max=5))

        # return self.y0 - coef1*exp1*(erf1-1) - coef2*exp2*(erf2-1) - coef3*exp3*(erf3-1) - coef4*exp4*(erf4-1) 
        # return self.y0 - coef1*exp1*(erf1-1) - coef2*exp2*(erf2-1) - coef3*exp3*(erf3-1)
        return self.y0 - coef1*exp1*(erf1-1) - coef2*exp2*(erf2-1)


# Load data
x_data, y_data = load_data_from_excel("150uw70uw.xlsx", 4)
# print(y_data)
# Ensure data is in the correct shape [num_samples, ]
x_data = x_data.view(-1)
y_data = y_data.view(-1)

# Initialize model, optimizer, and loss
model = ComplexGaussianModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, eps=1e-8)
loss_fn = torch.nn.MSELoss()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

epoch = 0
# Training loop
while True:
    optimizer.zero_grad()
    y_pred = model(x_data)
    # print(y_pred)
    loss = loss_fn(y_pred, y_data)
    loss.backward()
    optimizer.step()
    # print(f"Loss: {loss.item()}")
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    if loss.item() < 1: 
        break

    epoch += 1

# Print final parameters
print("\nFitted parameters:")
for name, param in model.named_parameters():
    if name in ['log_a', 'log_w', 'log_x0', 'log_t1', 'log_t2', 'log_t3', 'log_t4']:
        print(f"{name}: {np.exp(param.item())}")
        # print("detected")
    else:
        print(f"{name}: {param.item()}")