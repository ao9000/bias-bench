import matplotlib.pyplot as plt
import json
import argparse
import os

# Argument parser to specify the model output json file
parser = argparse.ArgumentParser()
parser.add_argument('--model_output', type=str, required=True)
args = parser.parse_args()

# Load the model output json
model_logs = json.load(open(os.path.join(args.model_output, "trainer_state.json"), 'r'))

# Go to log_history
log_history = model_logs["log_history"][:-1] # Exclude last entry because it is the final train results

# Extract the loss and step values from the model output
losses = [d['loss'] for d in log_history]
steps = [d['step'] for d in log_history]
lr = [d['learning_rate'] for d in log_history]
grad_norm = [d['grad_norm'] for d in log_history]
epoch = [d['epoch'] for d in log_history]

# Plot loss vs. step
plt.figure(figsize=(8, 6))
plt.plot(steps, losses)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training loss per step')

# Save the plot
plt.savefig(os.path.join(args.model_output, "training_loss_graph.png"))

# Plot learning rate vs. step
plt.figure(figsize=(8, 6))
plt.plot(steps, lr)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate per step')

# Save the plot
plt.savefig(os.path.join(args.model_output, "learning_rate_graph.png"))
