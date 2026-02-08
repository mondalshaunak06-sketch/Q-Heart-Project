import pennylane as qml
from pennylane import numpy as np

# --- 1. DATA SETUP ---
# Input Data: [Age, Cholesterol] scaled to radians (0 to Pi)
X_train = np.array([
    [0.2, 0.4],  # Patient A (Healthy)
    [2.8, 3.0],  # Patient B (High Risk)
    [0.1, 0.3],  # Patient C (Healthy)
    [2.5, 2.9]   # Patient D (High Risk)
], requires_grad=False)

# Targets: -1 = Healthy, 1 = High Risk
Y_train = np.array([-1, 1, -1, 1], requires_grad=False)

# --- 2. QUANTUM DEVICE ---
# Initialize a simulator with 2 qubits
dev = qml.device("default.qubit", wires=2)

# --- 3. QUANTUM CIRCUIT ---
@qml.qnode(dev)
def circuit(params, x):
    # Embedding Layer (Data Encoding)
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # Variational Layer (Trainable Weights)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    
    # Entanglement Layer
    qml.CNOT(wires=[0, 1])

    # Measurement
    return qml.expval(qml.PauliZ(0))

# --- 4. COST FUNCTION ---
def cost(params, x, y):
    prediction = circuit(params, x)
    return (prediction - y) ** 2

# --- 5. TRAINING LOOP ---
# Initialize random weights
params = np.array([0.1, 0.1], requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

print("Training Quantum Circuit...")
print("-" * 30)

epochs = 20
for i in range(epochs + 1):
    # Update parameters for each data point
    for j in range(len(X_train)):
        params = opt.step(lambda p: cost(p, X_train[j], Y_train[j]), params)

    if i % 5 == 0:
        loss_val = cost(params, X_train[0], Y_train[0])
        print(f"Step {i:2d} | Loss: {loss_val:.4f}")

# --- 6. VALIDATION ---
print("-" * 30)
print("FINAL PREDICTIONS")
print("-" * 30)

print(f"Healthy Patient Input {X_train[0]} -> Prediction: {circuit(params, X_train[0]):.4f} (Target: -1)")
print(f"High Risk Patient Input {X_train[1]} -> Prediction: {circuit(params, X_train[1]):.4f} (Target:  1)")
