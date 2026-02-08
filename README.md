# Q-Heart: Quantum Heart Disease Classifier 🫀

### 🚀 Hackathon Track
**Quantum Intelligence (Quantum Machine Learning)**

### 📖 Project Overview
Q-Heart is a Hybrid Quantum-Classical Machine Learning model designed to predict heart disease risk. By leveraging the principles of quantum mechanics (superposition and entanglement), this model explores how quantum circuits can detect patterns in medical data that classical algorithms might miss.

### 🛠️ Tech Stack
* **Language:** Python 3
* **Library:** PennyLane (Quantum Machine Learning)
* **Simulator:** Default Qubit

### 🧠 How It Works
1.  **Data Encoding:** Patient data (Age, Health Metrics) is converted into quantum states using rotation gates (`RX`, `RY`).
2.  **Quantum Circuit:** A Variational Quantum Circuit processes the data. We use `CNOT` gates to entangle the qubits, allowing the model to learn complex correlations between health metrics.
3.  **Optimization:** The model trains using Gradient Descent to minimize prediction error.

### 📊 Results
The model successfully converges, learning to distinguish between "Healthy" and "At Risk" patient profiles with high accuracy on the test dataset.
