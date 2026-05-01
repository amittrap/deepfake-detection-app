import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Manually copied from training logs
epochs = list(range(1, 11))
losses = [
    0.6797, 0.6185, 0.5034, 0.4171, 0.3658,
    0.3275, 0.2877, 0.2577, 0.2338, 0.2049
]

plt.figure()
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()
