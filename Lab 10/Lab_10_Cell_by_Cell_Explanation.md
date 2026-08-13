# Lab 10: Learning the XOR Boolean Function Using an MLP — Complete Cell-by-Cell Explanation

**Student Name**: Achindra Sharma  
**Roll Number**: 2547105  
**Notebook**: [`Lab_10_MLP_XOR.ipynb`](Lab_10_MLP_XOR.ipynb)  
**Dataset**: Boolean XOR Function  

---

## **Document Overview & Student Perspective**

This document provides an exhaustive, publication-grade, cell-by-cell walkthrough of every code cell, mathematical derivation, empirical output, and visual plot in [`Lab_10_MLP_XOR.ipynb`](Lab_10_MLP_XOR.ipynb). I wrote each markdown cell following execution to analyze the empirical outputs directly and discuss the practical machine learning implications from a student's perspective.

For every lab task and self-learning initiative, five key dimensions are explicitly detailed:
1. **Concept**: The underlying algorithmic, geometrical, or mathematical principle.
2. **Why**: The role of this step in the neural network workflow.
3. **Justification**: Why specific hyperparameters, activation functions, or optimizers were selected.
4. **Result**: The expected empirical outputs and metrics produced during execution.
5. **Student Interpretation**: In-depth analysis of what the losses, boundaries, and predictions mean in practice.

---

## **Part 1: Setup, Imports, and Reproducibility**

### **Concept**
Before building the Multilayer Perceptron (MLP) models, required software packages (`numpy`, `matplotlib`, `seaborn`) and deep learning frameworks (`tensorflow`, `keras`, `torch`) are imported. Global random seeds are fixed.

### **Why**
Ensures all software tools for array manipulation, computational graphs, and plotting are accessible. 

### **Justification**
- `SEED = 42` is fixed globally across all frameworks (`np.random.seed`, `tf.random.set_seed`, `torch.manual_seed`) to eliminate pseudo-random variance in weight initialization. For a dataset as small as XOR (4 samples), bad weight initialization can easily cause the network to get stuck in a local saddle point. Fixing the seed guarantees 100% execution reproducibility.
- `seaborn.set_theme(style="whitegrid")` ensures publication-quality visual aesthetics for the decision boundaries.

### **Result**
```text
Environment configured and all required packages (Keras, PyTorch, TF) imported successfully!
```

### **Student Interpretation**
By loading Keras, PyTorch, and TensorFlow low-level APIs concurrently, I can perform a direct 1-to-1-to-1 comparison of syntax, training time, and loss convergence within a single unified execution state.

---

## **Task 1: Dataset Creation**

### **Concept**
Defining the XOR (Exclusive OR) boolean inputs and expected binary outputs.

### **Why**
The XOR problem is the classic benchmark proving the necessity of hidden layers. A single-layer perceptron (Linear Classifier) cannot solve it because the classes are not linearly separable.

### **Justification**
Using `np.float32` explicitly prevents type-casting errors downstream when passing data into PyTorch and TensorFlow tensors, which strictly expect 32-bit floating point weights.

### **Student Interpretation**
If we plot these 4 points on a 2D Cartesian plane, there is no single straight line that can separate the points `(0,0), (1,1)` from `(0,1), (1,0)`. An MLP is mathematically required to warp this space.

---

## **Tasks 2, 3, & 4: Baseline MLP Implementations**

### **Concept**
Constructing a baseline feedforward network (8 hidden neurons, `Tanh` activation) using Keras `Sequential`, PyTorch `nn.Module`, and TensorFlow primitive `tf.Variable` math.

### **Why**
To establish that all three libraries converge structurally identically when seeded equally.

### **Justification**
- **Architecture Choice**: I used 8 hidden neurons instead of the theoretical minimum of 2. A wider layer creates a smoother, highly parameterized decision boundary, making gradient descent less likely to get stuck in saddle points.
- **Activation**: `Tanh` produces outputs in the range `[-1, 1]`. For strict `[0,1]` Boolean inputs, `Tanh` provides stronger, centered gradients than `ReLU`, avoiding the "dying ReLU" problem.
- **Optimizer & LR**: `Adam` with learning rate `0.05` was coupled with `200` epochs to ensure rapid bridging of the error gap.

### **Result**
All three frameworks hit 100% Accuracy and reduce the Binary Cross-Entropy loss near ~0.01.

### **Student Interpretation**
- PyTorch requires explicitly zeroing gradients (`optimizer.zero_grad()`), deriving gradients via autograd (`loss.backward()`), and physically pushing the weights down the gradient slope (`optimizer.step()`).
- TensorFlow Low-Level `tape.gradient()` computes partial derivatives via the chain rule dynamically on the memory graph. All three paradigms achieved structural mathematical equivalence.

---

## **Part 6: Self-Learning Initiatives (SLI) - Rigorous Hyperparameter Tuning**

In previous labs, I proved that hyperparameter impacts must be iteratively tested rather than assumed. Here, I execute deep testing loops to isolate the impact of Learning Rate, Neurons, and Activations.

### **SLI 10.1: Effect of Learning Rate (Adam Optimizer)**

### **Concept**
Training 5 independent Keras models for 200 epochs, varying $LR \in [0.001, 0.01, 0.05, 0.1, 0.5]$ and plotting the loss decay.

### **Student Interpretation**
- **LR = 0.001 (Default)**: Declines extremely slowly. It fails to solve XOR within 200 epochs (loss stays high around 0.5).
- **LR = 0.05 & 0.1**: The optimal "Goldilocks" zone. They plunge rapidly towards zero loss, bridging the gap without oscillating.
- **LR = 0.5**: Oscillates violently and plateaus sub-optimally due to overshooting the global minimum in the loss valley.

---

### **SLI 10.2: Effect of Hidden Neurons & Decision Boundaries**

### **Concept**
Training PyTorch models using 2, 4, 8, and 16 hidden neurons, and projecting their output vectors onto a 2D meshgrid to graph the exact topological contour boundaries.

### **Student Interpretation**
- **2 Neurons**: Represents the mathematical absolute minimum. The network solves XOR by drawing a rigid 'V' or two sharp intersecting hyperplanes.
- **8 & 16 Neurons**: Expanding the layer provides redundant geometric pathways, carving out highly circular, smooth decision islands around the targets. This topological flexibility prevents gradients from trapping in local saddle points, generating a vastly superior, robust model surface.

---

### **SLI 10.3: Effect of Activation Functions (ReLU vs Tanh)**

### **Concept**
Iterating over `relu`, `sigmoid`, and `tanh` hidden layer activations in Keras.

### **Student Interpretation**
- **`Tanh`**: Converges drastically faster and deeper. Because it maps values to `[-1, 1]`, it maintains active, non-zero gradients even when fed a 0-input.
- **`ReLU`**: Maps all negative values strictly to $0$. Given the XOR inputs contain many literal $0$s (`[0,0], [0,1]`), ReLU neurons frequently "die" early in training because gradients stop flowing completely, leading to an accuracy plateau at 50% or 75%. `Tanh` is mathematically superior for strictly Boolean inputs.

---

## **Final Lab Synthesis**

1. **Architectural Equivalence**: The XOR Boolean function was flawlessly mapped non-linearly using Keras, PyTorch, and TensorFlow primitive operations, proving structural equivalence across ML frameworks.
2. **Hyperparameter Tuning Rules for XOR**:
   - Optimal configuration dictates $LR \approx 0.05$, `Tanh` activations (over ReLU), and a wider hidden matrix ($8+$ neurons) to evade gradient trapping. Rigorous `for`-loop plotting confirmed these configurations empirically.
