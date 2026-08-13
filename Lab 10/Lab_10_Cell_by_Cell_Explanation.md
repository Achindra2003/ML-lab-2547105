# Lab 10: Learning the XOR Boolean Function Using an MLP — Complete Cell-by-Cell Explanation

**Student Name**: Achindra Sharma  
**Roll Number**: 2547105  
**Notebook**: [`Lab_10_MLP_XOR.ipynb`](Lab_10_MLP_XOR.ipynb)  
**Dataset**: Boolean XOR Function  

---

## **Document Overview & Student Perspective**

This document provides an exhaustive, publication-grade, cell-by-cell walkthrough of every code cell, mathematical derivation, empirical output, and visual plot in [`Lab_10_MLP_XOR.ipynb`](Lab_10_MLP_XOR.ipynb). I wrote each markdown cell following execution to analyze the empirical outputs directly and discuss the practical machine learning implications from a student's perspective.

For every lab task, optional exercise, and self-learning initiative, five key dimensions are explicitly detailed:
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

---

## **Task 1: Dataset Creation & Mathematical Proof of Linear Inseparability**

### **Concept**
Defining the XOR boolean inputs $X \in \mathbb{R}^{4 \times 2}$ and outputs $y \in \mathbb{R}^{4 \times 1}$ and proving mathematical non-linear separability.

### **Why**
The XOR problem is the classic benchmark proving the necessity of hidden layers. A single-layer perceptron (Linear Classifier) cannot solve it because the classes are not linearly separable.

### **Mathematical Proof**
A single-layer perceptron computes $y = \sigma(w_1 x_1 + w_2 x_2 + b)$. For 100% classification accuracy on XOR, there must exist weights $(w_1, w_2, b)$ satisfying:
1. $x_1=0, x_2=0 \implies b \le 0$
2. $x_1=0, x_2=1 \implies w_2 + b > 0 \implies w_2 > -b \ge 0$
3. $x_1=1, x_2=0 \implies w_1 + b > 0 \implies w_1 > -b \ge 0$
4. $x_1=1, x_2=1 \implies w_1 + w_2 + b \le 0$

Adding inequality (2) and (3) yields $w_1 + w_2 + 2b > 0$. Since $b \le 0$, it follows that $w_1 + w_2 + b > w_1 + w_2 + 2b > 0$, directly contradicting inequality (4). Thus, a single straight line cannot separate XOR.

---

## **Tasks 2–5: Baseline MLP Implementations Across Three Frameworks**

### **Concept**
Constructing a baseline feedforward network (8 hidden neurons, `Tanh` activation) using Keras `Sequential`, PyTorch `nn.Module`, and TensorFlow primitive `tf.Variable` math.

### **Justification of Hyperparameters**
- **Architecture**: 8 hidden neurons provide smooth parameterized decision surfaces, shielding the model from local saddle point traps.
- **Activation**: `Tanh` maps outputs to $[-1, 1]$, delivering centered non-zero gradients even for 0-inputs.
- **Optimizer & Loss**: `Adam` ($LR=0.05$) combined with Binary Cross-Entropy loss ($\mathcal{L} = -y \log \hat{y} - (1-y) \log(1-\hat{y})$) guarantees rapid convergence within 200 epochs.

### **Student Interpretation**
- PyTorch requires explicitly zeroing gradients (`optimizer.zero_grad()`), deriving gradients via autograd (`loss.backward()`), and physically pushing weights down the gradient slope (`optimizer.step()`).
- TensorFlow Low-Level `tape.gradient()` computes partial derivatives via the chain rule dynamically on the memory graph. All three paradigms achieved structural mathematical equivalence (100% accuracy, loss ~0.003).

---

## **Optional Exercises (Met & Exceeded)**

### **Optional 1: 2D Decision Boundary Contour Plots for Every Framework**
- **Concept**: Sampling a 10,000-point grid spanning $[-0.5, 1.5] \times [-0.5, 1.5]$ to extract continuous probability predictions.
- **Student Interpretation**: Contour plots visually prove that hidden layers warp the decision surface into curved manifolds enclosing $(0,1)$ and $(1,0)$ in high-probability red regions, while leaving $(0,0)$ and $(1,1)$ in blue low-probability regions.

### **Optional 2: Overlaid Training Loss Convergence Curves & Summary Table**
- **Concept**: Graphing Loss vs. Epochs on a log scale across Keras, PyTorch, and TF Low-Level.
- **Student Interpretation**: Proves identical log-linear loss decay slopes across all three frameworks.

### **Optional 3: Systematic Hyperparameter Studies**
- **Learning Rate**: $LR \le 0.001$ fails to converge in 200 epochs; $LR \in [0.05, 0.1]$ is optimal; $LR \ge 1.0$ causes violent overshooting.
- **Activation Functions**: `Linear` collapses to single-layer linear failure; `ReLU` suffers from the "Dying ReLU" phenomenon; `Tanh` is optimal.
- **Hidden Topologies**: 1 neuron fails; 2 neurons form a rigid V-shape; 8+ neurons produce smooth circular boundary islands.

---

## **Advanced Self-Learning Initiatives (SLIs)**

### **SLI 10.1: Pure NumPy Analytical Backpropagation from Scratch (Zero Libraries)**
- **Concept**: Deriving explicit matrix partial derivatives ($\frac{\partial L}{\partial W_2}, \frac{\partial L}{\partial b_2}, \frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial b_1}$) and writing a pure Python training loop using raw `numpy` matrix dot products.
- **Student Interpretation**: Confirms that autograd engines in PyTorch and TensorFlow execute exact matrix chain-rule updates under the hood.

### **SLI 10.2: Latent Hidden Space Feature Transformation Visualization**
- **Concept**: Extracting 2D hidden layer activation coordinates $(h_1, h_2) = \tanh(X W_1 + b_1)$ and plotting them on a 2D scatter plot.
- **Student Interpretation**: Visually proves how the non-linearly separable inputs $(0,0), (0,1), (1,0), (1,1)$ are geometrically warped in hidden space so that a simple straight line can linearly separate them!

---

## **Final Conclusions**
1. Multi-Layer Perceptrons resolve linear inseparability by warping feature space representations in hidden layers.
2. Keras, PyTorch, and TensorFlow Low-Level APIs exhibit mathematical equivalence under matching seeds.
3. Optimal hyperparameter configurations ($LR=0.05$, `Tanh` activation, $8+$ neurons) shield the network against gradient trapping.
