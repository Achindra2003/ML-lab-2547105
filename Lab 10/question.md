## **Lab Exercise 10: Learning the XOR Boolean Function Using an MLP**

### **Aim**

1. To understand how to implement neural networks using different deep learning libraries (**Keras, PyTorch, and TensorFlow**).  
2. To solve the non-linear XOR problem using an MLP and study the effect of hyperparameters such as learning rate, activation functions, number of neurons, and epochs on model performance.

---

### **Question**

**Implement an MLP to learn the XOR Boolean function**

The XOR function takes two binary inputs (0 or 1\) and produces a binary output (0 or 1\) based on the following rule:

| Input 1 | Input 2 | XOR Output |
| ----- | ----- | ----- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Steps:**

1. **Create the Dataset**  
   * Define input (X) and output (y) arrays for all 4 XOR combinations.  
2. **Build an MLP**  
   * Input layer: size 2 (for the two inputs).  
   * Hidden layer: at least 2 neurons with **ReLU** or **Tanh** activation.  
   * Output layer: 1 neuron with **sigmoid** activation for binary classification.  
3. **Compile the Model / Define Loss and Optimizer**  
   * Use **Binary Cross-Entropy** loss.  
   * Use an optimizer of your choice (e.g., **Adam** or **SGD**).  
4. **Train the Model**  
   * Train the model on the XOR dataset.  
   * Experiment with **epochs, learning rate, and number of neurons** to improve performance.  
5. **Evaluate the Model**  
   * Predict outputs for all 4 input combinations.  
   * Check if the network correctly learns the XOR function.  
6. **Implement Using Three Libraries**  
   * Repeat the above steps using:  
     1. **Keras (TensorFlow high-level API)**  
     2. **PyTorch**  
     3. **TensorFlow low-level API**

---

**Additional Exercises (Optional):**

* Plot the decision boundary for each implementation.  
* Compare training curves and final accuracy between libraries.  
* Discuss how changes in **learning rate, activation function, hidden layers, or epochs** affect learning.

---

### **Evaluation Rubrics**

1. 

| Rubrics | Marks |
| :---- | :---- |
| Correctness and Demonstration | 5 marks |
| Concept Clarity (Viva) | 3 marks |
| Initiative & Effort (self-learning) | 2 marks |

---

### **Submission Guidelines**

1. Make a copy of the lab manual template with your \<name\_reg:no\_subject name\>  
2. Copy the given question and the answer (lab code) with results, followed by the conclusion of that lab. Title the lab as Lab 1\.  
3. Keep updating your lab manual and show the lab manual of that particular lab for evaluation.  
4. Create a **Git repository** in your profile \<DL lab-reg no\>. Follow a different branch for each lab \<Lab 1, Lab 2 …\> and push the code to Git.  
5. Provide the Git link in **Google Classroom** along with the PDF of the lab manual.  
6. Upload the PDF to Google Classroom before the deadline.

