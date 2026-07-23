# Summer AI/ML Journey

This repository documents my AI/ML learning journey. The focus is on understanding concepts by implementing them from scratch using PyTorch, scikit-learn, and Hugging Face, while building projects and maintaining daily progress.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Transformers


## Progress

### Day 1 – ML Foundations & First Workflow
- Set up the AI/ML development environment (Python, Jupyter, Git, GitHub, Kaggle).
- Learned NumPy, Pandas, and Matplotlib fundamentals.
- Performed EDA and preprocessing on the Titanic dataset.
- Studied core machine learning concepts (train/test split, supervised learning, linear regression, loss functions).
- Built a Random Forest classifier and made my first Kaggle submission.
- Implemented a Student Score Predictor using Linear Regression.

### Day 2 – LLM Foundations & GenAI

- Learned how Large Language Models (LLMs) work, including tokenization, embeddings, and next-token prediction.
- Studied the Transformer architecture, self-attention, Query-Key-Value mechanism, multi-head attention, and the differences between GPT and BERT.
- Practiced NumPy fundamentals for AI, including vectors, matrices, dot products, matrix multiplication, broadcasting, and vectorization.
- Explored the Hugging Face ecosystem by running pretrained models, understanding tokenizers, pipelines, and GPT-2 text generation.
- Learned semantic embeddings, cosine similarity, vector databases, and the Retrieval-Augmented Generation (RAG) pipeline.
- Studied AI Agents, LangChain fundamentals, tools, memory, and the ReAct (Reason–Act–Observe) framework.
- Built a mini RAG-based Study Assistant that retrieves relevant notes using Sentence Transformers and cosine similarity.

### Day 3 – Deep Learning Foundations

- Learned the fundamentals of Artificial Neural Networks (ANNs), including perceptrons, neurons, weights, biases, hidden layers, and activation functions (ReLU, Sigmoid, Tanh).
- Studied backpropagation, gradient descent, learning rate, and the complete neural network training process.
- Learned PyTorch fundamentals, including tensors, autograd, `nn.Module`, optimizers, and model building.
- Built my first neural network using PyTorch.
- Implemented a Digit Classification project using the Scikit-learn Digits dataset.
- Learned the complete deep learning workflow: data loading, train/test split, training loop, loss computation, optimization, evaluation, and prediction.
- Gained hands-on experience with debugging models and understanding how neural networks learn from data.

### Day 4 – Deep Learning Training & Optimization

- Learned how neural networks are trained using forward propagation, loss computation, backpropagation, and gradient descent.
- Studied optimization algorithms, including SGD, Momentum, and Adam, and understood their impact on model convergence.
- Mastered the PyTorch training loop by implementing `forward()`, `loss.backward()`, `optimizer.step()`, and `optimizer.zero_grad()`.
- Learned the concepts of overfitting and regularization techniques such as Dropout, Weight Decay, and Early Stopping.
- Performed experiments on the Digits dataset by tuning learning rate, epochs, and hidden layer size, and analyzed their effects on training performance.
- Visualized loss and accuracy curves to evaluate model learning and understand training behavior.
* Strengthened debugging skills and gained a deeper understanding of the complete deep learning training pipeline. 

### Day 5 – Convolutional Neural Networks (CNNs)

* Learned the intuition behind CNNs, convolution, filters, feature maps, padding, stride, and max pooling.
* Built a CNN from scratch in PyTorch using `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, and `Linear` layers.
* Trained and evaluated a CNN on the MNIST dataset for handwritten digit classification.
* Understood the complete CNN workflow: Dataset → DataLoader → Model → Loss Function → Optimizer → Training Loop → Evaluation.
* Gained a clear understanding of `CrossEntropyLoss`, `argmax(dim=1)`, `model.train()`, `model.eval()`, and `torch.no_grad()`.
* Developed strong intuition for tensor shapes, forward/backward propagation, and CNN-based image classification.

### Day 6 – CIFAR-10 Image Classification

* Built an end-to-end CNN-based image classifier for the CIFAR-10 dataset using PyTorch.
* Learned to load and preprocess RGB image datasets using `Dataset`, `DataLoader`, and `torchvision.transforms`.
* Strengthened understanding of batching, training loops, evaluation loops, and model checkpointing.
* Learned the role of `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, `model.eval()`, and `torch.no_grad()` in the training pipeline.
* Developed a strong understanding of tensor shapes, `argmax(dim=1)`, and prediction generation for multi-class classification.
* Trained the model, evaluated its performance, analyzed predictions, and gained practical experience debugging and improving CNN models.


### Day 7 – Transfer Learning

• Learned the concept of Transfer Learning and why pretrained models outperform training CNNs from scratch on small datasets.
• Studied ResNet18 architecture, ImageNet pretraining, feature extraction, and fine-tuning.
• Learned to use pretrained models in PyTorch, freeze layers, and replace the final classification layer.
• Built a Cats vs Dogs image classifier using a pretrained ResNet18 model.
• Implemented image preprocessing, `ImageFolder`, `DataLoader`, training/validation split, and model evaluation.
• Practiced the complete transfer learning workflow, including training, validation, prediction, and model saving.
• Achieved **100% validation accuracy** on the Cats vs Dogs classification task.
• Strengthened understanding of transfer learning workflows commonly used in real-world computer vision applications.

### Day 8: Advanced PyTorch

• Built custom `Dataset` and `DataLoader` classes  
• Applied image augmentation and normalization  
• Created a modular PyTorch project structure  
• Implemented reusable training and validation pipelines  
• Saved and loaded model checkpoints  
• Used learning rate scheduling (`StepLR`)  
• Learned `model.train()`, `model.eval()`, and `torch.no_grad()`  
• Refactored the Transfer Learning project into clean, reusable code  


### Day 10: RNNs & LSTMs

• Learned why sequence models are needed for NLP  
• Understood the architecture of Recurrent Neural Networks (RNNs)  
• Studied hidden states and sequence processing  
• Explored the vanishing gradient problem in RNNs  
• Learned how LSTMs overcome long-term dependency issues  
• Understood the Forget, Input, and Output gates in LSTMs  
• Built and experimented with an LSTM model in PyTorch  
• Implemented a sentiment analysis model using LSTM  
• Learned the role of embeddings in sequence models  
• Gained hands-on experience with sequence classification


### Day 11: Sequence-to-Sequence (Seq2Seq)

• Learned the need for Seq2Seq models in sequence generation tasks  
• Understood the Encoder–Decoder architecture  
• Explored the concept of the Context Vector and its limitations  
• Learned Teacher Forcing and its role during training  
• Understood the difference between training and inference  
• Learned the purpose of special tokens: `<SOS>`, `<EOS>`, `<PAD>`, and `<UNK>`  
• Built a basic Seq2Seq model using PyTorch  
• Implemented a toy English-to-French translation model  
• Understood the context vector bottleneck and the motivation for Attention 


## Day 12: Attention Mechanism & NLP Project

### Topics
• Context Vector Bottleneck
• Luong Dot-Product Attention
• Attention Scores & Weights
• Attention Heatmaps

### Implementations
• Built Seq2Seq with Luong Dot-Product Attention
• Trained English → French translation model
• Visualized attention weights
• Compared Seq2Seq with and without Attention

### Mini Project
• AG News Text Classification
• TF-IDF Vectorization
• Logistic Regression
• Confusion Matrix & Classification Report

### Skills Gained
• Implemented attention from scratch
• Understood attention visualization
• Built a complete NLP classification pipeline
