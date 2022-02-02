![GithubProfile](https://user-images.githubusercontent.com/23042512/117491618-47e87700-af25-11eb-9164-b06f11bac5f2.png)

## SUMMARY
I have deep knowledge of various machine learning and deep learning algorithms as well as reinforcement learning as applied to analog circuit optimization. Some of which are highlighted in this repository.

**Relevant Course Work:** Deep Reinforcement Learning, Deep Learning for NLP, Deep Learning for Computer Vision, Advanced Robotics, Intro to AI, Intro to Machine Learning, Algorithms & Data Structures, Software Fundamentals for Engineering Systems  

## EXPERTISE
### Deep Learning
- Transformer based networks such as BERT and GPT as well as traditional LSTM based networks
- Improve inference performance using graph optimization and weight quantization
- CNN architectures such as AlexNet, VGGNet, GoogLeNet for image classification as well as object detection using YOLO and SSD
- Semantic segmentation using Fully Convolutional Networks  (https://amitp-ai.medium.com/fcn-571881788e70)
- Synthesize new images using Generative Adversarial Networks (GAN) and Variational Auto Encoders (VAE)

### Deep Reinforcement Learning
- Dynamic Programming, Bayesian Optimization, Thompson Sampling, Monte-Carlo (MC) learning
- Temporal Difference (TD) learning: SARSA, Q-Learning, Expected SARSA, Deep Q Network (DQN), Double DQN
- Policy Gradient Methods: Advantage Actor Critic (A2C), Deep Deterministic Policy Gradient  (https://medium.com/@amitp-ai/policy-gradients-1edbbbc8de6b)

### Natural Language Processing
- Text pre-processing methods such as Tokenization, Stemming, Lemmatization, etc
- Traditional feature extraction methods such as bag-of-words, TFIDF, word embeddings like word2vec, Glove, etc
- Deep learning based models for NER, POS, Sentiment Analysis, Dependency Parsing
- Advanced deep learning models for Text Summarization  (https://github.com/amitp-ai/Text_Summarization_UCSD)
- Machine Translation and Question-Answering System  (https://github.com/amitp-ai/CS224n_Stanford_NLP)

<!-- ### Traditional Machine Learning
- Thorough knowledge of Linear & Logistic Regression, Support Vector Machines (SVMs), Naïve Bayes Classifier, Random Forests,  Boosting, etc
- Strong understanding of unsupervised learning algorithms such as PCA, K-means, expectation-maximization
- Collaborative filtering and content based recommendation systems as well as various anomaly detection algorithms
- Github Link:  https://github.com/amitp-ai/UCSDX_Mini_Projects

### Autonomous Vehicle Path Planning and Control
- Behavior planning (in structured environments) using cost function based finite state machines as well as in (unstructured environments) using A* search algorithm.
- Machine learning based environmental prediction and trajectory generation using jerk minimization techniques.
- Control: proportional-integrate-derivative (PID), Linear Quadratic Regulator (LQR), and Model Predictive Control (MPC)
-->

## RELEVANT PROJECTS
### Banana Collection Agent (Fall 2018)
- Trained a robot to pick the maximum number of good bananas while avoiding bad bananas.
- Received a reward of +1 for picking a good banana and -1 for picking a bad banana.
- State augmentation by including previous observations to transform the problem from POMDP to MDP.
- Trained the agent (end-to-end) from raw pixels to q-values using CNN based double DQN learning algorithm.
- For faster training, batch normalization technique was used.
- Trained using PyTorch on Google Cloud, achieving a 100-episode average reward of 12.

### Text Summarization
- Input text was first pre-processed followed by data wrangling and data exploration.
- Thereafter experimented with various encoder-decoder type of models using LSTM, attention based LSTM, transformers, and memory efficient transformers. Memory efficient transformers performed the best with Rouge-1 and Rouge-2 scores of 38.3 and 13.3.
- Productionized using a Docker container deployed on an AWS EC2 instance and served using a Flask based API.

### Question-Answering System on the SQuAD2.0 Dataset
- As part of Stanford’s CS224N’s final project, I experimented with a few different architectures for this task.
- Using Bi-Directional Attention Flow (BiDAF) network, achieved an F1 score of 62 on the validation set.
- Then added character level embeddings (in addition to word embeddings) to BiDAF and achieved F1 of 65.
- Thereafter built the transformer based QANet to further improve the F1 score to 70.
- Lastly used a pretrained BERT network to further improve the F1 score.

<!-- ### Image Segmentation (part of Udacity-Lyft Perception Challenge) (Fall 2017)
- Developed a deep learning based image segmentation system to detect vehicles and road surfaces.
- FCN was used as the segmentation network, and its encoder network was built using VGG16.
- Replaced fully connected output layers of VGG16 with fully convolutional layers.
- Decoded VGG output back to the input dimensions using learnable transposed convolutional layers.
- Used skip connections to improve detection resolution.
- Used Bayesian optimization to search for the optimal regularization hyperparameter.
- Trained using TensorFlow on Google Cloud. Final test set FScore was 0.86.

### Road Traffic Sign Classification (German Traffic Signs Dataset) (Fall 2016)
- Deep learning based image classification system to detect 43 different types of roads signs.
- Network built using two convolutional layers (each followed by maxpool and relu non-linearity), followed by three fully-connected layers.
- Weight-decay and dropout were used for regularization.
- Final layer outputs Softmax probabilities, and Adam optimizer was used to train the network.
- Split data in to training, validation, and test sets. Preprocessed input images such that they spanned from 0 to 1; and for improved training, Glorot’s method was used to initialize the network weights.

### Implemented Backpropagation Algorithm for Various Network Types (Spring 2017)
- Using Numpy, implemented forward and backward passes for fully connected neural network, convolutional neural network (CNN), recurrent neural network (RNN), and Long Short-Term Memory (LSTM).
- Generated adversarial examples by computing the gradient of the loss function with respect to the input image pixels.
- Computed saliency maps and class visualizations to understand how different layers and neurons in the network learn.

### Path Planner for Highway Driving (part of Udacity-Bosch path planning challenge) (Fall 2017)
- Finite state machine based behavior planner and smooth trajectory generation using spline functions.
- State transition was determined using a cost function that included distance to other vehicles, ride comfort (i.e. minimize jerk), and speed. Controlled steering angle and vehicle acceleration to minimize this cost function.
- The planner was implemented in C++ and was one of the top 25 winners in the challenge.
 -->
