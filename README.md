![GithubProfile](https://user-images.githubusercontent.com/23042512/117491618-47e87700-af25-11eb-9164-b06f11bac5f2.png)

## SUMMARY
My name is Amit Patel. I am a Machine Learning Engineer/Researcher and I have deep knowledge of various machine learning and deep learning algorithms as applied to Natural Language Processing (NLP). Some of which are highlighted in this repository.

## SKILLS
**Programming:** Python (advanced), C++, SQL, Linux/Bash, Matlab  
**Frameworks/Libraries:** Pytorch (advanced), Tensorflow, SpaCy, NLTK, Pandas, PySpark, Dask, Git, Docker, RegEx, Scikit-Learn  
**Cloud Computing Platforms:** Google Cloud Platform (GCP), Amazon Web Services (AWS)  
**Relevant Course Work:** Intro to AI, Intro to Machine Learning, Natural Language Processing, Computer Vision, Advanced Robotics,
Reinforcement Learning, Algorithms & Data Structures, Software Fundamentals for Engineering Systems  
**Languages:** English, Swahili, Hindi, Gujarati  

## EXPERTISE
### Natural Language Processing
- Text pre-processing methods such as Tokenization, Stemming, Lemmatization, etc.
- Traditional feature extraction methods such as bag-of-words, TFIDF, etc.
- Deep learning based models for NER, POS, Sentiment Analysis.
- Advanced deep learning models for Text Summarization, Question-Answering Systems, and Machine Translation.

### Deep Learning
- Transformer based networks such as BERT and GPT as well as traditional LSTM based networks.
- Improve inference performance using graph optimization and weight quantization.
- CNN architectures such as AlexNet, VGGNet, GoogLeNet for image classification.
- Object detection using YOLO and SSD as well as semantic segmentation using Fully Convolutional Networks (FCN).
- Synthesize new images using Generative Adversarial Networks (GAN) and Variational Auto Encoders (VAE).

### Deep Reinforcement Learning
- Dynamic Programming, Bayesian Optimization, Thompson Sampling, Monte-Carlo (MC) learning.
- Temporal Difference (TD) learning: SARSA, Q-Learning, Expected SARSA, Deep Q Network (DQN), Double DQN.
- Policy Gradient Methods: Advantage Actor Critic (A2C), Deep Deterministic Policy Gradient (DDPG).

### Traditional Machine Learning
- Thorough knowledge of Linear & Logistic Regression, Support Vector Machines (SVMs), Naïve Bayes Classifier, Random Forests,
Boosting, etc.
- Strong understanding of unsupervised learning algorithms such as PCA, K-means, expectation-maximization.
- Collaborative filtering and content based recommendation systems as well as various anomaly detection algorithms.

## RELEVANT PROJECTS
### Text Summarization (Spring 2021)
- Mastered skills in the full machine learning stack, including data wrangling, deep learning, and deploying ML systems at scale.
- 400+ hours of hands-on course material with 1:1 industry expert mentor and completion of an in-depth capstone project.
- Capstone project was on text summmarization where input text was first pre-processed followed by data wrangling and data exploration.
- Thereafter experimented with various encoder-decoder type of models using LSTM, attention based LSTM, transformers, and memory efficient transformers. Memory efficient transformers performed the best with Rouge-1 and Rouge-2 scores of 38.3 and 13.3.
- Productionized using a Docker container and served the model using a web interface through AWS.

### Question-Answering System on the SQuAD2.0 Dataset (Fall 2020)
- As part of Stanford’s CS224N’s final project, I experimented with a few different architectures for this task.
- Using Bi-Directional Attention Flow (BiDAF) network, achieved an F1 score of 62 on the validation set.
- Then added character level embeddings (in addition to word embeddings) to BiDAF and achieved F1 of 65.
- Thereafter built the transformer based QANet to further improve the F1 score to 70.
- Lastly used a pretrained BERT network to further improve the F1 score.

### Banana Collection Agent (Fall 2018)
- Trained a robot to pick the maximum number of good bananas while avoiding bad bananas.
- Received a reward of +1 for picking a good banana and -1 for picking a bad banana.
- State augmentation by including previous observations to transform the problem from POMDP to MDP.
- Trained the agent (end-to-end) from raw pixels to q-values using CNN based double DQN learning algorithm.
- For faster training, batch normalization technique was used.
- Trained using PyTorch on Google Cloud, achieving a 100-episode average reward of 12.

### Image Segmentation (part of Udacity-Lyft Perception Challenge) (Fall 2017)
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

For further details, please contact me at amitpatel.gt@gmail.com
