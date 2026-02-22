Rental Recommendation System: Sequential LSTM
This project was developed as part of a data science competition/research hosted on Kaggle.
    Dataset: The raw interaction data and product metadata were sourced from the [Rental_Recommendation_Competition](https://www.kaggle.com/competitions/rental-product-recommendation-system/data) repository.

    Competition Context: The objective was to optimize rental recommendations by processing sequential user behavior and high-dimensional product embeddings.
    
## Project Description
The core of this project is capturing user intent through "journeys." By combining user behavior data with semantic product embeddings, the system learns the transition probabilities between different rental categories and products. This allows for personalized, real-time recommendations based on a user's current browsing session.


## Data Engineering Pipeline
The preprocessing pipeline was designed to transform unstructured web traffic data into a structured format suitable for sequential modeling.

### 1. Metrika Data Consolidation
    Temporal Alignment: Merged Metrika Hits and Visits logs to reconstruct a chronological "User Journey."

    Session Reconstruction: Grouped interactions by unique user identifiers to identify distinct browsing sessions.
### 2. Cross-Catalog Harmonization
    Entity Resolution: Mapped inconsistent product identifiers between "Old Site" and "New Site" schemas using a unified slug-to-ID dictionary.

    Cold-Start Handling: Implemented an UNKNOWN token mapping for products not found in the primary catalogs to maintain sequence integrity.
### 3. Feature Engineering
    Embedding Vectorization: Parsed 768-dimensional product embeddings from string-based CSV storage into high-performance float32 tensors.

    Sliding Window Generation: Developed a custom SlidingWindowDataset to slice long journeys into multiple training samples, allowing the model to learn from every step of the sequence.

## Model Architecture & Training
The recommendation engine uses a Many-to-One Recurrent Neural Network (RNN) to predict the next item in a sequence based on short-term historical context.

### 1. Model Topology
    Input Layer: Accepts a variable-length sequence of 768-dimensional embedding vectors.
    
    Hidden Layers: A 2-layer stacked LSTM (Long Short-Term Memory) with 512 hidden units to capture long-range dependencies and mitigate the vanishing gradient problem.
    
    Regularization: Integrated Dropout (0.2) between LSTM layers to prevent overfitting on specific user journey patterns.
    
    Output Layer: A Dense Linear layer mapping hidden states to the total number of unique product classes.

### 2. Training Strategy
    Sequence Packing: Utilized PyTorch’s pack_padded_sequence to handle variable-length inputs efficiently within batches, ignoring "padding" tokens during the backward pass.

    Optimization: Used the Adam Optimizer ($lr=0.01$) and CrossEntropyLoss for multi-class classification.

    Validation: Performed an 80/20 split at the journey level (rather than the sample level) to ensure the model was tested on entirely unseen user behaviors.

### 3. Evaluation Metrics
    Recall@6: The primary success metric. Because the goal is to provide a "Top 6" recommendation list, the model is considered successful if the ground-truth product appears within the top 6 predicted logits.
