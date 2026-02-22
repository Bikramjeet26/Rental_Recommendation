Rental Recommendation System: Sequential LSTM
This project was developed as part of a data science competition/research hosted on Kaggle.
    Dataset: The raw interaction data and product metadata were sourced from the [Rental_Recommendation_Competition](https://www.kaggle.com/competitions/rental-product-recommendation-system/data) repository.

    Competition Context: The objective was to optimize rental recommendations by processing sequential user behavior and high-dimensional product embeddings.
    
## Project Description
The core of this project is capturing user intent through "journeys." By combining user behavior data with semantic product embeddings, the system learns the transition probabilities between different rental categories and products. This allows for personalized, real-time recommendations based on a user's current browsing session.


## Data Engineering Pipeline
A significant portion of this project involved unifying multiple data sources to create a clean training set:

    Metrika Integration: Merged web analytics (Hits and Visits) into a single unified dataframe to track chronological user movements.

    Cross-Site Mapping: Reconciled product slugs from both the "New Site" and "Old Site" CSV catalogs to ensure consistent product identification.

    Slug-to-ID Mapping: Created a high-performance dictionary mapping unique slugs to global Product IDs across three different dataframes.

    Embedding Transformation: Processed 768-dimensional embeddings, converting string-based storage into optimized NumPy tensors for model consumption.

## Model Architecture
Type: Many-to-One 
LSTMLayers: 2-layer Stacked LSTM with a hidden dimension of 512.
Input: 768-dimensional vector sequences (Word/Product Embeddings).
Feature Engineering: Implemented a SlidingWindowDataset to create training     samples from variable-length journeys.
Optimization: Adam Optimizer ($lr=0.01$) with Cross-Entropy Loss.
