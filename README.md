# Image Classification and Recommendation System

## Overview
This project implements an image classification and recommendation system using Azure Computer Vision and cosine similarity. The system extracts features from images using Azure Cognitive Services and provides similar image recommendations based on feature embeddings.

## Features
- Extracts image features using Azure Computer Vision API.
- Computes similarity between images using cosine similarity.
- Provides recommendations for visually similar images.
- Supports bulk image processing for efficient feature extraction.

## Technologies Used
- Python
- Azure Computer Vision API
- NumPy
- Scikit-learn (for cosine similarity computation)

## Prerequisites
- Python 3.x installed
- An active Azure account with access to Computer Vision API
- Azure Cognitive Services API key and endpoint
- Required Python libraries installed:
  ```sh
  pip install requests numpy scikit-learn
  ```

## Setup
1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/image-classification-recommendation.git
   cd image-classification-recommendation
   ```

2. Set up environment variables for Azure API credentials:
   ```sh
   export AZURE_API_KEY="your_api_key"
   export AZURE_ENDPOINT="your_api_endpoint"
   ```
   (Alternatively, you can store these in a `.env` file and load them in your script.)

3. Run the script to extract image features and compute similarities:
   ```sh
   python main.py
   ```

## Usage
- Place images in the `images/` directory.
- Run the script to extract features and compute similarity scores.
- The system will output a ranked list of similar images.

## Example Output
```
Image: image1.jpg
Similar Images:
1. image2.jpg (Similarity: 0.89)
2. image3.jpg (Similarity: 0.85)
...
```

## Future Enhancements
- Implement a web-based UI for image upload and recommendations.
- Optimize feature extraction using deep learning models.
- Integrate with a database for storing image embeddings.

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

## Contact
For any questions or support, reach out to [Your Name] at [your-email@example.com].

