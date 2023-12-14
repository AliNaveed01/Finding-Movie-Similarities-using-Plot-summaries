# Movie Similarities using Plot Summaries

This repository contains a DataCamp project that explores the similarity of movies based on their plot summaries available on IMDb and Wikipedia. The analysis involves importing and observing a dataset of movie plots, combining Wikipedia and IMDb plot summaries, tokenization, stemming, and creating TF-IDF vectors. The movies are then clustered using the K-means algorithm, and a dendrogram is plotted to visualize the hierarchical clustering of movie similarities.

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AliNaveed01/Finding-Movie-Similarities-using-Plot-summaries.git
   cd movie-similarities
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   Open and run the Jupyter Notebook `movie_similarity_analysis.ipynb` to explore the project.

# Requirements

To run the movie similarity analysis project, make sure you have the following requirements installed on your system.

## System Requirements

- **Python:** Version 3.6 or higher. You can check your Python version by running the following command:
  ```bash
  python --version
  ```

  If you need to update Python, you can download the latest version from [python.org](https://www.python.org/downloads/).

## Python Packages

Install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

This will install the necessary libraries used in the project.

### List of Packages

- `numpy`: Library for numerical operations.
- `pandas`: Data manipulation and analysis library.
- `nltk`: Natural Language Toolkit for text processing.
- `scikit-learn`: Machine learning library for clustering and vectorization.
- `matplotlib`: Plotting library for visualizations.

## Running the Jupyter Notebook

Once the requirements are installed, you can run the Jupyter Notebook (`movie_similarity_analysis.ipynb`) to explore the project. Open a terminal, navigate to the project directory, and run:

```bash
jupyter notebook movie_similarity_analysis.ipynb
```

This will launch the Jupyter Notebook interface in your web browser.

Feel free to reach out if you encounter any issues or have questions about the installation process!

## Project Structure

- `datasets/`: Contains the movie dataset (`movies.csv`).
- `movie_similarity_analysis.ipynb`: Jupyter Notebook with the project code and analysis.
- `README.md`: Project overview and installation guide.

## Sample Code

```python
# Import necessary modules
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(5)

# ... (Code snippets from the project)

# Display dendrogram
plt.show()
```

## Contribution

Contributions to this project are welcome! If you have suggestions, improvements, or new ideas, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
