# CarQuest(Flask)

This is the Flask backend for handling machine learning inference and external API requests in the Car Search Recommendation System. It processes data sent from the Spring Boot backend and performs recommendations and chatbot interactions.

## Requirements

- **Python 3.8 or higher**: Ensure Python is installed on your system.
- **Virtual Environment** (optional but recommended): Set up a virtual environment to manage dependencies.

The following explains how to run this project in your local environment. If needed, you can try to change the port in the code and deploy it to the cloud by yourself.

## Installation

### Step 1: Clone the Repository

Clone the repository and navigate to the Flask backend directory:

```bash
git clone https://github.com/group9-pqwy/ISY5001-group9-model.git
cd yourrepository/your flask-app
```

### Step 2: Set Up Virtual Environment (Optional)

It’s recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

Install all required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional)

If your Flask app needs any environment variables (such as API keys for Google Gemini or other configuration options), create a `.env` file in the `flask-app` directory with the necessary keys:

```plaintext
GEMINI_API_KEY=your_api_key
```

Make sure to load these variables in your Flask application using the `python-dotenv` package or manually in the code.

### Step 5: Start the Flask Application

Run the Flask application with the following command:

```bash
python app.py
```

By default, this will start the application at `http://127.0.0.1:5000`.

### Step 6: Test Endpoints

Once the application is running, you can test the endpoints with tools like Postman or cURL.

### Directory Structure

```
your reposotory/
├── .venv/                             # Virtual environment directory
├── app.py                             # Main Flask application file
├── clean_data.csv                     # Preprocessed data file
├── model.ipynb                        # Jupyter notebook for analysis or model training
├── kmeans_model.pkl                   # Pickle file for KMeans model
├── matrix_factorization_model.h5      # HDF5 file for matrix factorization model
├── matrix_factorization_model.pkl     # Pickle file for matrix factorization model
├── ratings.csv                        # Ratings dataset
├── requirements.txt                   # File listing required Python dependencies
├── scaler.pkl                         # Pickle file for data scaler
└── word2vec_model.pkl                 # Pickle file for Word2Vec model

```

## API Endpoints

### `/recommend`
- **Method**: POST
- **Description**: Receives search criteria from the Spring Boot backend and returns recommended car results.
- **Request**: JSON object containing user search parameters.
- **Response**: JSON array of car recommendations.

### `/geminichat`
- **Method**: POST
- **Description**: Receives chat messages and forwards them to the Google Gemini API. Returns chatbot responses.
- **Request**: JSON object containing chat input.
- **Response**: JSON object with the chatbot's response.

## External Dependencies

- **Google Gemini API**: Used for generating responses in the chatbot functionality. Ensure you have your API key set up in `.env`.

## Contributing

We welcome contributions! Please ensure code style consistency and functionality requirements are met before submitting a pull request.

