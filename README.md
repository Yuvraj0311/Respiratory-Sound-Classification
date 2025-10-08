
# A.I.R.A. - AI Respiratory Assistant
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18.0+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

A.I.R.A. is an end-to-end application that serves as a personal AI health companion for respiratory analysis. It combines a user-friendly interface with a powerful deep learning backend to classify respiratory diseases from audio recordings and provide AI-driven health insights.

## ✨ Key Features

- **Interactive AI Chat:** Engage in a conversation with an AI health assistant for information and guidance.
- **Respiratory Audio Analysis:** Upload an audio file (`.wav`, `.mp3`, `.flac`) and receive a potential respiratory disease classification powered by a 2D CNN model.
- **Personalized Health Profiles:** Register, log in, and manage a detailed health profile, including comorbidities, medications, and allergies.
- **Dynamic Profile Editing:** Easily update your health information and profile picture.
- **Secure Authentication:** User accounts are protected with a username-based login system.
- **Full-Stack Architecture:** Built with a modern tech stack including a React frontend and a FastAPI backend.

## 🏛️ Architecture

A.I.R.A. is composed of three main components that work together to deliver a seamless experience:

- **Frontend (React):** A responsive user interface built with Vite that allows patients to register, log in, manage their profiles, and interact with the AI chatbot and audio analysis tools.
- **Backend (FastAPI):** A robust Python server that handles user authentication, manages patient data in a PostgreSQL database, processes audio files, runs the classification model, and communicates with the OpenAI API for chat responses.
- **Deep Learning Model (TensorFlow/Keras):** A 2D Convolutional Neural Network trained on Mel-spectrograms of lung sounds to classify respiratory conditions. The model's training incorporates a **selective augmentation** strategy to effectively handle class imbalance in the medical audio dataset.

## 💻 Technology Stack

- **Frontend:** React, Vite, React Router, Axios, React Hot Toast
- **Backend:** Python, FastAPI, Uvicorn, PostgreSQL
- **AI & Machine Learning:** TensorFlow, Keras, Librosa, OpenAI API, KaggleHub
- **Styling:** CSS Modules with a custom warm-brown theme.

## 🚀 Getting Started

Follow these instructions to get a local copy of the project up and running for development and testing purposes.

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or newer)
- [Python](https://www.python.org/) 3.9+ and Pip
- [PostgreSQL](https://www.postgresql.org/) running instance

### 1. Clone the Repository

```bash
git clone https://github.com/anjalichaturvedi/aira.git
cd aira
```

### 2. Backend Setup

First, set up and run the FastAPI server.

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt

# Create an environment file from the example
cp .env.example .env

# Edit the .env file with your credentials
# (PostgreSQL details, OpenAI API key, etc.)
nano .env

# Ensure your PostgreSQL server is running and the database exists.
# You will need to manually run the SQL scripts to create the required tables.

# Run the backend server
uvicorn main:app --reload
```
The backend API will be available at `http://localhost:8000`. You can view the interactive documentation at `http://localhost:8000/docs`.

### 3. Frontend Setup

In a new terminal, set up and run the React frontend.

```bash
# Navigate to the frontend directory from the root
cd frontend

# Install Node.js dependencies
npm install

# Create the environment file
touch .env

# Add the backend API URL to the .env file
echo "VITE_API_BASE_URL=http://localhost:8000" > .env

# Run the frontend development server
npm run dev
```
The application will be available in your browser at `http://localhost:5173`.


## ⚙️ Environment Variables

To run this project, you will need to add the following environment variables to your .env file

```
OPENAI_API_KEY = ''

POSTGRES_HOST= ''
POSTGRES_PORT= ''
POSTGRES_USER= ''
POSTGRES_PASSWORD= ''
POSTGRES_DB= ''

VECTOR_DB_PATH= ''

LOG_LEVEL= INFO
```


## 📁 Project Structure

```
aira/
├── backend/            # FastAPI server, API endpoints, and business logic
│   ├── .env.example    # Environment variable template
│   ├── main.py         # Main application entrypoint
│   └── requirements.txt
|   └── database.py     # Database manager
|   └── schema.sql      # Database schema
├── data/               # Information about the dataset used for the model
├── frontend/           # React frontend application (Vite-based)
│   ├── public/
│   ├── src/
│   │   ├── components/ # React components (Auth, Chatbot, Sidebar, etc.)
│   │   ├── css/        # Component-specific CSS files
│   │   └── App.jsx     # Main app component with routing
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🧠 Model Details

The core of the audio analysis is a deep learning model trained for automated respiratory disease classification.

- **Dataset:** The model uses the [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database/data) from Kaggle, downloaded automatically via `kagglehub`.
- **Methodology:** The pipeline involves converting raw audio into Mel-spectrograms, which serve as input to a 2D CNN. This approach preserves the crucial time-frequency relationships in the lung sounds.
- **Selective Augmentation:** To combat the severe class imbalance (8.5:1 ratio) in the original dataset, a selective augmentation strategy was employed. Techniques like noise addition, time stretching, and pitch shifting were applied *only to the minority classes*, improving the class balance to 2.1:1 and resulting in a more robust and balanced classifier.


## 📖 Dataset Citation

```bibtex
@dataset{respiratory_sound_db_2019,
  title     = {Respiratory Sound Database},
  author    = {vbookshelf},
  year      = {2019},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database}
}
```
