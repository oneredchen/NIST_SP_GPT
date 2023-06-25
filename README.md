# NIST SP GPT

This is a Streamlit application that utilizes the OpenAI GPT-3.5 model to generate responses based on user prompts. It leverages a NIST (National Institute of Standards and Technology) database to provide relevant information.

## Prerequisites

Before running the application, make sure you have the following:

- Python 3.x installed
- Streamlit package
- Dotenv package
- langchain package
- OpenAI API key

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Install the required packages:

   ```bash
   pip install streamlit dotenv langchain
   ```

3. Create a `.env` file in the root directory of the project and add your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY=<your_api_key>
   ```

## Usage

To start the application, run the following command in your terminal:

```bash
streamlit run app_chat.py
```

Once the application is running, you can access it in your browser at http://localhost:8501.

### Input Prompt

Enter your desired prompt in the input field provided and press Enter. The application will generate a response based on the prompt using the OpenAI GPT-3.5 model.

## Data and Database

The application uses a NIST database to provide relevant information. If the database doesn't exist, it will be generated from PDF documents located in the `raw_data` directory. The generated database will be stored in the `db` directory.

## Documentation

### File Structure

- `main.py`: The main script that runs the Streamlit application.
- `raw_data/`: Directory containing the raw PDF documents.
- `db/`: Directory containing the generated NIST database.

### Code Structure

- `setup_paths()`: Sets up the necessary paths for data and the database.
- `setup_vectorstore_agent(data_path, db_path)`: Sets up the agent that interacts with the NIST database and the OpenAI GPT-3.5 model.
- `main()`: The main function that runs the Streamlit application.
- `prompt`: Takes user input and generates a response based on the input.
- `Execution time`: Records the execution time of the application.
