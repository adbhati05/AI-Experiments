# Importing dotenv and os so I can access the variables in .env.
import dotenv
import os

from google import genai

# Loading the .env file and setting up the client with the API key.
dotenv.load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Creating a simple prompt for Gemini to respond to via user input.
while True:
    userInput = input("Ask Gemini a question (or 'quit' to exit): ")
    if userInput.lower() == "quit":
        break
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=userInput
    )

    print(response.text)