from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# A Pydantic model defines the structure of the data to be sent in the POST request body.
class UserData(BaseModel):
    name: str
    email: str

app = FastAPI()

# Mount a directory to serve static files (like your HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/submit_data")
async def submit_data(user_data: UserData):
    """
    Receives JSON data from the frontend and returns a confirmation message.
    """
    print(f"Received data: Name: {user_data.name}, Email: {user_data.email}")
    return {"message": "Data received successfully!", "submitted_data": user_data}

@app.get("/")
async def serve_frontend(request: Request):
    """
    Redirects the root URL to the static HTML file.
    """
    return app.get("/static/index.html")

