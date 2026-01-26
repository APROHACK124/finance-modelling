# Step 1: Start with an official Python base image.
# This is like choosing the right pot to start cooking in.
FROM python:3.9-slim

# Step 2: Set the working directory inside the container.
# This is where our app's files will live inside the container.
WORKDIR /app

# Step 3: Copy the requirements file into the container.
COPY requirements.txt .

# Step 4: Install the Python libraries the app needs.
# This is like adding the spices and ingredients from the recipe.
RUN pip install -r requirements.txt

# Step 5: Copy the rest of your application's code into the container.
COPY . .

# Step 6: Tell Docker what port your app runs on inside the container.
# Our Python app uses port 5000.
EXPOSE 5000

# Step 7: The command to run when the container starts.
# This tells Docker how to start your app.
CMD ["python", "app.py"]
