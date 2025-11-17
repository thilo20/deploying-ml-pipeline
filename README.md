# Overview

This repo holds my final project for Udacity course "Deploying a Scalable ML Pipeline in Production"
https://www.udacity.com/enrollment/cd0582

## Decisions

1. I decided to not use DVC. My reasoning:

- DVC with remote GoogleDrive currently has an issue (blocked app)
- data and model files are small, thus direct storage in git repo is feasible

2. I noticed a complex directory structure with 2 nested 'starter' directories.
   My first impulse was to flatten the structure - I resisted it because it might complicate reviewing the project for the mentors!

3. Unfortunately, the structure gave me a hard time testing and deploying:

- vscode Testing UI did not work (but `pytest` on the terminal works!)
- Heroku application setup was trial&error, as Procfile must be at root but main:app is inside subdir!
- notice also the duplicate requirements.txt (for build and deploy)

# Environment Set up

Find below the choices I took based on the starter instructions.

- **Option 2: Using conda**
  - Download and install conda if you don't have it already.
  - conda create -n [envname] "python=3.13" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn pydantic httpx matplotlib seaborn -c conda-forge

## Repositories

- Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
  - Make sure you set up the GitHub Action to use Python 3.13 (same version as development).
  - Note: Add flake8 to requirements.txt if you want to use it for linting: `pip install flake8`

fyi: I ran flake8 locally to discover that only the provided sanitycheck.py has issues ;)

# Data

- Download census.csv and commit it to dvc.
- To clean it, use your favorite text editor to remove all spaces.

# Model

- Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
- Write unit tests for at least 3 functions in the model code.
- Write a function that outputs the performance of the model on slices of the data.
  - Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
- Write a model card using the provided template.
  - see [model card](starter/model_card.md)

# API Creation

- Create a RESTful API using FastAPI this must implement:
  - GET on the root giving a welcome message.
  - POST that does model inference.
  - Type hinting must be used.
  - Use a Pydantic model to ingest the body from POST. This model should contain an example.
    - Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
- Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment

- Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
- Create a new app and have it deployed from your GitHub repository.
  - Enable automatic deployments that only deploy if your continuous integration passes.
- Write a script that uses the requests module to do one POST on your live API.

Final test of the live API:

    (udacity2) ➜  starter git:(main) ✗ python3 call_api.py
    Calling the live API on Heroku..
    statuscode: 200
    {'prediction': '>50K'}

Please see 'starter/screenshots' for visual documentation.
