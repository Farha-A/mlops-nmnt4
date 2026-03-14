# ML Pipeline Project

This is a machine learning pipeline project. It uses GitHub Actions to automate Continuous Integration (CI).

## Project Structure

- `.github/workflows/ml-pipeline.yml`: The GitHub Actions workflow file for CI. Runs validation and model tests.
- `train.py`: The Python script for training the machine learning models.
- `requirements.txt`: Project dependencies including PyTorch, torchvision, matplotlib, and tqdm.

## CI/CD Pipeline

The GitHub actions workflow triggers on:
- `push`: To all branches *except* `main`
- `pull_request`: To any branch

The basic pipeline steps include:
1. Checking out the code repository
2. Setting up Python 3.10
3. Installing dependencies from `requirements.txt`
4. Running a Linter check
5. Bootstrapping a dry test of the Model Environment
6. Uploading the `README.md` file as a GitHub artifact named `project-doc`

## Dependencies

Make sure to install dependencies via pip:
```bash
pip install -r requirements.txt
```
