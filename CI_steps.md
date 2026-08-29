# CI Steps

## 0. Setup DVC Remote & Requirements
- **Add AWS S3 remote**:
  - Run `python -m awscli configure`
  - Setup DVC remote to S3
  - Push data/artifacts: `dvc push` + `git push`
- **Generate the `requirements.txt`**

---

## 1. Run DVC Pipeline in CI
- Install everything on GitHub runner
- **Normal way**:
  - 2 problems: `requirements.txt` and Dagshub auth
  - Keep solving requirements until you are good to go
  - Realize and add cache step in CI workflow
  - Generate auth token inside Dagshub
  - Add environment variable (`DAGSHUB_USER_TOKEN` / credentials) inside GitHub Secrets
  - Change Dagshub auth code in `model_evaluation` and `register_model`

---

## 2. Model Testing

### 2a. Model Test 1 — Loading the Model
- Create `tests` directory
- Create new test file `test_model.py`
- Add step to CI workflow

### 2b. Model Test 2 — Check the Model Signature
- Modify code to log model signature in `model_evaluation`
- Add `test_model_signature` method to `test_model.py`
- Add step to CI workflow

### 2c. Model Test 3 — Performance Test
- Add `test_model_performance` method in `test_model.py`

---

## 3. Promote Model if Passes All Tests
- Create `scripts` directory
- Create `promote_model.py` code
- Add stage in CI workflow

---

## 4. Flask Application Testing
- Change directory name `flask-app` to `flask_app`
- Update `app.py`:
  - Dagshub auth code
  - Code inside `predict`
  - `if __name__ == "__main__":` block
- Create `test_flask_app.py` file
- Add step in CI workflow
