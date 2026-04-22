# 🧠 End-to-End Chest Cancer Classification using MLflow, DVC & CI/CD

This project demonstrates a **complete MLOps pipeline** for Chest Cancer Classification using Deep Learning, integrated with:

* ✅ **TensorFlow (VGG16)**
* ✅ **MLflow (Experiment Tracking)**
* ✅ **DVC (Pipeline Orchestration)**
* ✅ **Flask (Deployment API)**
* ✅ **Docker (Containerization)**
* ✅ **AWS (ECR + EC2 Deployment)**
* ✅ **GitHub Actions (CI/CD Automation)**

---

# 🚀 Project Overview

This project builds an end-to-end machine learning pipeline to:

1. Ingest medical image data
2. Train a deep learning model (VGG16)
3. Evaluate model performance
4. Track experiments using MLflow
5. Manage pipelines using DVC
6. Deploy model using Flask & Docker
7. Automate deployment using CI/CD

---

# 📂 Project Structure

```
Chest-Cancer-Classification/
│
├── src/cnnClassifier/
│   ├── components/
│   ├── pipeline/
│   ├── utils/
│   ├── entity/
│   └── config/
│
├── artifacts/                # Generated outputs (ignored in Git)
├── model/                    # Trained model for deployment
├── templates/                # HTML UI
├── app.py                    # Flask app
├── main.py                   # Pipeline runner
├── dvc.yaml                  # DVC pipeline
├── requirements.txt
└── Dockerfile
```

---

# ⚙️ Workflow Steps

1. Update `config.yaml`
2. Update `params.yaml`
3. Define entities (`config_entity.py`)
4. Build configuration manager
5. Implement components
6. Create pipeline stages
7. Execute via `main.py`
8. Track using MLflow
9. Orchestrate using DVC

---

# 🔬 MLflow (Experiment Tracking)

### ▶ Run MLflow UI

```bash
mlflow ui
```

👉 Open:

```
http://127.0.0.1:5000
```

### 🔹 Features

* Track experiments
* Log metrics (accuracy, loss)
* Save models
* Compare runs

---

# 🔁 DVC (Pipeline Management)

### ▶ Initialize DVC

```bash
dvc init
```

### ▶ Run pipeline

```bash
dvc repro
```

### ▶ Visualize pipeline

```bash
dvc dag
```

### ▶ Show metrics

```bash
dvc metrics show
```

---

# 🌐 Flask Web App

### ▶ Run application

```bash
python app.py
```

👉 Open:

```
http://127.0.0.1:8080
```

### Features:

* Upload CT scan image
* Predict Cancer / Normal
* Display result instantly

---

# 🐳 Docker Setup

### ▶ Build Image

```bash
docker build -t cancer-app .
```

### ▶ Run Container

```bash
docker run -p 8080:8080 cancer-app
```

---

# ☁️ AWS Deployment (ECR + EC2)

## Steps:

### 1. Create IAM User

Permissions:

* AmazonEC2FullAccess
* AmazonEC2ContainerRegistryFullAccess

---

### 2. Create ECR Repository

Example:

```
566373416292.dkr.ecr.ap-south-1.amazonaws.com/cancer-app
```

---

### 3. Launch EC2 Instance (Ubuntu)

---

### 4. Install Docker

```bash
sudo apt-get update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### 5. Setup GitHub Self-Hosted Runner

* Go to: GitHub → Settings → Actions → Runners
* Add self-hosted runner
* Execute commands on EC2

---

# 🔄 CI/CD using GitHub Actions

### Pipeline includes:

1. **CI Stage**

   * Code checkout
   * Linting
   * Testing

2. **Build Stage**

   * Build Docker image
   * Push to AWS ECR

3. **Deployment Stage**

   * Pull image on EC2
   * Run container
   * Serve application

---

# 🔐 GitHub Secrets

Add these:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_ECR_LOGIN_URI
ECR_REPOSITORY_NAME
```

---

# 🧠 MLflow vs DVC

| Feature             | MLflow | DVC        |
| ------------------- | ------ | ---------- |
| Experiment Tracking | ✅      | ⚠️ Limited |
| Pipeline Management | ❌      | ✅          |
| Model Registry      | ✅      | ❌          |
| Version Control     | ❌      | ✅          |

---

# 📊 Model Details

* Model: **VGG16 (Transfer Learning)**
* Input Size: 224x224
* Output: Binary Classification
* Classes:

  * Normal
  * Cancer

---

# 🎯 Results

* Accuracy: ~75% (example)
* Loss: ~0.65

(Stored in `scores.json` and MLflow)

---

# 💡 Key Highlights (For Interview)

✅ End-to-End MLOps pipeline
✅ Modular coding architecture
✅ MLflow experiment tracking
✅ DVC pipeline orchestration
✅ Docker containerization
✅ AWS deployment (ECR + EC2)
✅ CI/CD automation using GitHub Actions
✅ Real-time prediction using Flask

---

# 🚀 Future Improvements

* Add Grad-CAM visualization
* Improve model accuracy
* Add multi-class classification
* Deploy using Kubernetes
* Add monitoring (Prometheus/Grafana)

---

# 👨‍💻 Author

**Siva Paparao Medisetti**

---

# ⭐ If you like this project

Give it a ⭐ on GitHub!
