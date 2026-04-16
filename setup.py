from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


# Function to read requirements
def get_requirements(file_path: str):
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()
        requirements = [
            req.strip()
            for req in requirements
            if req.strip() and req.strip() != "-e ."
        ]
    return requirements


# Package metadata
__version__ = "0.0.0"
REPO_NAME = "Chest-Cancer-Classification"
AUTHOR_USER_NAME = "MSIVAPAPARAO13"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "msivapaparao@gmail.com"


# Setup configuration
setup(
    name=SRC_REPO,
    version=__version__,
    author="SIVA PAPARAO MEDISETTI",
    author_email=AUTHOR_EMAIL,
    description="End-to-End Chest Cancer Classification using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",  # 🔥 Important (MLflow + TensorFlow compatibility)
    install_requires=get_requirements("requirements.txt"),
)