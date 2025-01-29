# Student's Math score prediction

Objective of the project is to show MLOPs techniques and tools required to execute End to End Machine Learning Project using modular coding design.
Project includes from defining of problem statement to deployement of app on cloud services having separate modules for Data ingestion, Data transformation, Model building, Model evaluation, Training pipeline, and Prediction pipeline.   

Link to web app: [Student's score predictor](https://mathscorepredictor-umesh.streamlit.app/)

### Problem statement :
***
Prediction of Math score of students based on the input features such as Gender, Race, Parents education, food eating type, test preparations, writing score, and reading scores.  

### Tools used:
***
1. Git : Version controlling
2. Github : Code repository
3. Docker : Code containerization
4. Streamlit : User interface and deployement (Free source)
5. AWS EC2 and Elastic Beanstalk : Deployement on cloud services.

### Elastic Beanstalk deployement
***
- Create .ebextensions folder and add python.config file.
- Rename main.py to application.py or create application.py file
- Follow EBS deployement strategies as per AWS.

### Deployement of Streamlit app on EC2 instance
***
1. Login with your AWS console and launch an EC2 instance

2. Run the following commands

* sudo apt update

* sudo apt-get update

* sudo apt upgrade -y

* sudo apt install git curl unzip tar make sudo vim wget -y

* sudo apt install git curl unzip tar make sudo vim wget -y

* git clone "Your-repository"

* sudo apt install python3-pip

* pip3 install -r requirements.txt

* python3 -m streamlit run app.py (Temporary running)

* nohup python3 -m streamlit run app.py (Permanent running)

### Deployemnt of streamlit app on streamlit cloud sevices
***
Login with streamlit cloud account, then create app and link github repository with necessary details and deploy.

Note: requirements.txt file must be updated with the required libraries to run the app


***Thank you***