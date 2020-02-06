# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project Overview
In the Project Workspace/data, we have a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

THis project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Project Files
There are two important Files of this project.

1. ETL Pipeline

a Python pipeline script that cleaning the data as follow:
Loads two csv files messages and categories
Mergeing them
Cleaning the data
Stores it as a SQLite db

2. ML Pipeline

In a Python machine learning pipeline script that train our model then export it as a pickle file

### Project Screenshots (Flask Web App)
![Screen Shot 2020-02-07 at 00 59 27](https://user-images.githubusercontent.com/3581558/73986069-cc78ce00-494d-11ea-89e8-ae46e396856c.png)

![Screen Shot 2020-02-07 at 00 59 48](https://user-images.githubusercontent.com/3581558/73986079-d1d61880-494d-11ea-9380-81e0432af491.png)

![Screen Shot 2020-02-07 at 01 01 02](https://user-images.githubusercontent.com/3581558/73986091-d5699f80-494d-11ea-997b-440da4566212.png)

![Screen Shot 2020-02-07 at 01 00 38](https://user-images.githubusercontent.com/3581558/73986096-d7336300-494d-11ea-91f8-9a86c0919f4d.png)


