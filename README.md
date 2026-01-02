# IW Report Documentation

Installation Instructions:

Step 1: Prepare virtual environment

```
py -m venv venv
./venv/scripts/activate
pip install -r requirements.txt
```

Step 2: Download data and models from google drive

Google Drive Link: [https://drive.google.com/drive/folders/1_oOLZvf5pH8sX0rFASQpw-rWORMU8S3v?usp=sharing](https://drive.google.com/drive/folders/1_oOLZvf5pH8sX0rFASQpw-rWORMU8S3v?usp=sharing)

Save the models.pkl file into the 'training' folder and the rest of the files into a folder titled data.

Step 3: After completion of steps 1 and 2, all files should be in place to continue building upon the project

##

### Explanation of Files

Scripts Folder: Contains all relevant scripts to scraping data, exploratory data analysis and data cleaning.

datascaping.ipynb: First notebook used for pulling data from pybaseball.

eda_1-3.ipynb: Three notebooks for exploring the dataset and understanding the dataset and how it may be used.

cleaning_1.ipynb: Initial cleaning of the dataset for some of the EDA.

cleaning_final.ipynb: Contains all data preparation steps to form the entire dataset, generating the clean.csv file in the google drive.

Training folder: Contains all files related to training the models and evaluation of the models.

decision_tree.py, neural_network.py, svm.py, random_forest.py, hierarchical.py: Modules containing functions for training and evaluating models. (Note: The hyperaparameters for these models were found via grid search, but this code was removed after the chosen hyperparameters were found).

train_all.ipynb: Notebook that calls training and evaluation modules for training all the models. Saves outputs to output folder and the models to a pkl file in the Google drive (models.pkl).

presentation.ipynb: Notebook used for summarizing findings from the project to present.

util: Folder with helper functions for training the models.

##

Project by Mark Rubin

Acknowledgements:

Professor Xiaoyan Li, TA Andre Rubungo, classmates in IW Section 01, and ChatGPT (as cited in individual files where applicable in accordance with course and university policy strictly for getting example code to modify).

I pledge that this repository represents my own work in accordance with university policies. - Mark Rubin
