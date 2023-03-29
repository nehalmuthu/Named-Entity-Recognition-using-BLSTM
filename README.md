# Named-Entity-Recognition-using-BLSTM


Setting up the python environment:
In cmd
1. create the virtual environment: python -m venv ./venv
2. activate the virtual env
3. pip install requirements.txt
4. change path to the submission folder
5. Copy paste the data and Glove folder into the submission folder.
6. Follow steps as below

For more detailed steps and image of the directory structure, please check the last section "Run the code" in Report.

We have separate scripts for TASK 1 and TASK 2:

To Run the Code:

TASK 1:

1) 'task1-prediction.py'
For prediction on dev and test data, use the script 'task1-prediction.py.'
Place the folders Data, Glove, and Model in the same directory where the script is present  (OR) You can also directly give the filename in the code 'task1-prediction.py' in the first few lines.
The 'model' folder contains the model 'blstm1.pt' for TASK 1
Simply run the file using the command "python task1-prediction.py."
The script will create 'test1.out' and 'dev1.out' files in the same directory.


2) 'TASK1.ipynb'
'TASK1.ipynb' contains all the code from data preparation, model, training, and prediction on dev and test
Upload the file to Google Drive and open it on Colab
Upload the files 'train,' 'dev,' and 'test' in Colab
The Notebook is documented, so run the cells one by one to see all results


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TASK 2:

1) 'task2-prediction.py'
For prediction on dev and test data, use the script 'task2-prediction.py.'
Place the folders Data, Glove, and Model in the same directory where the script is present (OR) You can also directly give the filename in the code 'task2-prediction.py' in the first few lines
The 'model' folder contains the model 'blstm2.pt' for TASK 2
Simply run the file using the command "python task2-prediction.py."
The script will create 'test2.out' and 'dev2.out' files in the same directory


2) 'TASK2.ipynb'
'TASK2.ipynb' contains all the code from data preparation, model, training, and prediction on dev and test
Upload the file to Google Drive and open it on Colab
Upload the files 'train', 'dev', and 'test' in Colab
The GloVe path should be given in the Notebook. We can upload it and use the path
The Notebook is documented, so run the cells one by one to see all results

