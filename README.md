# News Article Classification

The repository already contains the data and the trained models. <br>
There are several ways to interact with the artifacts:


## Usage

- To run the end-to-pipeline i.e. 
  - Train the model 
  - Assign categories to the articles in "CPS_use_case_classification_response.json"
  - Dump the predictions in a separate "*predictions.json*" file <br> <br>
    
  execute the following in a terminal:
    ```
    chmod +x run_pipeline.sh
    ./run_pipeline.sh
    ```
    the resulting file will be stored in 
    ```
    news_article_classification/storage/data/predictions.json
    ```
  (Disclaimer: Tested in MacOS only)

  <br><br>
  
  

- To re-train the model and view the evaluation: <br> <br>

    Set-up virtual environment: 
    ```
    poetry env use $(pyenv local)
    poetry install
    ```
  
  ```
   
  ```
  
  after retraining, individual inference can be performed by the following: <br>
  ```
   
  ```
