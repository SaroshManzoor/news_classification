# <center> News Article Classification


This repository provides two main functionalities:
### 1. Pipeline: 
   This includes the following steps
   - Train the model 
   - Assign categories to the articles in "CPS_use_case_classification_response.json"
   - Dump the predictions in a separate "*predictions.json*" file <br> <br>

### 2. Live Inference: 
This is a rudimentary UI to predict the categories for a user-defined headline.
<br>
<br>

The "headline" attribute of news articles is used as the main predictor. An approach with full or partial article text, scrapped from the corresponding "link" was considered but not implemented due to the lack of time.
<br>
<br>
Note: Current default arguments of pipeline are optimized for training speed and not best accuracy score. See the performance section below for details.

## <center> Usage 
<br>

Pipeline and Live Inference can be accesses in two ways.
### 1. Via Docker:
This will bypass all dependency management but depending on the models used, could take anywhere from 5 to 30 minutes to complete the pipeline, as GPU support is not guaranteed.
With the current defaults, it should take about 5 minutes for the pipeline to run.


To run the end-to-pipeline i.e. execute the following in a terminal:
```
chmod +x run_pipeline.sh
./run_pipeline.sh
```

the resulting file will be stored in: news_classification/storage/data/predictions.json <br>
  
To perform inference on individual news articles manually, run:
```
chmod +x run_inference.sh
./run_inference.sh
```
then navigate to: http://0.0.0.0:8000


### 2. Local Environment:
This  would be much faster, especially if GPU is available. However, it necessitates that virtual environment is correctly generated.

Set-up virtual environment:

  ```
  poetry env use $(pyenv local)
  poetry install
  ```


To run the full pipeline, run:
  ```
  python news_classification/pipeline.py
  ```
For inference app, run:
   ```
   streamlit run news_classification/live_inference.py
   ```
then navigate to the "Local URL" printed on the terminal 


## <center> Performance

With the default arguments of the pipeline, a balanced accuracy score of ~44% is achieved. The default arguments are set to make the pipeline run faster. <br>

The highest balanced accuracy (i.e. average recall over all classes) observed in a local environment is around 61%. This was achieved by the combination of "BoostingClassifier" and "EmbeddingsExtractor". <br>

The choice of classifier and feature extractor, among other settings, can be set in: <br>
  ```
  news_classification/config/pipeline_config.yml
  ```
"EmbeddingsExtractor" generally outperforms other feature extractors but can be very slow during a docker run. <br>
For a quick-run, set the feature_extractor_name parameter of the pipeline_config.yml to "TfIdfExtractor".
<br>
<br>

After the training, the classifier performance can be viewed in the model_registry:
  ```
  news_classification/storage/model_registry/<classifier_name>
  ```
Along with the trained model file, an Excel file containing the accuracy and the balanced accuracy and an image displaying the 
confusion matrix can be found in the model registry.
