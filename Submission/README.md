### Authors:      
Daniel-Florin Dosaru   
Natasha Ørregaard Klingenbrunn   
Sena Necla Cetin   


### How to create a submission (out.csv file) using our best method
In the same folder:    
`unzip test.csv.zip`    
`unzip train.csv.zip`    
`python3 run.py`   

##### The `run.py` script:
- Loads the training data
- Splits training data with ratio 0.8
- Runs ridge regression to compute the combination of degree / lambda with the lowest validation MSE
- Predicts labels for the test data
- Saves this predictions in the `out.csv` submission file

Please note that the degree and lambda values fluctuate very slightly between runs (hence there may be a small descepancy between the recorded parameter values in the report and those calculated by run.py) though the output consistently has an accuracy of 78.9%
