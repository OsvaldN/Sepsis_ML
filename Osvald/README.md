## Bo Wang Lab submission for the PhysioNet/CinC Challenge 2019

## Contents

This prediction code uses two scripts:

* `get_sepsis_score.py` makes predictions on clinical time-series data.  Add your prediction code to the `get_sepsis_score` function.  To reduce your code's run time, add any code to the `load_sepsis_model` function that you only need to run once, such as loading weights for your model.
* `driver.py` calls `load_sepsis_model` and `get_sepsis_score` and performs all file input and output. -- provided by PhysioNet

Check the code in these files for the input and output formats for the `load_sepsis_model` and `get_sepsis_score` functions.

* `utility.py` contains evaluation functions provided by PhysioNet. `util.py` contains additional functions for plotting progress.

* `TCN.py` and `TCN_train.py` contain a Temporal Convolutional Network and its training script, respectively.  `LSTM.py` and `LSTM_train.py` contain an LSTM network and its training script, respectively.

## Use

You can run this prediction code by installing packages in requirements.txt and running

    python driver.py input_directory output_directory

where `input_directory` is a directory for input data files and `output_directory` is a directory for output prediction files.  The PhysioNet/CinC 2019 webpage provides a training database with data files and a description of the contents and structure of these files.

## Details

See the PhysioNet/CinC 2019 webpage for more details, including instructions for the other default files in this repository.
