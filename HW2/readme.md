# Bigram HMM
The code for 1st problem in the assignment.
## Files
+ StartPoint-BigramHMM.py       The start point of the program
+ BigramHMM.py                  The Bigram HMM without any smoothing
+ BigramHMMWithAddK.py          The Bigram Language Model with Add-K Smoothing
+ BigramHMMWithLI.py            The Bigram Language Model with Linear Interpolation Smoothing
+ BigramHMMWithSmoothing.py     The Bigram Language Model with Add-K & Linear Interpolation Smoothing

## Result
+ Result-BigramHMM.txt              The program result file
+ ConfusionMatrix-BigramHMM.txt     The confusion matrix of the Bigram HMM

For running the program, just go to the **code** folder, and run the following command:
```
python StartPoint-BigramHMM.py
```
The running will take few hours, and the result should be as same as the txt file. And the program will run the problem as the sequence in the assignment.

# Trigram HMM
The code for 2nd problem in the assignment.
## Files
+ StartPoint-TrigramHMM.py      The start point of the program
+ TrigramHMM.py                 The Trigram HMM without any smoothing
+ TrigramHMMWithAddK.py         The Trigram Language Model with Add-K Smoothing
+ TrigramHMMWithLI.py           The Trigram Language Model with Linear Interpolation Smoothing
+ TrigramHMMWithSmoothing.py    The Trigram Language Model with Add-K & Linear Interpolation Smoothing

## Result
+ Result-TrigramHMM.txt             The program result file
+ ConfusionMatrix-TrigramHMM.txt    The confusion matrix of the Trigram HMM

For running the program, just go to the **code** folder, and run the following command:
```
python StartPoint-TrigramHMM.py
```
The running will take few hours, and the result should be as same as the txt file. And the program will run the problem as the sequence in the assignment.

# Helper files
+ AssignmentHelper.py           The helper functions for the program
+ Constant.py                   The constant for the program
+ MeasureAccuracy.py            The helper class that compute the measure related statistic

# Environment
Use Python 3.6.5
```
python --version
Python 3.6.5
```
Test in Windows