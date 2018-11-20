# NER with Structure Perceptron
The code for this assignment.
## Files
+ start.py                      The start point of the program
+ perceptron.py                 The structure perceptron with viterbi decoding

## Result
+ Result-HW3.txt                The program output file
+ output\1\dev.smal.output*     The evaluation output for dev small data set with different iteration
+ output\1\test.smal.output*    The evaluation output for test small data set with different iteration
+ output\2\dev.smal.ablation_*  The evaluation output for dev small data set with different ablation feature study
+ output\2\test.smal.ablation_* The evaluation output for test small data set with different ablation feature study

For running the program, just go to the **code** folder, and run the following command:
```
python start.py
```
The running will take few minutes, and the result should be as same as the txt file. And the program will run the problem as the sequence in the assignment.

# Helper files
+ util.py                       The helper functions for the program
+ constant.py                   The constant for the program

# Environment
Use Python 3.6.5
```
python --version
Python 3.6.5
```
Test in Windows
