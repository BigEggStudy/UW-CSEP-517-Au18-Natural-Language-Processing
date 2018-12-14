# Recurrent Neural Networks, Attention, and Reading Comprehension
The code for this assignment.
## Files
+ model.py                      The different RNN model's implementation

### RNN Outputs
Use the mean of the two different direction's output from bi-directional RNN.

`output = (𝑜𝑢𝑡𝑝𝑢𝑡𝑠_𝑓𝑤 + 𝑜𝑢𝑡𝑝𝑢𝑡𝑠_𝑏𝑤) / 2`

### Bonus Part
In the bonus question of this assignment I try to increase the layer of the RNN. So you can find the method (named: `rnn_gru`, line 181) to build the Recurrent Neural Network have a parameter as `layer`. And this value set at line 95.

For the assignment 3, just set the `layer = 1` will performance the normal version of RNN with Attention. And by setting it to `2` or `4`, can start the experiments on Bonus questions.

## Result
Results is too large to be included in the upload package. Please check the report file: HW4.pdf

# Environment
Use Python 3.6.5
```
python --version
Python 3.6.5
```
Test in Windows
