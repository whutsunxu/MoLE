## Tasks Table
| task | status |
|---------|---------|
| run one of the cases completely on cpu device | done (MoLE_DL)|
| delete the randomness if any in inputs/added noise/strategy | done |
| clarify the relationship between the model(train/inferrence) paramaters and key operators paramaters, maintain a table | done |
| re-run the same cases to check if the results are stable and correct | done |
| set up op unit test to check the basic function and report issue if any | X |
| replace one by one to check the model function and debug at model level if any | X |
| stress test the model test | X |
| simplify the test script(fuse bash and python scripts) | X |
| repeat the above steps to all others inference models | X |
| prepare data/patches and write report for the above stage | X |
| do performance analysis and list fusion/refinement task | X |
| do fusion/refinement task and write report for the above stage | X |


## Models Parameters Table
| model | seq_len | label_len | pred_len | patch_len | stride | kernel_size | d_model | n_heads | batch_size | enc_in | udefined_v | t_dim |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| MoLE_DLinear-Inference | 336 | 336 | 96 | 16 | 1 | 25 | 512 | 8 | 8 | 321 | 4 | 1 |

## OP List and Info
| OPs | input0 name | input0 dtype | input0 dims | input1 name | input1 dtype | input1 dims | other configs | output0 name | output0 dtype | output0 dims | 
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| zeros_like | input | fp32 | (8, 96, 321) | N | N | N | N | output | fp32 | (8, 96, 321) | 
| cat | input0 | fp32 | (8, 336, 321) | input1 | fp32 | (8, 336, 321) | dim = 1 | output | fp32 | (8, 432, 321) |
| slice | input | fp32 | (8, 336, 4) | N | N | N | dim = 1, offset = 0, size = 1 | output | fp32 | (8, 4) |
| moving_avg(avgpool1d) | input | fp32 | (8, 336, 321) | N | N | N | avg_dim = 1, kernel=25, stride = 1, padding_low = 12, padding_high = 12, padding_value = boundary_value, dilation = 1| output | fp32 | (8, 336, 321) |
| slice | input | fp32 | (8, 336, 321) | N | N | N | dim = 1, offset = 0, size = 1 | output | fp32 | (8, 1, 321) |
| repeat | input | fp32 | (8, 1, 321) | N | N | N | dim = 1, times = 12 | output | fp32 | (8, 12, 321) |
| cat | input0/input2 | fp32 | (8, 12, 321) | input1 | fp32 | (8, 336, 321) | dim = 1 | output | fp32 | (8, 360, 321) |
| transpose | input | fp32 | (8, 360, 321) | N | N | N | permutate = {0, 2 ,1} | output | fp32 | (8, 321, 360) |
| avgpool1d | input | fp32 | (8, 321, 360) | N | N | N | avg_dim = 1, kernel=25, stride = 1, padding = 0| output | fp32 | (8, 321, 336) |
| transpose | input | fp32 | (8, 321, 336) | N | N | N | permutate = {0, 2 ,1} | output | fp32 | (8, 336, 321) |
| substract | input0 | fp32 | (8, 336, 321) | input1 | fp32 | (8, 336, 321) | N | output | fp32 | (8, 336, 321) |
| transpose | input | fp32 | (8, 336, 321) | N | N | N | permutate = {0, 2 ,1} | output | fp32 | (8, 321, 336) |
| linear | input0 | fp32 | (8, 321, 336) | input1 | fp32 | (336, 96) | N | output | fp32 | (8, 321, 96) |
| add | input0 | fp32 | (8, 321, 96) | input1 | fp32 | (8, 321, 96) | N | output | fp32 | (8, 321, 96) |
| linear | input0 | fp32 | (8, 1, 4) | input1 | fp32 | (4, 321) | N | output | fp32 | (8, 1, 321) |
| ReLU | input0 | fp32 | (8, 1, 321) | N | N | N | N | output | fp32 | (8, 1, 321) |
| linear | input0 | fp32 | (8, 1, 321) | input1 | fp32 | (321, 321) | N | output | fp32 | (8, 1, 321) |
| reshape | input0 | fp32 | (8, 1, 321) | N | N | N | N | output | fp32 | (2568, 1) |
| head_dropout | input0 | fp32 | (2568, 1) | N | N | N | p = 0.5; binary_mask = rand_like(input0) > p; input0 * binary_mask + (1 - binary_mask) * (-1e20) | output | fp32 | (2568, 1) |
| softmax | input0 | fp32 | (2568, 1) | N | N | N | dim = 1 | output | fp32 | (2568, 1) | 
| reshape | input0 | fp32 | (8, 336, 321) | N | N | N | N | output | fp32 | (8988.0, 96, 1) |
| matmul | input0 | fp32 | (2568, 96, 1) | input1 | fp32 | (2568, 1, 1) | N | output | fp32 | (2568, 96, 1) |
| reshape | input0 | fp32 | (2568, 96, 1) | N | N | N | N | output | fp32 | (2568, 96) |
| reshape | input0 | fp32 | (2568, 96, 1) | N | N | N | N | output | fp32 | (8, 321, 96) |
| transpose | input | fp32 | (8, 321, 96) | N | N | N | permutate = {0, 2 ,1} | output | fp32 | (8, 96, 321) |