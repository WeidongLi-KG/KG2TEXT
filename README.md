# KG2TEXT
## This codebase contains PyTorch implementation of the paper:
Knowledge graph based natural language generation
with adapted pointer-generator networks
> We build our model on the top of PointerTP: https://github.com/EagleW/Describing_a_Knowledge_Base, and the datasets can also be downloaded from that repo.

## Dependencies
* Python 3.6
* PyTorch >= 0.4
* Pandas
* Numpy
* nvidia-smi


## Usage

Step 1. Randomly split the data into train, dev and test by runing split.py under utils folder:

```
python split.py
```

Step 2. Run preprocess.py under the same folder. You can choose person (type 0) or animal (type 1):
```
python preprocess.py --type 1
```
Step 3. Pretrain (for type Animal): 
```  
python main.py --cuda --mode 0 --type 1
```

