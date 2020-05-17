## MGMC - Multigraph Geometric Matrix Completion (MGMC)

### 1. Requirements
To make sure you have the right packages.
```
pip install -r requirements.txt
```
### 2. Dataset
Make sure dataset in `./data/` folder. Alternatively, you can load your own 
dataset by implementing the `load_data` method after inheriting 
`MultiGMCTrainer` class from `trainer.py`. Just make sure you change the 
following variables below. Please check `MultiGMCTrainerPPMI` in `trainer.py`
 as an example.
```
class MultiGMCTrainerPPMI(MultiGMCTrainer):
    # ... 
    def load_data(self):
        # Just make sure you change the following variables accordingly.
        self.data_x = data_x
        self.data_y = data_y
        self.data_meta = data_meta
        self.M = M
        self.initial_tr_mask = initial_tr_mask
        self.Lrow = Lrow
        self.A_matrix_row = A_matrix_row
        self.impute_idx = impute_idx
```
### 3. Train model
Now we are ready to train the model including hyperparameter optimization.
```
python main.py --train_hyperopt=1
```

### 4. Test model
Test the model given the hyperparameters from the previous step.
```
python main.py --train_hyperopt=0
```

### 5. Check output results
Output results will be saved at `./gmc_output/`. You will find a pickle file 
containing a list of 4 elements (experiments from randomly dropping of 
entries. Each element is a list of outputs coming from the 
outputs from the ten-fold Stratified-CV.
```
# load pickle file
with open('./gmc_output/output.pkl', 'rb') as load_file:
    loaded_pickle_obj = pickle.load(load_file)

loaded_pickle_obj # this is now a list containing 4 lists
loaded_pickle_obj[0] # this is a list with 10 elements
```

`loaded_pickle_obj[0][0]` is a tuple with the following information:
```
(ground_truth_test_set, 
 predicted_probabilities_test_set, 
 predicted_labels_test_set, 
 RMSE)
 ```
    
### Citation
If you use this code, please cite:
```
@misc{vivar2020simultaneous,
    title={Simultaneous imputation and disease classification in incomplete medical datasets using Multigraph Geometric Matrix Completion (MGMC)},
    author={Gerome Vivar and Anees Kazi and Hendrik Burwinkel and Andreas Zwergal and Nassir Navab and Seyed-Ahmad Ahmadi},
    year={2020},
    eprint={2005.06935},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```