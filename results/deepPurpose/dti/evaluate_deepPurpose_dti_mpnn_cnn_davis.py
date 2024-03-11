from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
import warnings
from lifelines.utils import concordance_index
warnings.filterwarnings("ignore")

X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False,
                                                  convert_to_log = True, # Enable for DAVIS dataset
                                                   threshold = 30)
print('Drug 1: ' + X_drugs[0])
print('Target 1: ' + X_targets[0])
print('Score 1: ' + str(y[0]))

# DTI prediction framework
drug_encoding, target_encoding = 'MPNN', 'CNN'
#drug_encoding, target_encoding = 'Morgan', 'Conjoint_triad'

# Data preparation
train, val, test = utils.data_process(X_drugs, X_targets, y,
                                drug_encoding, target_encoding,
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
train.head(1)

# Model configuration generation
config = utils.generate_config(drug_encoding = drug_encoding,
                         target_encoding = target_encoding,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 50,
                         LR = 0.001,
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3,
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )

# Model initialization
model = models.model_initialize(**config)
model

# Model training
model.train(train, val, test)

# Load dataset
import pandas as pd

test_data = pd.read_csv('../../../cancer_datasets/breast_cancer/test_new.csv')
test_data

# Predict
y_pred = []
for i in range(len(test_data)):
  X_drug = [test_data['drug'][i]]
  X_target = [test_data['target'][i]]
  y = [test_data['pIC50'][i]]
  X_pred = utils.data_process(X_drug, X_target, y,
                                  drug_encoding, target_encoding,
                                  split_method='no_split')
  pred = model.predict(X_pred)
  y_pred.append(pred)
  print('The predicted score is ' + str(pred))

# Evaluation metrics - MSE, R2, PCC, CI
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Assuming y_true and y_pred are your data
y_true = np.array(test_data['pIC50'])
y_pred = np.squeeze(y_pred)

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R2
r2 = r2_score(y_true, y_pred)
print(f'R-squared: {r2}')

# Calculate PCC
pcc = np.corrcoef(y_true, y_pred)
print(f'Pearson Correlation Coefficient: {pcc}')

# Calculate CI
ci = concordance_index(y_true, y_pred)
print(f'Concordance Index: {ci}')

pd.DataFrame(y_true).to_csv('y_true.csv', index=False, header=None)
pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False, header=None)