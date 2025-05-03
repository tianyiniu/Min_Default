### **⚠️ Repo currently under construction!**
More detailed documentation will be uploaded promptly &  current codebase will be cleaned up to provide a more consistent interface. Stay tuned!


# File Directory

`EqualDefault/`, `MajDefault/`, `MinDefault/` provides the training and testing stimuli for the frequency conditions. 

`Feature-files/` contains the phonological features used. 

`lr.py`: suite to tools to train and obtain heatmaps and accuracy curves

`lr_evaluation.py`: suite of tools to analyze failure cases on different test stimuli 

`Transformer_BCE_train.ipynb` trains the Transformer Encoder, `Transformer_BCE_test.ipynb` test the Transformer Encoder on different test stimuli. `Transformer_BCE_visualization.ipynb` creates saliency maps to analyze feature sensitivity. 

`LSTM.ipynb` trains and test the LSTM Encoder-Decoder model. 

# Data distribution across default conditions

## Equal Freq 'regular' 
| | training_cat1/2 | training_cat3 | testing_cat1/2 | testing_cat3 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| cvc | 132 | 99 | 32 | 24  | 
| cvcvc | 132 | 99 | 32 | 24  | 
| vcvc | 132 | 99 | 32 | 24  | 
| cvcv | 0 | 99 | 0 | 24  | 
| total | 396 + 396 | 396 | 96 | 96


## Min Default 'regular' 
| | training_cat1/2 | training_cat3 | testing_cat1/2 | testing_cat3 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| cvc | 177 | 31 | 32 | 24  | 
| cvcvc | 177 | 31 | 32 | 24  | 
| vcvc | 177 | 31 | 32 | 24  | 
| cvcv | 0 | 31 | 0 | 24  | 
| total | 531 + 531 | 124 | 96 | 96    

## Maj Default 'regular' 
| | training_cat1/2 | training_cat3 | testing_cat1/2 | testing_cat3 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| cvc | 39 | 238 | 32 | 24  | 
| cvcvc | 39 | 238 | 32 | 24  | 
| vcvc | 39 | 238 | 32 | 24  | 
| cvcv | 0 | 238 | 0 | 24  | 
| total | 117 + 117 | 952 | 96 | 96


## Equal Freq test, L, and Mutants
|  Template | Amount | 
| ------------- | ------------- | 
| cvc | 32 | 
| cvcvc | 32 |
| vcvc | 32 | 
| cvcv | 0 | 


Mutants: words from "regular" training with the last consonant C changed as follows:
If C belongs to class 1: randomly change it either to class 2 or class 3
If C belongs to class 2: randomly change it either to class 3 or class 1
If C belongs to class 3: randomly change it either to class 2 or class 1


## New templates
|  | Template | Amount | 
| ------------- | ------------- | ------------- | 
| vc | 20 | 20 |
| cvcc | 20 | 20 | 
| ccv |  | 20 |

-h final words are "mutants" of templates cvc and cvcvc
