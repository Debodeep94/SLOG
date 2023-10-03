# SLOG
## Objective:

The general idea is to produce a clinical reports from chest x-ray images with human scores.  
## Pipeline-

\$bullet$ We train the baseline LLM with the file mimic_train.sh.

\$bullet$ Once trained, we train our surrogate model (a regression model) based on the latent distribution of the trained data and the corresponding information scores. The codes for training the surrogate is provided in the tester.py files. 

$\bullet$ Once the the surrogate is trained, we finetune the LLM along with the help of the surrogate model. At the time of finetunig, our hybrid loss function is-

$$
L_{hdm} = CE_{LLM}+\lambda*I, \text{where I is the mean information score obtained from the surrogate}
$$

The codes for finetuning is provided in the finetuner_train.py file. 

