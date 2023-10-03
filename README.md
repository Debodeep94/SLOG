# SLOG
## Objective:

The general idea is to produce a clinical reports from chest x-ray images with human scores.  
## Pipeline-

$\bullet$ We train the baseline LLM with the file mimic_train.sh. The codes are provided in the trainer.py file.
  The baseline code is taken from the [this](https://github.com/zhjohnchan/R2Gen) github repository and for the corresponding paper, please click [here](https://arxiv.org/pdf/2010.16056.pdf).
$\bullet$ Once trained, we train our surrogate model (a regression model) based on the latent distribution of the trained data and the corresponding information scores. The codes for training the surrogate is provided in the tester.py files. 

$\bullet$ Once the the surrogate is trained, we finetune the LLM along with the help of the surrogate model. At the time of finetunig, our hybrid loss function is-

$$
L_{hdm} = CE_{LLM}+\lambda*I, \text{where I is the mean information score obtained from the surrogate}
$$

The codes for finetuning is provided in the finetuner_train.py file. 

$\bullet$ For final testing, please run mimic_surr.sh file and the codes are provided in surrogate_tester.py file. 

