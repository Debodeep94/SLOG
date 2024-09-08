$\bullet$ In order to train the language model for the first time, use the mimic_train.sh (command: sbatch mimic_train.sh)

$\bullet$ Training of the surrogate models are done using the mimic_test.sh file. The codes are available at the tester.py file available in the modules folder. 

$\bullet$ For Finetuning, please run the mimic_ft.sh file and corresponding codes are availble at the finetuner.py file. 

$\bullet$ To access the weights of the pre-trained VLM, the surrogate 1 and surrogate 2, please [click here](https://drive.google.com/drive/folders/1Qchyt7dieWFeN4Kr_UTvpqEn2g4paH2a?usp=sharing)

The codes are based on the following papers and github.

Paper - [Generating Radiology Reports via Memory-driven Transformer](https://arxiv.org/abs/2010.16056)

Codes - [R2Gen](https://github.com/zhjohnchan/R2Gen)

Paper - [Learning to Generate Clinically Coherent Chest X-Ray Reports](https://aclanthology.org/2020.findings-emnlp.110.pdf)

Codes - [Click here](https://github.com/justinlovelace/coherent-xray-report-generation)


For the chexpert evaluation, please follow these instruction:
[Click here](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt/chexpert)