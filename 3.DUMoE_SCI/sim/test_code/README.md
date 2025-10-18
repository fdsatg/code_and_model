## Evaluation
The following command is used to generate results of DUMoE on the simulation HSI data for the SCI tasks:
```
python main_test.py --method=dumoe --data_root="./data/SCI" --outf="./results/dumoe/" --input_setting="Y" --input_mask='Phi'
```
Place the KAIST dataset in the folder "./data/SCI".
The results will be saved in the folder "./results/dumoe/".