## Evaluation
The following command is used to generate results of DUMoE on the real HSI data for the SCI tasks:
```
python main_test.py --method=dumoe --data_root="./data/SCI" --outf="./results/dumoe/" --input_setting="Y" --input_mask='Phi'
```
Place the real HSI data in the folder "./data/SCI". 
The results will be saved in the folder "./results/dumoe/".