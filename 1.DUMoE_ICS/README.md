## Evaluation
The following command is used to generate results of DUMoE at a sampling ratio of 0.25 for the ICS tasks:
```
python main_test.py --model=dumoe --result_dir=results --dataset='' --ratio=25
```
Put the test datasets for ICS in the folder "data/test" and replace the option "dataset" with the name of the dataset folder.
The results will be generated in the folder "./{result_dir}/{model}/{dataset}/{ratio}/", where results.csv file will save the results in the format "{Image Name},{PSNR},{SSIM},{LPIPS}".
