## Evaluation
The following command is used to generate results of DUMoE at a sampling ratio of 0.10 for the CS-MRI tasks:
```
python main_test.py --model=dumoe --ratio=10 --dataset=Brain_test --result_dir=results --mask_type=Radial --input_size=256
```
Place the Brain dataset in the folder "data/test" and put the Pseudo Radial masks in the folder "./data/mask/256".
The results will be generated in the folder "./{result_dir}/{model}/{dataset}/{ratio}/", where results.csv file will save the results in the format "{Image Name},{PSNR},{SSIM},{LPIPS}".
