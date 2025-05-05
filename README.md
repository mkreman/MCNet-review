# MCNet-review
Testing the Official implementation of MCNet: Rethinking the Core Ingredients for Accurate and Efficient Homography Estimation.

## Useful Commands
```
conda deactivate
conda deactivate
conda activate cv2
```

## Test the model on Custom Images pair
```
python test_2_images.py --img1 '/home/mayank.mds2023/CV/MCNet/Photos-002/20250402_135410.jpg' --img2 '/home/mayank.mds2023/CV/MCNet/Photos-002/20250402_135412.jpg' --note 412

python test_2_images.py --img1 '/home/mayank.mds2023/CV/MCNet/Photos-002/20250408_103236.jpg' --img2 '/home/mayank.mds2023/CV/MCNet/Photos-002/20250408_103240.jpg' --note 236

python test_2_images.py --img1 '/home/mayank.mds2023/CV/MCNet/Photos-001/20250423_130011.jpg' --img2 '/home/mayank.mds2023/CV/MCNet/Photos-001/20250423_130015.jpg' --note 1115
```

## Evaluate

Please modify the dataset path in datasets.py and run the following code with situable variables:
```bash
python test.py --gpuid ${GPU_ID} --dataset ${DATASET} --checkpoints ${WEIGHT_PATH}
```
