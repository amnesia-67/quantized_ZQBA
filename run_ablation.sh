#!/bin/bash

# Generate the data splits
python generate_8_splits.py

# Train the single baseline Target model (FP32 ResNet-34)
echo "Training Fixed Target Model..."
python train_ablation.py --role target \
    --split_file data_splits_8/Target_Fixed_40k.pt \
    --save_name Target_Fixed.pth

# Train the 8 Surrogates (INT8 QAT ResNet-18)
for overlap in 100 85 70 55 40 25 10 0
do
   echo "Training Surrogate with ${overlap}% Overlap..."
   python train_ablation.py --role surrogate \
       --split_file data_splits_8/Surrogate_${overlap}_Overlap.pt \
       --save_name Surrogate_${overlap}.pth
done

echo "All 9 models trained successfully."