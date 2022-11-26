# Model Card

## Model Details

This model aims to classificate salary using Census Data into <=50K or >50K.

## Intended Use

Prediction of salary based on demographic data from Census.

## Training Data

For training data it was used random 80% from full dataset using train_test_split.

## Evaluation Data

The remaining 20% data was used as test data.

## Metrics

Evaluated metrics:

* Precision: TP / (TP + FP)
* Recall: TP / (TP + FN)
* FBeta: The FBeta balances the precision and recall.

| Group | Precision | Recall | FBeta |
|-------|-----------|--------|-------|
| Train |   78.7%   | 59.1%  | 67.5% |
| Test  |   78.0%   | 58.2%  | 66.7% |

## Ethical Considerations

This model uses colected data from Census and use demographic informations to predict salary. It may have instrinsic bias due to historical society issues. So it's important be careful on it's uses.

## Caveats and Recommendations

Due to it's potencial bias it's recommended to check slices of data to be aware of potencial discriminatory issues on it's uses. You can check the same metrics above on slices of data in: `output/slice_output.txt`
