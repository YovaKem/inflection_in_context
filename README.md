## Winning system for Task 2 of CoNLL--SIGMORPHON 2018 on morphological inflection in context 

Train a multi-lingual model, e.g. for English and German
```
python3 src/train.py en,de trainsets/{}-track1-low devsets/{}-track1-covered devsets/{}-uncovered test_exp training
```

Finetune a trained multi-lingual model for a particular language, e.g. English. The code also accepts 
```
python3 src/train.py en,de trainsets/{}-track1-low devsets/{}-track1-covered devsets/{}-uncovered test_exp finetuning en
```

Predict on test data, e.g. with finetuned English model for the low resource setting
```
python3 src/train.py en,de trainsets/{}-track1-low devsets/{}-track1-covered devsets/{}-uncovered test_exp_en testing en low 
```

