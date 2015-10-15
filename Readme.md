
# machine-translation

This code implements GRU for training/sampling from word-level language models. The model learns to translation english to france in a sequence. 

This code was originally based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6), which is in turn based on [learning to execute](https://github.com/wojciechz/learning_to_execute) code from Wojciech Zaremba. 

### Training

Start training the model using `train-zzd.lua`, for example:

```
$ th train-zzd.lua -data_dir data/en-fr -gpuid -1
```
