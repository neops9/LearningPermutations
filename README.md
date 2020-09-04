## Faststuff setup

python3 setup.py build_ext --inplace

## Options

```
Options:
  -h, --help                Help screen
  --model                   Path to the model
  --data                    Path to the data
  --lang                    Language of the dataset
  --embeddings              Path to the embedding file
  --keep-mwe                Keep multi-word expressions order
  --program                 Path to the re-ordering program executable
  --algo-version (=n6)      Version of the re-ordering algorithm you want to use: n4, n6, n7
  --format (conllu)         Format of the input data
  --storage-device (cpu)    Device where to store the data
  --device (cpu)            Device to use for computation
  --verbose                 Verbose
  --bias (0.0)              Bias added to the bigram weights of the original order 
  --output                  Output for the re-ordered file
  --left-limit (-1)         Distance limit of the left item in the re-ordering algorithm
  --right-limit (-1)        Distance limit of the rigth item the re-ordering algorithm
  --name                    Name of the corpus (useful when running multiple instance of the script)
```

Example:

```
python permute.py 
    --model ./models/en_gum_exp_mcmc_2opt_loss_structured_explr_pd03_id05_proj4096_model_best.pth.tar 
    --data ./data/test/dev.conllu 
    --lang test 
    --embeddings ./embeddings/test.vec 
    --program ./reordering_program
```

To run the **O(n)** version of the algorithm set options "left-limit" and "right-limit" to 0.

## Run cross-lingual dependency parser

**(!) Il faut inverser les colonnes 4 et 5 des fichiers permutés pour les lancer sur le parseur**

```
awk -v FS='\t' -v OFS='\t' '{ t = $4; $4 = $5; $5 = t; print; }' test.conllu> test.conllu.tmp ; mv test.conllu.tmp test.conllu
```

### SelfAtt-Graph & RNN-Graph

```
python /examples/analyze.py 
    --parser biaffine 
    --ordered 
    --gpu 
    --test ./test.conllu 
    --decode mst
    --model_path /pretrained_models/final_{grnn.sh_1 | gtrans.sh_1}
    --out_filename whatever
    --model_name network.pt
    --punctuation 'PUNCT' 'SYM'
    --extra_embed /ud2_embeddings/wiki.multi.{lang}.vec
```

### SelfAtt-Stack & RNN-Stack

```
python /examples/analyze.py 
    --parser stackptr 
    --beam 5
    --ordered 
    --gpu 
    --test ./test.conllu 
    --decode mst
    --model_path /pretrained_models/final_{srnn.sh_1 | strans.sh_1}
    --out_filename whatever
    --model_name network.pt
    --punctuation 'PUNCT' 'SYM'
    --extra_embed /ud2_embeddings/wiki.multi.{lang}.vec
```
