
get data set:


```bash
mkdir -p data/raw/
curl -o data/raw/simple-examples.tgz http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz 
```

unzip data
```bash
mkdir -p data/processed/
tar zxvf data/raw/simple-examples.tgz -C data/processed/
```