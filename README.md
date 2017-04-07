# Tree-Structured Long Short-Term Memory Networks
A [PyTorch](http://pytorch.org/) based implementation of Tree-LSTM from Kai Sheng Tai's paper
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory
Networks](http://arxiv.org/abs/1503.00075).

### Requirements
- [PyTorch](http://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 2.7

### Usage
First run the script `./fetch_and_preprocess.sh`, which downloads:
  - [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness task)
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!
  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)

The preprocessing script also generates dependency parses of the SICK dataset using the
[Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).

To try the Dependency Tree-LSTM from the paper to predict similarity for pairs of sentences on the SICK dataset, run `python main.py` to train and test the model, and have a look at `config.py` for command-line arguments.

The first run takes a few minutes because the GLOVE embeddings for the words in the SICK vocabulary will need to be read and stored to a cache for future runs. In later runs, only the cache is read in during later runs.

This code with `--lr 0.01 --wd 0.0001 --optim adagrad --batchsize 25` gives a Pearson's coefficient of `0.8336` and a MSE of `0.3119`, as opposed to a Pearson's coefficient of `0.8676` and a MSE of `0.2532` in the original paper. The difference might be because of differences in the way the word embeddings are updated.

### Acknowledgements
Shout-out to [Kai Sheng Tai](https://github.com/kaishengtai/) for the [original LuaTorch implementation](https://github.com/stanfordnlp/treelstm), and to the [Pytorch team](https://github.com/pytorch/pytorch#the-team) for the fun library.

### Author
[Riddhiman Dasgupta](https://researchweb.iiit.ac.in/~riddhiman.dasgupta/)

*This is my first PyTorch based implementation, and might contain bugs. Please let me know if you find any!*

### License
MIT