# Tree-Structured Long Short-Term Memory Networks
This is a [PyTorch](http://pytorch.org/) implementation of Tree-LSTM as described in the paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075) by Kai Sheng Tai, Richard Socher, and Christopher Manning. On the semantic similarity task using the SICK dataset, this implementation reaches a Pearson's coefficient of `0.8476` and a MSE of `0.2896`.

### Requirements
- Python (tested on **2.7.13** and **3.6.3**)
- [PyTorch](http://pytorch.org/) (tested on **0.1.12** and **0.2.0**)
- [tqdm](https://github.com/tqdm/tqdm)
- Java >= 8 (for Stanford CoreNLP utilities)

### Usage
 - First run the script `./fetch_and_preprocess.sh`, which, as the name suggests, does two things:
     - Fetch data, such as:
         - [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness task)
         - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!
         - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)
     - Preprocess data, i.e. generate dependency parses using [Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).
 - Run `python main.py` to try the Dependency Tree-LSTM from the paper to predict similarity for pairs of sentences on the SICK dataset. For a list of all command-line arguments, have a look at `config.py`.  
     - The first run takes a few minutes to read and store the GLOVE embeddings for the words in the SICK vocabulary to a cache for future runs. In later runs, only the cache is read in during later runs.
     - Logs and model checkpoints are saved to the `checkpoints/` directory with the name specified by the command line argument `--expname`.

### Results
Using hyperparameters `--lr 0.01 --wd 0.0001 --optim adagrad --batchsize 25` gives a Pearson's coefficient of `0.8476` and a MSE of `0.2896`, as opposed to a Pearson's coefficient of `0.8676` and a MSE of `0.2532` in the original paper. 
The difference might be because of the way the word embeddings are updated. In the paper, embeddings are updated using plain SGD, separate from the rest of the model, while here the same optimizer updates all the model parameters.

### Notes
 - (**Nov 08, 2017**) Refactored model to get **1.5x - 2x speedup**.
 - (**Oct 23, 2017**) Now works with **PyTorch 0.2.0**.
 - (**May 04, 2017**) Added support for **sparse tensors**. Using the `--sparse` argument will enable sparse gradient updates for `nn.Embedding`, potentially reducing memory usage. 
     - There are a couple of caveats, however, viz. weight decay will not work in conjunction with sparsity, and results from the original paper might not be reproduced using sparse embeddings.

### Acknowledgements
Shout-out to [Kai Sheng Tai](https://github.com/kaishengtai/) for the [original LuaTorch implementation](https://github.com/stanfordnlp/treelstm), and to the [Pytorch team](https://github.com/pytorch/pytorch#the-team) for the fun library.

### Contact
[Riddhiman Dasgupta](https://researchweb.iiit.ac.in/~riddhiman.dasgupta/)

*This is my first PyTorch based implementation, and might contain bugs. Please let me know if you find any!*

### License
MIT
