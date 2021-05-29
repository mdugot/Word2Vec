# Word2Vec

Implementation in pytorch of Word2Vec neural network.
This implementation follow the Skip-gram negative sampling architecture describe in the paper [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf).
It has been trained on the Wikitext103 dataset, a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.

## Setup

The source are written in Python3.
The requirements can be installed with `pip3 install -r requirements.txt`.
Be sure to install the version 0.6.0 of torchtext for comnpatibility.

The following settings can be change in `config.json`:

 * **epoch** : number of training epoch
 * **negatives_sizes** : number of negative samples randomly draw at each training step
 * **window** : size of the windows for selecting nearby words
 * **min_freq** : minimum number of occurences in the corpus a word need to have to be kept for training
 * **log_cycle** : log tensorboard summaries every N steps
 * **save_cycle** : save the model state every N steps
 * **latent_space** : number of dimension in which a word will be embedded
 * **learning_rate** : learning rate during training
 * **learning_rate_decay** : learning rate decay at each epoch
 * **batch_size** : number of words trained in one batch
 * **restore** : model to restore before pursuing training (null to start training from scratch
 * **save_path** : where to save the model states
 * **log_path** : where to save the tensorboad logs
 * **device** : name of the device (cpu or cuda) to use for the training

## Training

Run `python3 train.py` to start to train a model.
The Wikitext103 dataset will be automatically downloaded the first time this script is run and save in the folder `.data` from where it will be loaded the next time the script is run.
One epoch, the time to pass throughout the dataset, will take tens of hours.
The model will be saved regularly based on the value of the setting `save_cycle`.
The file `trained.chkpt` contains a checkpoint of a trained model for the default config file.

## Encoder

Once trained, the class `Encoder` inside `encoder.py` can be used to wrap the model into an encoder.
The encoder can be initialized from a model checkpoint.

```
>>> from encoder import Encoder
>>> encoder = Encoder(checkpoint="trained.chkpt")
```

The encoder can be saved into a numpy pickle for a faster initialization therafter.
The file `codes.npy` already contains the encoder pickle for `trained.chkpt`.

```
>>> encoder.save("codes.npy")
>>> encoder = Encoder(pickle="codes.npy")
```

Use `encoder(<word>)` to get the numpy embedding of a word.
The number of dimension of the embedding is define by the setting `latent_space`.

```
>>> encoder("cat")
array([ 0.1479291 , -0.09920859, -0.35985562,  0.03662522, -0.31170434,
       -0.07307368,  0.05548168,  0.3211098 , -0.17774254, -0.18545032,
       -0.04707159, -0.04823684, -0.21294212,  0.37715054, -0.13487978,
       -0.04816947, -0.23025192,  0.14023067,  0.09613717,  0.09197191,
       -0.22896263, -0.04995414,  0.22161873, -0.22339153,  0.23970565,
        0.27494398, -0.32025686,  0.30171278,  0.37912276, -0.0723994 ,
        0.17689596, -0.18415405,  0.3185459 , -0.08690535,  0.10939758,
       -0.45774344, -0.12183642,  0.2105336 ,  0.10366029,  0.18736827,
        0.3161276 ,  0.15514486, -0.03422137, -0.01610632, -0.05822205,
        0.04097118, -0.09033743, -0.2506522 ,  0.10169964, -0.50079954,
        0.35568124,  0.24932037, -0.45425844, -0.009784  ,  0.06848705,
        0.11588038, -0.00651547, -0.23260884,  0.3548363 , -0.00171355,
       -0.2382073 , -0.29762125, -0.1823044 , -0.11441848, -0.39999095,
       -0.08809714, -0.13157777,  0.36048064,  0.41963845, -0.15511222,
       -0.29268256, -0.13951126, -0.15465052,  0.26883987, -0.26183766,
       -0.18846683,  0.16583268,  0.23982847, -0.13617523, -0.07869859,
        0.26444152, -0.04706819, -0.29681388,  0.2104976 ,  0.34585062,
       -0.43181938,  0.48085475,  0.09694713,  0.41979265, -0.19059187,
       -0.16374677, -0.03960251, -0.26500064, -0.02279791,  0.17458245,
       -0.18659289,  0.03939817, -0.2540078 , -0.00819635,  0.38976005],
      dtype=float32)
```

The method `similar_words` can be used to get words with a similar semantic according to the model.

```
>>> encoder.similar_words("clarinet")
['trumpet', 'cello', 'flute', 'oboe', 'continuo', 'soprano', 'flutes', 'oboes', 'violin', 'tuba']
>>> encoder.similar_words("consciousness")
['beings', 'deliberate', 'confusion', 'humanity', 'tension', 'notions', 'conscience', 'cruel', 'guilt', 'painful']
>>> encoder.similar_words("computer")
['software', 'graphics', 'electronic', 'apple', 'multiplayer', '3d', 'internet', 'mode', 'ds', 'virtual']
>>> encoder.similar_words("skin")
['flesh', 'ear', 'teeth', 'soft', 'ears', 'feathers', 'bones', 'eyes', 'darker', 'legs']
```

The method `plot` can be used to visualize the high dimensional embedding of a list of words with TSNE.

```
>>> encoder.plot("mutton", "rabbit", "mouse", "snake", "turtle", "lion", "cat", "dog", "horse", "bird", "train", "car", "helicopter", "plane", "truck", "boat", "rocket")
```

In this example, we can see two seperate clusters for the animals and the vehicule with `horse` being the closest animal of the vehicule cluster.

<p align="center">
	<img src="https://github.com/mdugot/Word2Vec/blob/master/w2v_plot.png" />
</p>
