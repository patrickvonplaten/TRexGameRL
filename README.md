This repo contains a private project applying Reinforcement learning for the TRex game provided by chromium 

Installation:
- chrome
- chromedriver (sudo apt-get install chromium-chromedriver &&
sudo ln -s /usr/lib/chromium-browser/chromedriver /usr/bin/chromedriver)
- Tkinter (apt-get install python-tk, for pyenv users check [this](https://stackoverflow.com/questions/22550068/python-not-configured-for-tk/31299142#31299142))
- pip install -r requirements.txt

The motivation is to master the TRex game purely based on visual input in form of screenshots of the game. This way, the model 
has access to the same and only the same information a human play would have for the Game.

The DQN model architecture is taken from [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

Iteratively, the model was improved by implementing the [duel networks architecture](http://proceedings.mlr.press/v48/wangf16.pdf), [prioritized experience replay](https://arxiv.org/pdf/1511.05952.pdf) and [double q-learning](https://arxiv.org/pdf/1509.06461.pdf)


An average score of ~800 (see python/plots/score_plot_trial19_mem_50k_lr_0.0001_decay_5000.png) is achieved by using a large 
memory of 50.000 and Adam optimization. 

Using a smaller memory of only 12.000 a score of could be achieved.
