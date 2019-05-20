Applying Reinforcement learning for the TRex game provided by chromium 

Installation:
- chrome
- chromedriver (sudo apt-get install chromium-chromedriver &&
sudo ln -s /usr/lib/chromium-browser/chromedriver /usr/bin/chromedriver)
- Tkinter (apt-get install python-tk, for pyenv users check [this](https://stackoverflow.com/questions/22550068/python-not-configured-for-tk/31299142#31299142))
- pip install -r requirements.txt

12/2019:
Using the DQN - Network proposed in Deep Mind's Atari Paper together with Dueling 
Network Architectures and Prioritized Experience Replay, 
we manage to average a score of ~800 (see python/plots/score_plot_trial19_mem_50k_lr_0.0001_decay_5000.png).
