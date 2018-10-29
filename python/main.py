#!/usr/bin/env python
import sys
import os
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TREX_MODULES = CUR_PATH + '/src'
sys.path.insert(0, PATH_TO_TREX_MODULES)

PATH_TO_MODELS = os.path.join(CUR_PATH, './models')
PATH_TO_LOG = os.path.join(CUR_PATH, './log')

from tRexModel import TFRexModel  # noqa: E402
from tRexGame import TRexGame  # noqa: E402
from tRexMemory import Memory  # noqa: E402
from tRexPreprocessor import Prepocessor  # noqa: E402
from tRexLogger import Logger  # noqa: E402
from tRexAgent import Agent  # noqa: E402
from tRexDriver import ChromeDriver  # noqa: E402
from configobj import ConfigObj  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from tRexUtils import convert_config_to_correct_type  # noqa: E402
import ipdb  # noqa: E402, F401


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default='training.config')
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    config = ConfigObj(args.config).dict()
    config = convert_config_to_correct_type(config)
    memory_config = config['memory_config']
    game_config = config['game_config']
    preprocessor_config = config['preprocessor_config']
    log_config = config['log_config']
    log_config['PATH_TO_LOG'] = PATH_TO_LOG
    log_config['PATH_TO_MODELS'] = PATH_TO_MODELS
    model_config = config['model_config']
    restore_epoch = model_config['restore_epoch'] if 'restore_epoch' in model_config else None
    agent_config = config['agent_config']
    mode = agent_config['mode']

    if(args.debug):
        agent_config['epochs_to_train'] = 2
        agent_config['num_control_environments'] = 0
        mode = 'train'

    driver = ChromeDriver(display=args.display)
    memory = Memory(config=memory_config)
    game = TRexGame(config=game_config, chrome_driver=driver)
    preprocessor = Prepocessor(config=preprocessor_config)
    logger = Logger(config=log_config)

    if(restore_epoch is not None or mode == 'play'):
        assert restore_epoch is not None, 'if mode is "play", a the network parameters have to be restore'
        model = TFRexModel.restore_from_epoch(epoch=restore_epoch, config=model_config, logger=logger)
    else:
        model = TFRexModel.create_network(config=model_config, logger=logger)

    agent = Agent(model=model, memory=memory, preprocessor=preprocessor,
            game=game, logger=logger, config=agent_config)  # noqa: E128
