#!/usr/bin/env python
import sys
import os
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_TREX_MODULES = CUR_PATH + '/src'
sys.path.insert(0, PATH_TO_TREX_MODULES)

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


def parse_to_current_path(path):
    return os.path.join(CUR_PATH, path)


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
    log_config['PATH_TO_LOG'] = parse_to_current_path(log_config['PATH_TO_LOG'])
    log_config['PATH_TO_MODELS'] = parse_to_current_path(log_config['PATH_TO_MODELS'])
    model_config = config['model_config']
    agent_config = config['agent_config']

    if(args.debug):
        agent_config['epochs_to_train'] = 3
        agent_config['num_control_environments'] = 0
        log_config['save_model_every_epoch'] = 1
        log_config['keep_models'] = 3

    driver = ChromeDriver(display=args.display)
    memory = Memory(config=memory_config)
    game = TRexGame(config=game_config, chrome_driver=driver)
    preprocessor = Prepocessor(config=preprocessor_config)
    logger = Logger(config=log_config)
    restore_epoch = model_config['restore_epoch'] if 'restore_epoch' in model_config else None
    model = TFRexModel.restore_from_epoch(epoch=restore_epoch, config=model_config, logger=logger) if restore_epoch is not None else TFRexModel.create_network(config=model_config, logger=logger)
    agent = Agent(model=model, memory=memory, preprocessor=preprocessor,
            game=game, logger=logger, config=agent_config)  # noqa: E128
    agent.run()
#    agent.save_screenshots()
    agent.end()
