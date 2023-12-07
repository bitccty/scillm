import yaml


def load_config(args):
    '''the configuration of each model can rewrite the base configuration'''
    # base configuration
    base_configuration = load_base_config()
    
    # load one model configuration
    configuration = load_model_config(args["model"], args["mode"])
    
    # update and add special config for base config
    base_configuration.update(configuration)
    configuration = base_configuration
    return configuration


def load_base_config():
    config_path = f'config/base.yaml'
    with open(config_path, 'r') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print(f'[!] load base configuration {config_path}')
    return configuration


def load_model_config(model, mode):
    # load special config for each model
    config_path = f'config/{model}.yaml'
    with open(config_path, 'r') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
        new_config = {}
        for key, value in configuration.items():
            if key in ['train', 'test', 'validation']:
                if key == mode:
                    new_config.update(value)
            else:
                new_config[key] = value
        configuration = new_config
    return configuration