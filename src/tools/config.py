import yaml
import os


def load_config(config_path):
    config = _load_config_yaml(config_path)

    src_root = config['src_root']
    package_config = config['packages']
    config['c3d_exe'] = os.path.join(src_root, package_config['c3d'])
    config['niftyreg'] = os.path.join(src_root, package_config['niftyreg'])
    config['niftyreg_resample'] = os.path.join(config['niftyreg'], 'reg_resample')

    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
