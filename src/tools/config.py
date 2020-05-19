import yaml
import os


def load_config(config_path):
    config = _load_config_yaml(config_path)

    src_root = config['src_root']
    package_config = config['packages']
    config['c3d_exe'] = os.path.join(src_root, package_config['c3d'])
    niftyreg_path = os.path.join(src_root, package_config['niftyreg'])
    config['niftyreg_resample'] = os.path.join(niftyreg_path, 'reg_resample')
    config['niftyreg_reg_measure'] = os.path.join(niftyreg_path, 'reg_measure')
    deedsBCV_path = os.path.join(src_root, package_config['deedsBCV'])
    config['deedsBCV_jacobian'] = os.path.join(deedsBCV_path, 'getJacobian')

    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
