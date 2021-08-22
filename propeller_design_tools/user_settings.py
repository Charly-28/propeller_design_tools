import sys
import os


def set_airfoil_database(path: str):
    _save_settings({'airfoil_database': path})
    return


def set_propeller_database(path: str):
    _save_settings({'propeller_database': path})
    return


def _get_env_dir():
    return os.path.split(os.path.split(sys.executable)[0])[0]


def _get_pdt_pkg_dir():
    return os.path.join(_get_env_dir(), 'Lib', 'site-packages', 'propeller_design_tools')


def _get_settings_fpath():
    pkg_dir = _get_pdt_pkg_dir()
    return os.path.join(pkg_dir, 'user-settings.txt')


def _save_settings(new_sett: dict = None, savepath: str = None):
    defaults = {
        'airfoil_database': None,
        'propeller_database': None
    }

    if new_sett is None:
        new_sett = {}

    if savepath is None:
        savepath = _get_settings_fpath()

    if os.path.exists(savepath):
        old_sett = _get_user_settings(settings_path=savepath)
    else:
        old_sett = {}

    with open(savepath, 'w') as f:
        for key in defaults:
            if key in new_sett:
                val = new_sett[key]
            elif key in old_sett:
                val = old_sett[key]
            else:
                val = defaults[key]
            f.write('{}: {}\n'.format(key, val))

    return


def _get_user_settings(settings_path: str = None) -> dict:
    if settings_path is None:
        settings_path = _get_settings_fpath()

    with open(settings_path, 'r') as f:
        txt = f.read().strip()

    lines = [ln for ln in txt.split('\n') if ln.strip() != '']
    settings = {}
    for line in lines:
        key, val = line.split(': ', 1)
        if val == 'None':
            val = None
        elif val == 'True':
            val = True
        elif val == 'False':
            val = False
        settings[key] = val

    return settings
