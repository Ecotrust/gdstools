# %%
import os
from pathlib import Path
import sys
import contextlib
import configparser
import yaml
import re

import numpy as np


# %%
class AttrDict(dict):
    """Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)

    Stolen from: https://stackoverflow.com/a/48806603/1913361
    """

    def __init__(self, mapping=None):
        super(AttrDict, self).__init__()

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__


class ConfigLoader:
    def __init__(self, root, config_file="config.yaml"):
        self.config = AttrDict({})
        self.root = root
        self.config_file = config_file

    def load_from_env(self):
        from dotenv import dotenv_values
        env = dotenv_values(os.path.join(self.root, ".env"))
        for key in env:
            self.config[key] = env[key]

    def load_from_file(self):
        with open(os.path.join(self.root, self.config_file), "r") as f:
            config = yaml.safe_load(f)
            self.config.update(config)

    def load(self):
        self.load_from_env()
        self.load_from_file()
        return self.config

    def get(self, key):
        return self.config.get(key, None)


def search(pattern, string, *args):
    """Return True if pattern is found in string."""
    try:
        if re.search(pattern, string, *args):
            return True
        else:
            return False

    except TypeError:
        return False


def find(pattern, values, ignore_case=True):
    """Search pattern in list.

    Arguments:
        pattern {str, list} -- [description]
        values {list/array} -- List/array of strings.

    Returns:
        [numpy array] -- array
    """
    arg = None
    if ignore_case:
        arg = re.IGNORECASE

    if isinstance(pattern, list):
        array = []
        for p in pattern:
            array.append(list(map(lambda x: search(p, x, arg), values)))
            # np.array(array)
        return np.max(array, 0)

    else:
        return np.array(list(map(lambda x: search(pattern, x, arg), values)))


def degrees_to_meters(deg, angle="lon"):
    """Convert degrees to meters.

    From https://earthscience.stackexchange.com/questions/7350/
    converting-grid-resolution-from-degrees-to-kilometers

    Parameters
    ----------
    deg : float
        Degrees to convert.
    plane : str, optional
        Indicates whether deg are degrees latitude or degrees longitude.
        The default is 'lon'.

    Returns
    -------
    float : Distance in meters.
    """
    import math

    R = 6378137.0  # Earth radius in meters
    rad = math.radians(deg)

    if angle == "lon":
        return R * rad * math.cos(rad)

    elif angle == "lat":
        return R * rad

# See also this utm package https://github.com/Turbo87/utm 
def infer_utm(bbox):
    """
    Infer the UTM Coordinate Reference System (CRS) by determining the UTM zone where a given lat/long bounding box is located.
    Modified from: https://stackoverflow.com/a/40140326/1913361

    :param bbox: A list-like object containing the bounding box coordinates (minx, miny, maxx, maxy).
    :type bbox: list-like
    :param zone_number: An optional integer specifying the UTM zone number. If not provided, the UTM zone will be inferred based on the bounding box coordinates.
    :type zone_number: int
    :return: pyproj.CRS object representing the UTM CRS.
    :rtype: pyproj.CRS
    """
    from pyproj import CRS

    xmin, _, xmax, _ = bbox
    midpoint = xmin + (xmax - xmin) / 2

    epsg_list = []
    for zone in range(1, 61):
        if (midpoint > -180 + (zone - 1) * 6) & (midpoint <= -180 + zone * 6):
            epsg = 32600 + zone
            epsg_list.append(CRS.from_epsg(epsg))

    return epsg_list.pop(0)


def split_bbox(dim, bbox_to_split):
    """Split a bounding box into dim x dim bounding boxes.

    Parameters
    ----------
    dim : int
        Number of splits per dimension of bbox with shape (dim, dim). The number
        of parts (n) to split the original bounding box will be dim x dim.
    bbox_to_split : list-like
        The bounding box to split, with format [xmin, ymin, xmax, ymax].

    Returns
    -------
    np.array
        An array of n bounding boxes.
    """
    xmin, ymin, xmax, ymax = bbox_to_split

    w = (xmax - xmin) / dim
    h = (ymax - ymin) / dim

    # For testing
    # cols = ['xmin', *[f'xmin + w*{dim + 1}' for dim in range(dim - 1)], 'xmax']
    # rows = ['ymin', *[f'ymin + l*{dim + 1}' for dim in range(dim - 1)], 'ymax']

    cols = [xmin, *[xmin + w * (dim + 1) for dim in range(dim - 1)], xmax]
    rows = [ymin, *[ymin + h * (dim + 1) for dim in range(dim - 1)], ymax]

    coords = np.array(np.meshgrid(cols, rows)).T

    bbox_splitted = []
    for i in range(dim):
        bbox_splitted.append(
            [
                np.array([coords[i][j], coords[i + 1][k]]).flatten()
                for j, k in zip(range(dim), range(1, dim + 1))
            ]
        )

    return np.array([x for sbl in bbox_splitted for x in sbl])


def create_directory_tree(*args):
    """Set the directory tree for the dataset.

    Parameters
    ----------
    args : str
        The directory names to be created.

    Returns
    -------
        PosixPath object
    """
    filepath = Path(*args)

    if not filepath.exists():
        os.makedirs(filepath)

    return filepath


def print_message(msg, progressbar=None):
    """Print message."""
    if progressbar:
        progressbar.write(msg)
    else:
        print(msg)


# from https://stackoverflow.com/a/37243211/1913361
class StdOutFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            from tqdm import tqdm

            tqdm.write(x, file=self.file)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = StdOutFile(sys.stdout)
    yield
    sys.stdout = save_stdout


# %%
def multithreaded_execution(function, parameters, threads=20):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import inspect
    import tqdm

    # Get parameter list
    fun_params = inspect.signature(function).parameters.keys()
    fun_progressbar_param = True if "progressbar" in fun_params else False

    n_items = len(parameters)
    assert n_items > 0, "Empty list of parameters passed."
    print("\n", "Processing {:,d} images".format(n_items))

    with tqdm.tqdm(
        total=n_items, bar_format="{l_bar}{bar:75}{r_bar}{bar:-50b}", file=sys.stdout
    ) as pbar:
        if fun_progressbar_param:
            _ = [p.update({"progressbar": pbar}) for p in parameters]

        with nostdout():
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [executor.submit(function, **param) for param in parameters]
                results = []

                try:  # catch exceptions
                    for future in as_completed(futures):
                        results.append(future.result())
                        pbar.update(1)

                except Exception as e:
                    print(f'Exception "{e}" raised while processing files.')
                    raise e

        return np.array(results)
