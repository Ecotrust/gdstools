import os

import seaborn as sns
import numpy as np
from rasterio import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


# Suppress errors
# Get cut-down GDAL that rasterio uses
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')


def image_collection(path, file_pattern='*.tif'):
    import glob
    return glob.glob(f'{path}/**/{file_pattern}', recursive=True)


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, y):
        x = y.new(*y.size())
        for i in range(x.shape[0]):
            x[i, :, :] = y[i, :, :] * self.std[i] + self.mean[i]
        return x


def get_colors(n=10, palette="Spectral_r", output="list"):
    """Return a list of colors/cmap from a seaborn palette.

    Parameters
    ----------
    n : int, optional
        Number of colors to return, by default 10
    palette : str, optional
        Name of seaborn palette, by default "Spectral_r"
    output : str, optional
        Output type. "list" returns a list of colors. "cmap" returns 
        a matplotlib ListedColormap. Defaults to "list."

    Returns
    -------
    list or matplotlib.colors.ListedColormap
    """
    from matplotlib.colors import ListedColormap

    colors = list(reversed(sns.color_palette(palette, n).as_hex()))

    if output == "list":
        return colors
    else:
        return ListedColormap(colors)


def html_to_rgb(html_color):
    # Remove the '#' character from the beginning of the HTML color string
    html_color = html_color.lstrip('#')

    # Convert the HTML color string to a tuple of integers
    rgb_tuple = tuple(int(html_color[i:i+2], 16) for i in (0, 2, 4))

    return rgb_tuple


def save_cog(
    data: np.ndarray,
    profile: dict,
    path,
    colordict=None,
    categories=None,
    overwrite=False,
):
    cog_profile = cog_profiles.get("deflate")
    if not os.path.exists(path) or overwrite:
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data)

                if colordict is not None:
                    dst.write_colormap(1, colordict)
                if categories is not None:
                    dst.update_tags(CATEGORY_NAMES=categories)

                cog_translate(dst, path, cog_profile, in_memory=True, quiet=False)
    else:
        print(f"File {path} already exists")

    return

# %%
# TODO: Move code below to standalone script
# TODO: implement tests
# if __name__ == "__main__":
    # %%
    # collection_name = '3dep'
    # year = None

    # repo = Path(os.path.expanduser('~/mapping_forest_types/data/prod/train'))
    # collection_path = repo / f'{collection_name}'
    # if year:
    #     collection_path = collection_path / str(year)
    # collection = image_collection(collection_path)

    # stats = get_collection_stats(collection, BANDS[collection_name])
