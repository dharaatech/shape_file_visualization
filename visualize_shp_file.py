import geopandas as gpd
import rasterio as rio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from rasterio.plot import show
import cv2
mpl.rcParams['agg.path.chunksize'] = 10000


global outpath
shape_file = "/home/abulala/vasundharaa/shape_files/compiled.shp"
# df.type
# df.bounds
# df.crs

def tif_to_jpg(inp_tif_img):
    global outpath
    dataset = rio.open(inp_tif_img)
    data = dataset.read([1,2,3]) # array with int val
    height=dataset.height
    width=dataset.width

    def normalize(x, lower, upper):
        """Normalize an array to a given bound interval"""

        x_max = np.max(x)
        x_min = np.min(x)

        m = (upper - lower) / (x_max - x_min)
        x_norm = (m * (x - x_min)) + lower

        return x_norm

    # Normalize each band separately
    data_norm = np.array([normalize(data[i,:,:], 0, 255) for i in range(data.shape[0])]) # array with float val
    data_rgb = data_norm.astype("uint8") # array with int val
    outpath = "/home/abulala/vasundharaa/Davis_jpg.jpg"
    data_rgb = data.astype("uint8")
    with rio.open(outpath, 'w',height=height,width=width,dtype='uint8',driver='JPEG',count=3) as outds:
        outds.write(data_rgb)

        return inp_tif_img


def plot_shp_file(shape_file_loc, jpg_img_path=None, _figsize=(5, 5), _edgeC="red", _facecolor=None):
    rows, columns = 2, 2
    df = gpd.read_file(shape_file_loc)
    f, axarr = plt.subplots(1, 2)
    df.plot(figsize=_facecolor, edgecolor=_edgeC, facecolor=_facecolor, ax=axarr[0])
    Davis_tif = rio.open("/home/abulala/vasundharaa/Davis_tif.tif")
    show(Davis_tif, ax=axarr[1])
    # axarr[1].imshow([[1, 2, 3, 4, 5, 6, 7]])


# tif_to_jpg("/home/abulala/vasundharaa/Davis_tif.tif")
plot_shp_file(shape_file)
plt.show()
