__all__ = ["radar_cmap", "radar_norm"]


precip_colors = [
    "#04e9e7",  # 0.01 - 0.10 inches
    "#019ff4",  # 0.10 - 0.25 inches
    "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#fdf802",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
    "#f800fd",  # 6.00 - 8.00 inches
    "#9854c6",  # 8.00 - 10.00 inches
    "#fdfdfd",
]  # 10.00+

radar_colors = [
    "#FFFFFF",  # 0
    "#808080",  # 0-5
    "#00FFFF",  # 5-10
    "#00BFFF",  # 10-15
    "#0000FF",  # 15-20
    "#00FF00",  # 20-25
    "#32CD32",  # 25-30
    "#008000",  # 30-35
    "#FFFF00",  # 35-40
    "#DAA520",  # 40-45
    "#FFA500",  # 45-50
    "#FF0000",  # 50-55
    "#8B0000",  # 55-60
    "#800000",  # 60-65
    "#FF00FF",  # 65-70
    "#8A2BE2",  # 70-75
    "#FFFFFF",
]  # 75+

radar_norms = [
    -32768,
    0,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    32767,
]


snow_colors = [
    "#FFFFFF",
    "#C2BFFF",
    "#867FFF",
    "#493FFF",
    "#0D00FF",
    "#0C3FBF",
    "#0B7F7F",
    "#0ABF3F",
    "#09FF00",
    "#46E800",
    "#84D200",
    "#C1BC00",
    "#FFA600",
    "#FF7C00",
    "#FF5300",
    "#FF2900",
    "#FF0000",
    "#FF004E",
    "#FF009C",
    "#FF00EA",
]

hail_colors = [
    "#FFFFFF",
    "#AEAAFF",
    "#5D55FF",
    "#0D00FF",
    "#0B55AA",
    "#0AAA55",
    "#09FF00",
    "#5BE100",
    "#ADC300",
    "#FFA600",
    "#FF6E00",
    "#FF3700",
    "#FF0000",
    "#FF0075",
    "#FF00EA",
]


class _MissingMatplotlibColor:
    def __init__(self, name):
        self.name = name

    def _raise(self):
        raise ImportError(f"{self.name} requires matplotlib. Install matplotlib or avoid using xmetai.utils.{self.name}.")

    def __call__(self, *args, **kwargs):
        self._raise()

    def __getattr__(self, name):
        self._raise()

    def __repr__(self):
        return f"<{self.name} unavailable: matplotlib is not installed>"


try:
    from matplotlib.colors import BoundaryNorm, ListedColormap
except ImportError:
    radar_cmap = _MissingMatplotlibColor("radar_cmap")
    radar_norm = _MissingMatplotlibColor("radar_norm")
else:
    radar_cmap = ListedColormap(radar_colors)
    radar_norm = BoundaryNorm(radar_norms, ncolors=len(radar_colors), clip=True)
