#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive visualization tool for pred_*.nc inference outputs.

Step 1: scan directory for pred files, init-date range, variables.
Step 2: user selects variable and time range.
Step 3: render cartopy maps and save as GIF.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from PIL import Image  # noqa: E402

# Help cartopy/pyproj locate projection database on Windows conda envs.
if "PROJ_DATA" not in os.environ:
    try:
        import pyproj

        os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
    except Exception:
        pass

PRED_PATTERN = re.compile(r"^pred_(\d{8})\.nc$", re.IGNORECASE)

VAR_META = {
    "z500": ("500hPa Geopotential Height", "gpm"),
    "u200": ("200hPa U Wind", "m/s"),
    "u850": ("850hPa U Wind", "m/s"),
    "msl": ("Mean Sea Level Pressure", "Pa"),
    "t2m": ("2m Temperature", "K"),
    "sst": ("Sea Surface Temperature", "K"),
    "ttr": ("Top Net Thermal Radiation", "J/m^2"),
    "tp": ("Total Precipitation", "m"),
    "ot50": ("Ocean Temperature 50m", "K"),
    "ssh": ("Sea Surface Height", "m"),
}

CMAP_BY_VAR = {
    "sst": "RdYlBu_r",
    "t2m": "RdYlBu_r",
    "ot50": "RdYlBu_r",
    "msl": "viridis",
    "z500": "viridis",
    "u200": "RdBu_r",
    "u850": "RdBu_r",
    "ttr": "magma",
    "tp": "Blues",
    "ssh": "PuOr",
}


@dataclass
class PredInventory:
    directory: Path
    files: list[Path]
    init_dates: list[datetime]
    variables: list[str]
    n_leads: int
    lead_hours: int
    sample_valid_start: datetime
    sample_valid_end: datetime

    @property
    def init_start(self) -> datetime:
        return self.init_dates[0]

    @property
    def init_end(self) -> datetime:
        return self.init_dates[-1]


def parse_date(text: str) -> datetime:
    text = text.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {text}")


def scan_pred_directory(directory: str | Path) -> PredInventory:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")

    files = sorted(p for p in directory.iterdir() if p.is_file() and PRED_PATTERN.match(p.name))
    if not files:
        raise FileNotFoundError(f"目录下未找到 pred_YYYYMMDD.nc 文件: {directory}")

    init_dates = [parse_date(PRED_PATTERN.match(p.name).group(1)) for p in files]

    with xr.open_dataset(files[0]) as ds:
        if "level" not in ds.coords:
            raise ValueError(f"文件缺少 level 坐标: {files[0]}")
        variables = [str(v) for v in ds.level.values]
        n_leads = int(ds.sizes["time"])
        times = ds.time.values
        sample_valid_start = np.datetime64(times[0], "ns").astype("datetime64[s]").astype(datetime)
        sample_valid_end = np.datetime64(times[-1], "ns").astype("datetime64[s]").astype(datetime)
        if n_leads > 1:
            delta = (np.datetime64(times[1], "ns") - np.datetime64(times[0], "ns")) / np.timedelta64(1, "h")
            lead_hours = int(delta)
        else:
            lead_hours = 24

    return PredInventory(
        directory=directory,
        files=files,
        init_dates=init_dates,
        variables=variables,
        n_leads=n_leads,
        lead_hours=lead_hours,
        sample_valid_start=sample_valid_start,
        sample_valid_end=sample_valid_end,
    )


def print_inventory(inv: PredInventory) -> None:
    print("\n========== 预测文件扫描结果 ==========")
    print(f"目录: {inv.directory}")
    print(f"pred 文件数: {len(inv.files)}")
    print(f"起报日范围: {inv.init_start:%Y-%m-%d} ~ {inv.init_end:%Y-%m-%d}")
    print(f"每个文件预报步数: {inv.n_leads} (步长 {inv.lead_hours}h)")
    print(f"示例有效时间: {inv.sample_valid_start:%Y-%m-%d} ~ {inv.sample_valid_end:%Y-%m-%d}")
    print("可用变量 (level):")
    for i, name in enumerate(inv.variables):
        label, unit = VAR_META.get(name, (name, ""))
        unit_text = f" [{unit}]" if unit else ""
        print(f"  [{i:2d}] {name:6s} - {label}{unit_text}")
    print("====================================\n")


def prompt_default(prompt: str, default: str) -> str:
    text = input(f"{prompt} [{default}]: ").strip()
    return text or default


def choose_variable(inv: PredInventory, var_text: str | None) -> str:
    if var_text:
        var = var_text.strip()
        if var.isdigit():
            idx = int(var)
            if idx < 0 or idx >= len(inv.variables):
                raise ValueError(f"变量索引越界: {idx}")
            return inv.variables[idx]
        if var not in inv.variables:
            raise ValueError(f"未知变量: {var}")
        return var

    while True:
        text = input("请输入变量名或序号 (如 sst 或 5): ").strip()
        try:
            return choose_variable(inv, text)
        except ValueError as exc:
            print(exc)


def choose_mode(mode_text: str | None) -> str:
    modes = {
        "1": "lead",
        "2": "init",
        "3": "valid",
        "lead": "lead",
        "init": "init",
        "valid": "valid",
    }
    if mode_text:
        key = mode_text.strip().lower()
        if key not in modes:
            raise ValueError(f"未知模式: {mode_text}")
        return modes[key]

    print("动画模式:")
    print("  1) lead  - 固定起报日，按预报时效播放")
    print("  2) init  - 固定预报时效，按起报日播放")
    print("  3) valid - 按有效时间范围播放 (多起报时取最短时效)")
    while True:
        text = input("请选择模式 [1/2/3，默认 1]: ").strip() or "1"
        if text in modes:
            return modes[text]
        print("无效模式，请重新输入。")


def find_file_by_init(inv: PredInventory, init_date: datetime) -> Path:
    name = f"pred_{init_date:%Y%m%d}.nc"
    path = inv.directory / name
    if not path.exists():
        raise FileNotFoundError(f"未找到起报日文件: {name}")
    return path


def load_var_slice(path: Path, variable: str, time_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    with xr.open_dataset(path) as ds:
        da = ds["data"].sel(level=variable)
        field = da.isel(time=time_idx).values
        lat = ds.lat.values
        lon = ds.lon.values
        valid_time = np.datetime64(da.time.values[time_idx], "ns").astype("datetime64[s]").astype(datetime)
    return field, lat, lon, valid_time


def collect_lead_frames(inv: PredInventory, variable: str, init_date: datetime, lead_start: int, lead_end: int):
    path = find_file_by_init(inv, init_date)
    lead_start = max(1, lead_start)
    lead_end = min(inv.n_leads, lead_end)
    frames = []
    for lead in range(lead_start, lead_end + 1):
        field, lat, lon, valid_time = load_var_slice(path, variable, lead - 1)
        frames.append(
            {
                "title": f"Init {init_date:%Y-%m-%d} | Lead {lead}d | Valid {valid_time:%Y-%m-%d}",
                "field": field,
                "lat": lat,
                "lon": lon,
            }
        )
    return frames


def collect_init_frames(inv: PredInventory, variable: str, init_start: datetime, init_end: datetime, lead_day: int):
    lead_day = max(1, min(inv.n_leads, lead_day))
    frames = []
    for path, init_date in zip(inv.files, inv.init_dates):
        if init_date < init_start or init_date > init_end:
            continue
        field, lat, lon, valid_time = load_var_slice(path, variable, lead_day - 1)
        frames.append(
            {
                "title": f"Init {init_date:%Y-%m-%d} | Lead {lead_day}d | Valid {valid_time:%Y-%m-%d}",
                "field": field,
                "lat": lat,
                "lon": lon,
            }
        )
    return frames


def collect_valid_frames(inv: PredInventory, variable: str, valid_start: datetime, valid_end: datetime):
    frames = []
    cursor = pd.Timestamp(valid_start).normalize()
    end = pd.Timestamp(valid_end).normalize()
    while cursor <= end:
        best = None
        for path, init_date in zip(inv.files, inv.init_dates):
            with xr.open_dataset(path) as ds:
                times = pd.to_datetime(ds.time.values).normalize()
                hit = np.where(times == cursor)[0]
                if hit.size == 0:
                    continue
                lead_idx = int(hit[0])
                lead_day = lead_idx + 1
                field = ds["data"].sel(level=variable).isel(time=lead_idx).values
                lat = ds.lat.values
                lon = ds.lon.values
                candidate = (lead_day, init_date, field, lat, lon)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is not None:
            lead_day, init_date, field, lat, lon = best
            frames.append(
                {
                    "title": f"Valid {cursor:%Y-%m-%d} | Init {init_date:%Y-%m-%d} | Lead {lead_day}d",
                    "field": field,
                    "lat": lat,
                    "lon": lon,
                }
            )
        cursor += pd.Timedelta(days=1)
    return frames


def prepare_lon_lat_grid(lon: np.ndarray, lat: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    field = np.asarray(field, dtype=float)

    # Data saved as 0-360; cartopy global view uses -180-180.
    if lon.max() > 180:
        if lon.size > 1 and np.isclose(lon[-1], 360.0):
            lon = lon[:-1]
            field = field[..., :-1]
        lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon)
        lon = lon[order]
        field = field[..., order]
    return lon, lat, field


def compute_vlim(frames: list[dict], variable: str, vmin: float | None, vmax: float | None):
    if vmin is not None and vmax is not None:
        return vmin, vmax
    values = np.concatenate([np.ravel(f["field"][np.isfinite(f["field"])]) for f in frames])
    if values.size == 0:
        return 0.0, 1.0
    if vmin is None and vmax is None and variable in {"u200", "u850"}:
        limit = np.nanpercentile(np.abs(values), 98)
        return -limit, limit
    low = vmin if vmin is not None else np.nanpercentile(values, 2)
    high = vmax if vmax is not None else np.nanpercentile(values, 98)
    if low == high:
        high = low + 1e-6
    return float(low), float(high)


def render_frame(
    frame: dict,
    variable: str,
    vmin: float,
    vmax: float,
    extent: tuple[float, float, float, float] | None,
    figsize: tuple[float, float],
    dpi: int,
) -> Image.Image:
    label, unit = VAR_META.get(variable, (variable, ""))
    cmap = CMAP_BY_VAR.get(variable, "viridis")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#d7e9f7", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f2efe9", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.4, linestyle=":", zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    lon, lat, field = prepare_lon_lat_grid(frame["lon"], frame["lat"], frame["field"])
    mesh = ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    unit_text = f" ({unit})" if unit else ""
    ax.set_title(f"{label}{unit_text}\n{frame['title']}", fontsize=11, pad=8)
    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="vertical",
        fraction=0.018,
        pad=0.02,
        shrink=0.78,
        aspect=35,
    )
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label(unit, fontsize=8, labelpad=4) if unit else None

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def save_gif(frames: list[dict], variable: str, output_path: Path, duration_ms: int, vmin, vmax, extent, figsize, dpi):
    if not frames:
        raise ValueError("没有可绘制的帧，请检查时间范围。")

    vmin, vmax = compute_vlim(frames, variable, vmin, vmax)
    images = [render_frame(frame, variable, vmin, vmax, extent, figsize=figsize, dpi=dpi) for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"GIF 已保存: {output_path} ({len(images)} 帧)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pred_*.nc 可视化并导出 GIF")
    parser.add_argument("--dir", default=r"D:\groupvaeinfer", help="pred 文件目录")
    parser.add_argument("--var", default=None, help="变量名或序号")
    parser.add_argument(
        "--mode",
        choices=["lead", "init", "valid"],
        default=None,
        help="lead=固定起报按时效; init=固定时效按起报; valid=按有效时间",
    )
    parser.add_argument("--init", dest="init_date", default=None, help="起报日 YYYYMMDD (lead 模式)")
    parser.add_argument("--init-start", default=None, help="起报开始日 (init 模式)")
    parser.add_argument("--init-end", default=None, help="起报结束日 (init 模式)")
    parser.add_argument("--lead", type=int, default=None, help="预报时效天数 (init 模式)")
    parser.add_argument("--lead-start", type=int, default=1, help="起始时效 (lead 模式)")
    parser.add_argument("--lead-end", type=int, default=None, help="结束时效 (lead 模式)")
    parser.add_argument("--valid-start", default=None, help="有效时间开始 YYYYMMDD (valid 模式)")
    parser.add_argument("--valid-end", default=None, help="有效时间结束 YYYYMMDD (valid 模式)")
    parser.add_argument("--output", default=None, help="输出 GIF 路径")
    parser.add_argument("--duration", type=int, default=500, help="每帧时长(ms)")
    parser.add_argument("--dpi", type=int, default=120, help="图像 DPI")
    parser.add_argument("--vmin", type=float, default=None, help="色标下限")
    parser.add_argument("--vmax", type=float, default=None, help="色标上限")
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=None,
        help="可选区域裁剪(默认全球)。例如中国区: 70 140 15 55",
    )
    parser.add_argument("--scan-only", action="store_true", help="仅扫描目录后退出")
    parser.add_argument("--non-interactive", action="store_true", help="参数不足时报错，不进入交互")
    return parser


def run_interactive(inv: PredInventory, args: argparse.Namespace):
    variable = choose_variable(inv, args.var)
    mode = choose_mode(args.mode)

    if mode == "lead":
        init_default = inv.init_dates[0].strftime("%Y%m%d")
        init_text = args.init_date or prompt_default("起报日 YYYYMMDD", init_default)
        init_date = parse_date(init_text)
        lead_start = args.lead_start
        lead_end = args.lead_end
        if args.lead_end is None:
            lead_end = int(prompt_default("结束时效(天)", str(inv.n_leads)))
        frames = collect_lead_frames(inv, variable, init_date, lead_start, lead_end)
        default_name = f"{variable}_lead_{init_date:%Y%m%d}_{lead_start}-{lead_end}.gif"
    elif mode == "init":
        lead_day = args.lead
        if lead_day is None:
            lead_day = int(prompt_default("固定预报时效(天)", "7"))
        init_start = parse_date(args.init_start or prompt_default("起报开始日", inv.init_start.strftime("%Y%m%d")))
        init_end = parse_date(args.init_end or prompt_default("起报结束日", inv.init_end.strftime("%Y%m%d")))
        frames = collect_init_frames(inv, variable, init_start, init_end, lead_day)
        default_name = f"{variable}_init_lead{lead_day}_{init_start:%Y%m%d}-{init_end:%Y%m%d}.gif"
    else:
        valid_start = parse_date(args.valid_start or prompt_default("有效时间开始", inv.sample_valid_start.strftime("%Y%m%d")))
        valid_end = parse_date(args.valid_end or prompt_default("有效时间结束", inv.sample_valid_end.strftime("%Y%m%d")))
        frames = collect_valid_frames(inv, variable, valid_start, valid_end)
        default_name = f"{variable}_valid_{valid_start:%Y%m%d}-{valid_end:%Y%m%d}.gif"

    output = args.output or prompt_default("输出 GIF 路径", str(inv.directory / default_name))
    duration = args.duration
    if not args.non_interactive:
        duration_text = input(f"每帧时长(ms) [{duration}]: ").strip()
        if duration_text:
            duration = int(duration_text)

    save_gif(
        frames,
        variable,
        Path(output),
        duration_ms=duration,
        vmin=args.vmin,
        vmax=args.vmax,
        extent=tuple(args.extent) if args.extent else None,
        figsize=(12, 6),
        dpi=args.dpi,
    )


def run_from_args(inv: PredInventory, args: argparse.Namespace):
    if args.var is None or args.mode is None:
        raise SystemExit("非交互模式需要 --var 和 --mode")

    variable = choose_variable(inv, args.var)
    if args.mode == "lead":
        if args.init_date is None:
            raise SystemExit("lead 模式需要 --init")
        init_date = parse_date(args.init_date)
        lead_end = args.lead_end if args.lead_end is not None else inv.n_leads
        frames = collect_lead_frames(inv, variable, init_date, args.lead_start, lead_end)
        default_name = f"{variable}_lead_{init_date:%Y%m%d}_{args.lead_start}-{lead_end}.gif"
    elif args.mode == "init":
        if args.lead is None:
            raise SystemExit("init 模式需要 --lead")
        init_start = parse_date(args.init_start or inv.init_start.strftime("%Y%m%d"))
        init_end = parse_date(args.init_end or inv.init_end.strftime("%Y%m%d"))
        frames = collect_init_frames(inv, variable, init_start, init_end, args.lead)
        default_name = f"{variable}_init_lead{args.lead}_{init_start:%Y%m%d}-{init_end:%Y%m%d}.gif"
    else:
        valid_start = parse_date(args.valid_start or inv.sample_valid_start.strftime("%Y%m%d"))
        valid_end = parse_date(args.valid_end or inv.sample_valid_end.strftime("%Y%m%d"))
        frames = collect_valid_frames(inv, variable, valid_start, valid_end)
        default_name = f"{variable}_valid_{valid_start:%Y%m%d}-{valid_end:%Y%m%d}.gif"

    output = Path(args.output or inv.directory / default_name)
    save_gif(
        frames,
        variable,
        output,
        duration_ms=args.duration,
        vmin=args.vmin,
        vmax=args.vmax,
        extent=tuple(args.extent) if args.extent else None,
        figsize=(12, 6),
        dpi=args.dpi,
    )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        inv = scan_pred_directory(args.dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)

    print_inventory(inv)
    if args.scan_only:
        return

    if args.non_interactive or (args.var and args.mode):
        run_from_args(inv, args)
    else:
        run_interactive(inv, args)


if __name__ == "__main__":
    main()
