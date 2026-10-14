# Radio Map Format

WWM-NAV reads radio maps from NumPy `.npz` files. The first three positional arrays are required:

| Field | Shape | Unit | Meaning |
|---|---:|---|---|
| `arr_0` | `(nx, ny)` | probability | Outage probability in `[0, 1]`. Higher is worse. |
| `arr_1` | `(nx,)` | map units | x-axis coordinates. The loader multiplies by `coordinate_scale` and defaults to meters via `10.0`. |
| `arr_2` | `(ny,)` | map units | y-axis coordinates. The loader multiplies by `coordinate_scale` and defaults to meters via `10.0`. |

The default training path requires physical context fields because the RSSM is conditioned on nearby base-station and building features:

| Field | Shape | Unit | Meaning |
|---|---:|---|---|
| `bs_loc_km` | `(num_bs, 3)` | km | Base-station x, y, height. |
| `sector_boresite_deg` | `(num_sector,)` | degree | Sector boresight angles. |
| `bs_power_watt` | scalar/`(1,)` | watt | Shared BS transmit power fallback. |
| `height_map` | `(hx, hy)` | m | Building/terrain height raster. |
| `best_bs` | `(nx, ny)` | index | Best base-station index for each grid cell. |
| `best_sector` | `(nx, ny)` | index | Best sector index for each grid cell. |

Optional fields supported by the loader include `bs_power_watt_by_bs`, `active_original_bs_ids`, `uav_height_km`, `max_height_m`, `bs_downtilt_deg`, `carrier_freq_ghz`, and `los_samples`.

A tiny synthetic example can be generated with:

```bash
python scripts/generate_sample_map.py --output data/sample_radiomap.npz
```

This sample is for smoke tests and tutorials, not for reporting research conclusions.
