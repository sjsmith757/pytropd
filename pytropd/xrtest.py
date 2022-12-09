import logging
import numpy as np
import xarray as xr
from pandas import date_range
import pytropd as pyt
from pathlib import Path

# Validate metrics
# Check calculations with precalculated values from file within roundoff error
# Psi500
# read meridional velocity V(time,lat,lev), latitude and level
root = Path(__file__).absolute().parent.parent
data_dir = root / "ValidationData"
metrics_dir = root / "ValidationMetrics"
logging.basicConfig(level=logging.INFO)


def get_validated_metric(metric: str) -> xr.DataArray:
    """
    for the given metric, open the validation file and reshape and label to match the
    data format output by the xarray methods

    Parameters
    ----------
    metric : str
        the metric file name to open

    Returns
    -------
    xr.DataArray
        the validated metric data
    """
    validated_metric = xr.open_dataset(metrics_dir / f"{metric}.nc")
    validated_metric = validated_metric.to_array(dim="hemsph")
    monthly = False
    if "DJF" in metric:
        start_date = "1978-12-01"
    elif "MAM" in metric:
        start_date = "1979-03-01"
    elif "JJA" in metric:
        start_date = "1979-06-01"
    elif "SON" in metric:
        start_date = "1979-09-01"
    else:
        start_date = "1979-01-01"
        monthly = "ANN" not in metric
    return validated_metric.assign_coords(
        hemsph=[h[-2:].upper() for h in validated_metric.hemsph.values],
        time=date_range(
            start=start_date,
            periods=validated_metric.time.size,
            freq="12MS" if not monthly else "MS",
        ),
    )


# Define a time axis
def add_times(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign_coords(
        time=date_range(start="1979-01-01", periods=ds.time.size, freq="MS")
    )


# cleaner warning system - sjs 2022.01.28
def validate_against_metric(
    computed_arr: xr.DataArray, metric: str, tol: float = 1e-5
) -> bool:
    """
    compare computed array for equality (within round-off error) to data
    in provided validation files and print detalied messages of results

    ::input::
    computed_arr - array to be validated
    metric (string) - short string corresponding to metric to validate with
    tol (float) - relative tolerance for comparing to validation data

    ::output::
    (bool) - whether or not the array is valid
    """

    # parse input metric str for ann vs monthly vs seasonal
    base_metric = metric.split("_")[0]
    if "_" not in metric:
        freq_str = "Monthly"
    elif "_ANN" in metric:
        freq_str = "Annual-mean"
    else:
        freq_str = metric.split("_")[-1]

    # get the validated metric
    Phi_valid = get_validated_metric(metric)

    # check if equal (within round-off)
    computed_arr, Phi_valid = xr.align(computed_arr, Phi_valid, join="outer")
    pct_passing = (
        np.isclose(computed_arr, Phi_valid, rtol=tol).sum() / computed_arr.size * 100.0
    )
    if pct_passing > 99.999:
        logging.info(
            f"OK. {freq_str} validation and calculated "
            f"{base_metric} metrics are the same!"
        )
        return True
    else:
        logging.warning(
            f"{freq_str} validation and calculated "
            f" {base_metric} metrics are NOT equal with "
            f"{pct_passing:.1f}% matching"
        )
        return False


ANN_TEST = False

v_data = xr.open_dataset(data_dir / "va.nc")
u_data = xr.open_dataset(data_dir / "ua.nc")
uv_data = xr.merge([v_data, u_data])
uv_data = add_times(uv_data.rename(ua="u", va="v"))


if ANN_TEST:  # yearly mean of data
    uv_data = uv_data.resample(time="AS").mean()
psi_metrics, psi_vals = uv_data.pyt_metrics.xr_psi()
validate_against_metric(psi_metrics, "PSI" + ("_ANN" if ANN_TEST else ""))


# Tropopause break
# read temperature T(time,lat,lev), latitude and level
T_data = add_times(xr.open_dataset(data_dir / "ta.nc"))
if ANN_TEST:  # yearly mean of data
    T_data = T_data.resample(time="AS").mean()
tpb_metrics = T_data.pyt_metrics.xr_tpb()
validate_against_metric(tpb_metrics, "TPB" + ("_ANN" if ANN_TEST else ""))

# Surface pressure max (Invalid in NH)
# read sea-level pressure ps(time,lat) and latitude
psl_data = add_times(xr.open_dataset(data_dir / "psl.nc"))
psl_seasonal = psl_data.resample(time="QS-DEC").mean()
psl_seasonal_metrics = psl_seasonal.pyt_metrics.xr_psl()
for ssn in ["DJF", "MAM", "JJA", "SON"]:
    psl_metrics = psl_seasonal_metrics.sel(time=psl_seasonal.time.dt.season == ssn)
    if ssn == "DJF":
        psl_metrics = psl_metrics.isel(time=slice(None, -1))
    validate_against_metric(psl_metrics, f"PSL_{ssn}")


# Eddy driven jet
edj_metrics, edj_vals = uv_data.sel(lev=850, method="nearest").pyt_metrics.xr_edj()
validate_against_metric(edj_metrics, "EDJ" + ("_ANN" if ANN_TEST else ""))


# Subtropical jet
stj_metrics, stj_vals = uv_data.pyt_metrics.xr_stj()
validate_against_metric(stj_metrics, "STJ" + ("_ANN" if ANN_TEST else ""))


# OLR
# read zonal mean monthly TOA outgoing longwave radiation olr(time,lat)
olr_data = add_times(-xr.open_dataset(data_dir / "rlnt.nc"))
if ANN_TEST:  # yearly mean of data
    olr_data = olr_data.resample(time="AS").mean()
olr_metrics = olr_data.pyt_metrics.xr_olr()
validate_against_metric(olr_metrics, "OLR" + ("_ANN" if ANN_TEST else ""))

# P minus E
# read zonal mean monthly precipitation pr(time,lat)
pr_data = xr.open_dataset(data_dir / "pr.nc")
latent_heat_flux = xr.open_dataset(data_dir / "hfls.nc")
# use latent heat of vap to convert LHF to evap
er_data = -latent_heat_flux / 2510400.0
pe_data = add_times((pr_data.pr - er_data.hfls).to_dataset(name="pe"))
if ANN_TEST:  # yearly mean of data
    pe_data = pe_data.resample(time="AS").mean()
pe_metrics = pe_data.pyt_metrics.xr_pe()
validate_against_metric(pe_metrics, "PE" + ("_ANN" if ANN_TEST else ""))


# Surface winds
# read zonal mean surface wind U(time,lat)
uas_data = add_times(xr.open_dataset(data_dir / "uas.nc"))
if ANN_TEST:  # yearly mean of data
    uas_data = uas_data.resample(time="AS").mean()
uas_metrics = uas_data.pyt_metrics.xr_uas()
validate_against_metric(uas_metrics, "UAS" + ("_ANN" if ANN_TEST else ""))
