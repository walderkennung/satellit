from typing import Literal

DbhUnit = Literal["mm", "cm", "m"]


def to_dbh_cm(dbh_raw: float, dbh_unit: DbhUnit) -> float:
    """Convert a DBH value into centimeters.

    Args:
        dbh_raw: Raw DBH value from the input inventory.
        dbh_unit: Unit of ``dbh_raw`` (``mm``, ``cm``, or ``m``).

    Returns:
        DBH value converted to centimeters.
    """
    if dbh_unit == "mm":
        return dbh_raw / 10.0
    if dbh_unit == "cm":
        return dbh_raw
    return dbh_raw * 100.0


CrownModel = Literal["linear", "power"]


def compute_crown_radius_m(
    dbh_cm: float,
    crown_model: CrownModel,
    linear_factor_m_per_cm: float,
    linear_intercept_m: float,
    power_a: float,
    power_b: float,
    min_crown_radius_m: float,
    max_crown_radius_m: float,
) -> float:
    """Estimate crown radius from DBH and clamp to configured bounds.

    Args:
        dbh_cm: Diameter at breast height in centimeters.
        crown_model: Regression family used for the estimate.
        linear_factor_m_per_cm: Slope used by the ``linear`` model.
        linear_intercept_m: Intercept used by the ``linear`` model.
        power_a: Multiplicative coefficient used by the ``power`` model.
        power_b: Exponent used by the ``power`` model.
        min_crown_radius_m: Lower bound for the returned radius.
        max_crown_radius_m: Upper bound for the returned radius.

    Returns:
        Estimated crown radius in meters.
    """
    if crown_model == "linear":
        value = linear_intercept_m + linear_factor_m_per_cm * dbh_cm
    else:
        value = power_a * (dbh_cm**power_b)

    if value < min_crown_radius_m:
        return min_crown_radius_m
    if value > max_crown_radius_m:
        return max_crown_radius_m
    return value
