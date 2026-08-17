"""Physical constants used by the Tolman-Oppenheimer-Volkoff equations.

The solver uses radius in km, mass in solar masses, and energy density and
pressure in MeV fm^-3.  Keeping every conversion here makes the unit system
visible without reading the numerical integrator.
"""

from __future__ import annotations

import math

STELLAR_CONSTANT_AUTHORITY = (
    "CompOSE Reference Manual v3.01 constants table; SI definitions"
)
STELLAR_CONSTANT_REFERENCE_URL = (
    "https://compose.obspm.fr/download/pdf/manual_v3.00.pdf"
)

SPEED_OF_LIGHT_M_S = 299_792_458.0
NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11
SOLAR_MASS_KG = 1.98841e30
MEV_J = 1.602176634e-13
FM3_M3 = 1.0e-45

# In dm/dr = A r^2 epsilon, r is km, epsilon is MeV fm^-3, and m is
# measured in solar masses.  The 1e9 factor converts km^3 to m^3.
GRAVITY_CONVERSION = (
    4.0 * math.pi * 1.0e9 * (MEV_J / FM3_M3) / (SPEED_OF_LIGHT_M_S**2 * SOLAR_MASS_KG)
)

# Geometrized length GM_sun/c^2 expressed in km.
SOLAR_MASS_LENGTH_KM = (
    NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2
    * SOLAR_MASS_KG
    / SPEED_OF_LIGHT_M_S**2
    / 1.0e3
)


def constants_to_dict() -> dict[str, str | float]:
    """Return the exact constants recorded with every stellar result."""

    return {
        "authority": STELLAR_CONSTANT_AUTHORITY,
        "authority_url": STELLAR_CONSTANT_REFERENCE_URL,
        "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
        "newtonian_gravitational_constant_m3_kg_s2": (
            NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2
        ),
        "solar_mass_kg": SOLAR_MASS_KG,
        "MeV_J": MEV_J,
        "fm3_m3": FM3_M3,
        "gravity_conversion_Msun_per_km3_per_MeV_fm3": GRAVITY_CONVERSION,
        "solar_mass_length_km": SOLAR_MASS_LENGTH_KM,
    }


# Historical private name retained for internal and downstream compatibility.
_constants_to_dict = constants_to_dict
