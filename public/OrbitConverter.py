from math import degrees, radians

import orekit
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.orbits import KeplerianOrbit, PositionAngle, OrbitType, CircularOrbit, CartesianOrbit
from org.orekit.utils import Constants, PVCoordinatesProvider, PVCoordinates
from orekit.pyhelpers import setup_orekit_curdir
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import IERSConventions
from org.orekit.frames import FramesFactory

from public.DatasetParser import DatasetParser


class OrbitConverter():
    def __init__(self):
        vm = orekit.initVM()
        print("Java version:", vm.java_version)
        print('Orekit version:', orekit.VERSION)
        setup_orekit_curdir()

        # Some Constants
        self.ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        self.mu = Constants.WGS84_EARTH_MU
        self.utc = TimeScalesFactory.getUTC()
        self.convention = IERSConventions.IERS_2010

        # Inertial frame where the satellite is defined
        self.inertialFrame = FramesFactory.getEME2000()

    def cartesianToCircular(self, x, y, z, dx, dy, dz, date):
        pv_coordinates = PVCoordinates(Vector3D(float(x), float(y), float(z)),
                                       Vector3D(float(dx), float(dy), float(dz)))
        cartesian_orbit = CartesianOrbit(
            pv_coordinates,
            self.inertialFrame,
            AbsoluteDate(date, self.utc),
            self.mu
        )
        circular_orbit = CircularOrbit.cast_(OrbitType.CIRCULAR.convertType(cartesian_orbit))
        return {"a": circular_orbit.getA() / 1000, "ex": circular_orbit.getCircularEx(),
                "ey": circular_orbit.getCircularEy(), "i": degrees(circular_orbit.getI()) % 360,
                "raan": degrees(circular_orbit.getRightAscensionOfAscendingNode()) % 360,
                "alpha_v": degrees(circular_orbit.getAlphaV() % 360)}

    def circularToCartesian(self, a_km, ex, ey, i_deg, raan_deg, alpha_v, date):
        cicular_orbit = CircularOrbit(
            float(a_km * 1000), float(ex), float(ey), float(radians(i_deg)), float(radians(raan_deg)), float(radians(alpha_v)), PositionAngle.TRUE,
            self.inertialFrame,
            AbsoluteDate(date, self.utc),
            self.mu
        )
        cartesian_orbit = CartesianOrbit.cast_(OrbitType.CARTESIAN.convertType(cicular_orbit))
        pv_coordinates = cartesian_orbit.getPVCoordinates()
        position = pv_coordinates.getPosition()
        velocity = pv_coordinates.getVelocity()
        return {"x": position.getX(), "y": position.getY(), "z": position.getZ(), "dx": velocity.getX(),
                "dy": velocity.getY(), "dz": velocity.getZ()}


if __name__ == '__main__':
    print("--Converting cartesian to circular--")
    parser = DatasetParser(False, input_path="../output/challenge_1_test_dataset_full_year_eval.json")
    parser.min_max_scale = False
    cartesian_coordinates, _ = parser.create_input_target(0)
    converter = OrbitConverter()
    cartesian_coordinates = cartesian_coordinates.numpy()[0, :]
    print(f"initial coordinates:{cartesian_coordinates}")
    circular_output=converter.cartesianToCircular(cartesian_coordinates[0], cartesian_coordinates[1], cartesian_coordinates[2],
                                        cartesian_coordinates[3], cartesian_coordinates[4], cartesian_coordinates[5],
                                        parser.parse_data(0, "date")[0])
    print(f"final coordinates:{circular_output}")

    print("--Converting circular to cartesian--")
    parser.input_variables = ["a", "ex", "ey", "i", "raan", "av"]
    circular_coordinates, _ = parser.create_input_target(0)
    circular_coordinates = circular_coordinates.numpy()[0, :]
    print(f"initial coordinates:{circular_coordinates}")
    cartesian_output=converter.circularToCartesian(circular_coordinates[0], circular_coordinates[1], circular_coordinates[2],
                                        circular_coordinates[3], circular_coordinates[4], circular_coordinates[5],
                                        parser.parse_data(0, "date")[0])
    print(f"final coordinates:{cartesian_output}")
