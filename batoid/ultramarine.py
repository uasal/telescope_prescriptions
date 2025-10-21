
"""
Ultra Marine Space Telescope Optical System on Batoid

Author: Pierre Raphaël Nicolas
Date: 10/14/2025
"""


import os
import batoid
import logging
import yaml
import astropy.units as u
from scipy.spatial.transform import Rotation
import numpy as np
from functools import reduce

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from BaZIL.core.optical_system import OpticalSystem

class UltraMarine(OpticalSystem):
    def __init__(self, config=None, show_methods=False):

        if config is not None:
            self.config = config
        else:
            raise ValueError("Configuration file must be provided.")

        # Load metadata from YAML file
        self.metaData = self._get_metadata(self.config)

        # Key parameters
        self.fieldBias = self.metaData['fieldBias'] * u.deg
        self.wavelength = self.metaData['wavelengths'][0] * u.m

        # Build telescope
        self.telescope = None
        self.initialize_telescope()

        super().__init__(self.telescope)

        if show_methods:
            logger.info(f'Available methods: {self.methods}')
    
    @classmethod
    def _get_metadata(cls, filename):
        with open(filename) as f:
            config = yaml.safe_load(f)
        return config['metaData']
    
    def get_field(self, field_number):
        return np.array(self.metaData['fields'][field_number-1])

    def initialize_telescope(self):
        telescope = batoid.Optic.fromYaml(self.config)

        # Field bias applied to the whole optical system (except entrance pupil)
        field_bias_rotation_matrix = Rotation.from_euler('XYZ', np.array([self.fieldBias.to_value(u.deg), 0, 0]), degrees=True).as_matrix()
        telescope = reduce(
            lambda tel, optic: tel.withGloballyRotatedOptic(optic.name, field_bias_rotation_matrix, rotCenter=[0,0,0]),
            telescope.items, telescope
        )

        self.telescope = telescope


####################################################################################

def main():
    YAML_FILE_NAME = os.path.join(os.path.dirname(__file__), "Lazuli_Mark-14_release_17_config1.yaml")

    UM = UltraMarine(config=YAML_FILE_NAME, show_methods=True)
    field_number = 1 # Define target field based on index (same convention as Zemax)

    theta_x, theta_y = -UM.get_field(field_number) * u.deg
    logger.info(f'theta_x = {theta_x.to_string()}, theta_y = {theta_y.to_string()}')

    # UM.display_wavefront(
    #     display_params={'type':'original'}, # remove_tilt / original
    #     exit_pupil_size=UM.metaData['exitPupilSize'] * u.m,
    #     wavelength=UM.wavelength,
    #     theta_x=theta_x, theta_y=theta_y,
    #     nx=128
    # )

    # UM.display_spot(
    #     wavelength=UM.wavelength,
    #     theta_x=theta_x, theta_y=theta_y,
    #     density=50, pattern='Hexapolar', scale=10
    # )

    UM.show_layout(
        wavelength=UM.wavelength,
        theta_x=theta_x, theta_y=theta_y,
        show='3D', axis='xy', N=20
    )


if __name__ == "__main__": 
    main()