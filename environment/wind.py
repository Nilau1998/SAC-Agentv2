import numpy as np
from scipy.interpolate import interp1d


class Wind:
    """
    This class defines the wind for the boat environment. Within this class a 2 functions will be defined together with helper methods that might come in handy.
    The first function will define a continous range which describes the winds velocity over a whole episode.
    The second function will define the continous range for the angle of the wind over a whole episode.
    """

    def __init__(self, config):
        self.config = config
        self.wind_range_length = int(
            config.base_settings.t_max / config.base_settings.dt
        )
        self.wind_velocity = 0
        self.wind_angle = 0
        self.generate_wind()

    def get_wind(self, index):
        """
        Returns wind attributes for this timestep.
        """
        return np.array([self.wind_velocity[index], self.wind_angle[index]])

    def generate_wind(self):
        """
        Generates a range that defines the wind force that can be used to affect the boat.
        """
        match int(self.config.base_settings.experiment):
            case 1:  # No wind
                self.wind_velocity = np.full(self.wind_range_length, 0)
                self.wind_angle = np.full(self.wind_range_length, 0)

            case 2:  # Changing starting y value, no wind
                self.wind_velocity = np.full(self.wind_range_length, 0)
                self.wind_angle = np.full(self.wind_range_length, 0)
                # y position changing happens in boat_env.py

            case 3:  # Constant wind velocity, wind direction bottom to top
                max_velocity = float(self.config.wind.max_velocity)
                self.wind_velocity = np.full(self.wind_range_length, max_velocity)
                angle = float(self.config.wind.direction) * (np.pi / 180)
                self.wind_angle = np.full(self.wind_range_length, angle)

            case 4:  # Changing wind velocity, wind direction bottom to top
                max_velocity = float(self.config.wind.max_velocity)
                self.wind_velocity = self.generate_random_curve() * max_velocity
                angle = float(self.config.wind.direction) * (np.pi / 180)
                self.wind_angle = np.full(self.wind_range_length, angle)

            case 5:  # Constant wind velocity, swap between b to t or t to b
                max_velocity = float(self.config.wind.max_velocity)
                self.wind_velocity = np.full(self.wind_range_length, max_velocity)
                self.wind_angle = (
                    self.rect_random_curve(middle=0.5) * np.pi
                ) + np.pi / 2

            case 6:  # Changing wind velocity, all wind directions random
                max_velocity = float(self.config.wind.max_velocity)
                self.wind_velocity = self.generate_random_curve() * max_velocity
                self.wind_angle = self.generate_random_curve() * np.pi * 2

            case _:
                raise ValueError(
                    "Well someone tried to use an experiment that doesnt exist!"
                )

    def generate_random_curve(self):
        """
        Generates a random range/curve around n fixed points. This function can be used to generate for example the wind curves. The curve is generated in a 0-1 range and therefore has to be multiplied by 2pi or whatever wind speed is used.
        """
        if self.config.wind.fixed_points < 4:
            raise ValueError(
                "Please select at least 4 fixed_points in your config. The interpolation doesn't work otherwise!"
            )
        fixed_points = np.linspace(
            0, self.wind_range_length, num=self.config.wind.fixed_points
        )
        fixed_point_values = np.random.sample(self.config.wind.fixed_points)

        complete_range = np.linspace(
            0, self.wind_range_length, num=self.wind_range_length, endpoint=True
        )
        interpolation = interp1d(
            fixed_points, fixed_point_values, kind="cubic", fill_value="extrapolate"
        )
        interpolated_range = interpolation(complete_range)

        # Normalize incase interpolation exceeds 0-1 range.
        if np.any((interpolated_range < 0) | (interpolated_range > 1)):
            interpolated_range = (interpolated_range - np.min(interpolated_range)) / (
                np.max(interpolated_range) - np.min(interpolated_range)
            )
        return interpolated_range

    def rect_random_curve(self, middle):
        random_curve = self.generate_random_curve()
        for i, value in enumerate(random_curve):
            if value <= middle / 2:
                random_curve[i] = 0
            else:
                random_curve[i] = 1
        return random_curve
