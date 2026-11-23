import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations, combinations_with_replacement


class RaceSimulator:
    """
    Simulates an F1 race strategy by calculating the time added
    from tire degradation and pit stops.
    """
    RACE_LAPS = 60
    NUM_STINTS = 3
    PIT_STOP_TIME = 25

    def __init__(self):
        """
        Initialize the simulator and store tire data.
        """
        self.tires=pd.DataFrame(
            {
                "Tmin": [0, 1, 2],
                "Tmax": [10, 10, 10],
                "alpha": [1 / 2, 1 / 4, 1 / 16],
                "beta": [15, 25, 55],
            },
            index=["Soft", "Medium", "Hard"],
        )


    def tire_time(self, age, Tmin, Tmax, alpha, beta):
        """
        Calculate the time added due to tire degradation.

        Parameters:
            age: Tire age in laps.
            Tmin: Minimum time added by the tire.
            Tmax: Maximum time added by the tire.
            alpha: Tire degradation rate parameter.
            beta: Tire degradation midpoint parameter.

        Returns:
            Tire time added for the given tire age.
        """
        return Tmin + (Tmax - Tmin) / (1 + np.exp(-alpha * (age - beta)))

    def _validate(self, stint_lengths, tire_types):

        if len(stint_lengths) != self.NUM_STINTS or len(tire_types) != self.NUM_STINTS:
            raise ValueError(
                f"A race needs exactly {self.NUM_STINTS} stints (two pit stops), so exactly {self.NUM_STINTS} lengths and tire types."
            )

        for length in stint_lengths:
            if not isinstance(length, (int, np.integer)) or length < 1:
                raise ValueError("Each stint length must be a whole number of laps, at least 1.")

        if sum(stint_lengths) != self.RACE_LAPS:
            raise ValueError(
                f"Stint lengths must add up to {self.RACE_LAPS} laps, got {sum(stint_lengths)}."
            )

        cleaned_tires = []
        for tire in tire_types:
            name = str(tire).strip().capitalize()
            if name not in self.tires.index:
                raise ValueError(
                    f"Unknown tire type {tire!r}. Choose from {list(self.tires.index)}."
                )
            cleaned_tires.append(name)

        if len(set(cleaned_tires)) < 2:
            raise ValueError("At least 2 different tire types must be used.")

        return cleaned_tires

    def build_race(self, stint_lengths, tire_types):
        """
        Builds DataFrames describing every lap of the race.

        One DataFrame is built for each stint, then they are concatenated into a single DataFrame for the whole race.

        Parameters:
            stint_lengths: Length of each race stint.
            tire_types: Tire used for each stint.

        Returns:
            DataFrame with one row per lap with the columns:
            lap, stint, tire, age, tire_time, pit_time, total_time.
        """
        tire_types = self._validate(stint_lengths, tire_types)

        stint_frames = []
        start_lap = 1

        for stint_number in range(self.NUM_STINTS):
            length = stint_lengths[stint_number]
            tire = tire_types[stint_number]
            params = self.tires.loc[tire]

            # new tires on the first lap of the stint, so ages start at 0
            ages = np.arange(length)
            tire_added = self.tire_time(
                ages, params["Tmin"], params["Tmax"], params["alpha"], params["beta"]
            )

            # a pit stop happens on the first lap of every stint after the first
            if stint_number > 0:
                pit_amount = self.PIT_STOP_TIME
            else:
                pit_amount = 0.0
            pit_added = np.where(ages == 0, pit_amount, 0.0)

            stint_frame = pd.DataFrame(
                {
                    "lap": start_lap + ages,
                    "stint": stint_number + 1,
                    "tire": tire,
                    "age": ages,
                    "tire_time": tire_added,
                    "pit_time": pit_added,
                }
            )
            stint_frames.append(stint_frame)
            start_lap += length

        race = pd.concat(stint_frames, ignore_index=True)
        race["total_time"] = race["tire_time"] + race["pit_time"]
        return race


    def simulate(self, stint_lengths, tire_types):
        """
        Simulate a complete race strategy.

        Parameters:
            stint_lengths: Length of each race stint.
            tire_types: Tire compound used for each stint.

        Returns:
            Total time added from tire degradation and pit stops.
        """
        race = self.build_race(stint_lengths, tire_types)
        return float(race["total_time"].sum())


def plot_tires(simulator):
    """
    Plot tire degradation curves for all tire compounds.

    Parameters:
        simulator: A RaceSimulator object containing tire data.

    Returns:
        None.
    """
    tires = simulator.tires

    # ages as a column (61) and parameters as rows (3) so NumPy broadcasting computes every tire's curve at once (61 x 3)
    ages = np.arange(simulator.RACE_LAPS + 1)
    times = simulator.tire_time(
        ages[:, np.newaxis],
        tires["Tmin"].to_numpy(),
        tires["Tmax"].to_numpy(),
        tires["alpha"].to_numpy(),
        tires["beta"].to_numpy(),
    )

    plt.figure(figsize=(9, 6))
    plt.plot(ages, times)
    plt.legend(tires.index)
    plt.xlabel("Tire age (laps)")
    plt.ylabel("Time added (s)")
    plt.title("Tire degradation")
    plt.grid(True)
    plt.show()




def find_best_strategy(simulator):
    """
    Test all valid race strategies and find the one
    with the minimum total time added.

    Pit stop laps come from combinations (first stop before the second).
    Tire choices come from combinations_with_replacement. 
    The order of the tires does not change the total time (each stint's time depends only on  its own tire and length,
    and every stint length order is tried), so trying each set of tires once is enough.

    Parameters:
        simulator: A RaceSimulator object.

    Returns:
       A dictionary with the best strategy's pit_laps, stint_lengths, tire_types, and total_time.
    """

    laps = simulator.RACE_LAPS
    tire_names = list(simulator.tires.index)

    tire_options = list(combinations_with_replacement(tire_names, simulator.NUM_STINTS))
    pit_options = list(combinations(range(2, laps + 1), 2))

    best = None

    for tire_types in tire_options:
        # must use at least 2 different tire types
        if len(set(tire_types)) < 2:
            continue

        for first_pit, second_pit in pit_options:
            stint_lengths = (first_pit - 1, second_pit - first_pit, laps - second_pit + 1)
            total = simulator.simulate(stint_lengths, tire_types)

            if best is None or total < best["total_time"]:
                best = {
                    "pit_laps": (first_pit, second_pit),
                    "stint_lengths": stint_lengths,
                    "tire_types": tire_types,
                    "total_time": total,
                }

    return best