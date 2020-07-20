"""
This module contains the class Prediction Manager.
"""
# To manipulate intervals
from src.d7_visualization.prediction_manager.interval.interval import Interval


class PredictionManager:
    """
    Class to manage the predictions.
    """
    def __init__(self):
        self.intervals = []

    def add_prediction(self, begin_limit, end_limit):
        """
        Add the prediction to the registered list.

        Args:
            begin_limit (int or float): the left side of the interval.

            end_limit (int or float): the right side of the interval.
        """
        # Construct the new interval
        new_interval = Interval(begin_limit, end_limit)

        # -- Insert the new interval -- #
        useful = True

        # List of the interval to be removed
        to_remove = []

        # Check if any interval is in the new interval or if any interval contains the new interval
        for interval in self.intervals:
            # If the interval contains the new interval
            if new_interval in interval:
                useful = False
            # If the possible interval contains the interval
            elif interval in new_interval:
                to_remove.append(interval)

        if useful:
            # Remove the stopped intervals
            for interval in to_remove:
                self.intervals.remove(interval)
            # Update to_remove
            to_remove = []

            # Manage over_laps
            for interval in self.intervals:
                # If the possible interval is at the right of the interval and the intersection is not empty
                if interval.right in new_interval:
                    new_interval.left = interval.left
                    to_remove.append(interval)
                # If the possible interval is at the left of the interval and the intersection is not empty
                elif interval.left in new_interval:
                    new_interval.right = interval.right
                    to_remove.append(interval)
            for interval in to_remove:
                self.intervals.remove(interval)

            # Add the new interval
            self.intervals.append(new_interval)

    def get_bigest_interval(self):
        """
        Returns the biggest interval considering the len of the interval.
        """
        if len(self.intervals) > 0:
            return max(self.intervals)
        else:
            return Interval(-1, -1)


if __name__ == "__main__":
    INTERVALS = PredictionManager()

    print(INTERVALS.get_bigest_interval())
