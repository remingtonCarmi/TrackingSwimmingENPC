"""
This module containts the class interval.
"""


class Interval:
    """
    Class interval that defines an interval on N or R.
    """
    def __init__(self, left, right):
        """
        Construct the interval.

        Args:
            left (int or float): the left side of the interval.

            right (int or float): the right side of the interval.
        """
        self.left = left
        self.right = right

    def __len__(self):
        return int(self.right - self.left)

    def __contains__(self, item):
        """
        If the item is an interval, says if the interval contains the item.
        Otherwise, says if the int or the float is in the interval.
        """
        if type(item) == Interval:
            return self.left <= item.left and item.right <= self.right
        else:
            return self.left <= item <= self.right

    def __repr__(self):
        return "[{}, {}]".format(self.left, self.right)

    def __gt__(self, other):
        """
        Compare the len of the intervals.
        """
        return len(self) > len(other)


if __name__ == "__main__":
    INTERVAL = Interval(9, 19)
    INTERVAL1 = Interval(1, 2)

    print(INTERVAL)
    print(INTERVAL < INTERVAL1)
