import colorsys
from typing import List, Tuple

Color = Tuple[int, int, int]


def fibonacci_color_palette(num_colors: int, background: bool = False) -> List[Color]:
    """
    Generates a list of colors in a Fibonacci spacing in HSV space.

    Args:
    - num_colors (int): An integer representing the number of colors to generate.
    - background (bool): A boolean indicating if a black color should be added to the list.

    Returns:
    List[Tuple[int, int, int]]: A list of RGB tuples representing the generated colors.
    """

    # Define the golden ratio constant
    golden_ratio_conjugate = 0.6180339887498948

    # Initialize the list to store the colors in
    if background:
        colors = [(0, 0, 0)]
    else:
        colors = []

    # Generate the colors using the golden ratio in HSV space
    hue = 0.5
    for i in range(num_colors - len(colors)):
        # Use modulo to keep hue between 0 and 1
        hue = (hue + golden_ratio_conjugate) % 1

        # Set saturation and value to 0.8 and 0.9 respectively
        saturation = 0.8
        value = 0.9

        # Convert from HSV to RGB
        rgb_tuple = tuple(
            round(i * 255) for i in colorsys.hsv_to_rgb(hue, saturation, value)
        )

        # Append the RGB tuple to the list of colors
        colors.append(rgb_tuple)

    return colors
