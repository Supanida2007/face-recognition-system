from enum import Enum

class Color(Enum):
    """
    Standard BGR color values for use with OpenCV drawing functions.
    These colors follow the BGR (Blue, Green, Red) format used by OpenCV.
    """
    RED     = (0, 0, 255)
    GREEN   = (0, 255, 0)
    BLUE    = (255, 0, 0)
    YELLOW  = (0, 255, 255)
    CYAN    = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    ORANGE  = (0, 165, 255)
    PURPLE  = (128, 0, 128)
    GRAY    = (128, 128, 128)
    BLACK   = (0, 0, 0)
    WHITE   = (255, 255, 255)
    CUSTOM   = (127, 95, 255)

if __name__ == "__main__":
    print("Available BGR Colors:")
    for color in Color:
        print(f"{color.name}: {color.value}")