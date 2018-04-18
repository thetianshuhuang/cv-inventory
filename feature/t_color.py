#
# t_color.py
# terminal color definitions
#
# Original implementation written for Serial-Vis by Tianshu Huang, Spring 2018
#


class color:

    # standard 8 color set
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'
    magenta = '\033[35m'
    cyan = '\033[36m'
    white = '\033[37m'

    # 16 color set
    bblack = '\033[30;1m'
    bred = '\033[31;1m'
    bgreen = '\033[32;1m'
    byellow = '\033[33;1m'
    bblue = '\033[34;1m'
    bmagenta = '\033[35;1m'
    bcyan = '\033[36;1m'
    bwhite = '\033[37;1m'

    # 8 color background
    bgblack = '\033[40m'
    bgred = '\033[41m'
    bggreen = '\033[42m'
    bgyellow = '\033[43m'
    bgblue = '\033[44m'
    bgmagenta = '\033[45m'
    bgcyan = '\033[46m'
    bgwhite = '\033[47m'

    # 16 bit background
    bgbblack = '\033[40;1m'
    bgbred = '\033[41;1m'
    bgbgreen = '\033[42;1m'
    bgbyellow = '\033[43;1m'
    bgbblue = '\033[44;1m'
    bgbmagenta = '\033[45;1m'
    bgbcyan = '\033[46;1m'
    bgbwhite = '\033[47;1m'

    # format
    bold = '\033[1m'
    underline = '\033[4m'
    inverted = '\033[7m'
    end = '\033[0m'
    bell = '\a'
