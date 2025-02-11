import os
import re
import curses
import subprocess
from functools import partial

def get_script_files():
    """
    Scan the current directory and return a sorted list of files
    that match the pattern va_v1.X.py (e.g., va_v1.0.py, va_v1.1.py, etc.).
    """
    files = os.listdir('.')
    # Compile a regex to match filenames like va_v1.0.py, va_v1.1.py, etc.
    pattern = re.compile(r'^va_v1\.\d+\.py$')
    scripts = [f for f in files if pattern.match(f)]
    # Sort the scripts by their version number (converted to float for proper sorting)
    scripts.sort(key=lambda f: float(re.search(r'\d+\.\d+', f).group()))
    return scripts

def menu(stdscr, options):
    """
    Display a menu using curses where the user can navigate with arrow keys.
    Returns the option that the user selects when Enter is pressed.
    """
    # Hide the cursor
    curses.curs_set(0)
    # Initialize a color pair (foreground, background)
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    
    current_row = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Display each menu option, highlighting the current selection
        for idx, row in enumerate(options):
            x = width // 2 - len(row) // 2
            y = height // 2 - len(options) // 2 + idx
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row)
        stdscr.refresh()

        # Wait for user input
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        # Handle Enter (curses.KEY_ENTER may not be available on all systems; also check 10 and 13)
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_row]

def run_script(script):
    """
    Run the selected script using the python command.
    """
    subprocess.call(['python', script])

def run_menu(stdscr, options):
    """
    Wrapper function for curses which returns the selected option.
    """
    return menu(stdscr, options)

def main():
    scripts = get_script_files()
    if not scripts:
        print("No matching script files found in the current directory.")
        return

    # Use curses.wrapper to initialize curses and call our menu function.
    selected_script = curses.wrapper(lambda stdscr: run_menu(stdscr, scripts))
    print(f"Running: {selected_script}")
    run_script(selected_script)

if __name__ == "__main__":
    main()
