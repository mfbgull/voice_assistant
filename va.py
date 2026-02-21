import os
import re
import subprocess

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

    # If no va_v1.x.py files found, look for va1.py, va2.py, va3.py, va4.py pattern
    if not scripts:
        pattern_alt = re.compile(r'^va[1-9]\.py$')
        scripts = [f for f in files if pattern_alt.match(f)]
        scripts.sort()

    return scripts

def display_menu(options):
    """
    Display a simple menu with numbered options.
    Returns the option that the user selects.
    """
    if not options:
        print("No script files found!")
        return None

    print("\n" + "="*50)
    print("Available Voice Assistant Scripts:")
    print("="*50)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print(f"{len(options)+1}. Exit")
    print("-"*50)

    while True:
        try:
            choice = int(input(f"Select an option (1-{len(options)+1}): "))
            if 1 <= choice <= len(options):
                return options[choice-1]
            elif choice == len(options) + 1:
                return None
            else:
                print(f"Please enter a number between 1 and {len(options)+1}")
        except ValueError:
            print("Please enter a valid number")

def run_script(script):
    """
    Run the selected script using the python3 command.
    """
    try:
        result = subprocess.run(['python3', script], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        return e.returncode

def main():
    scripts = get_script_files()
    if not scripts:
        print("No matching script files found in the current directory.")
        print("Expected files matching pattern: va_v1.X.py or va1.py, va2.py, etc.")
        return

    print(f"Found {len(scripts)} script(s): {scripts}")

    selected_script = display_menu(scripts)
    if selected_script:
        print(f"\nRunning: {selected_script}")
        run_script(selected_script)
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
