import sys
import os


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the root directory of the workspace
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

print(os.path.abspath(__file__))
print(current_dir)
print(os.path.join(current_dir, "..", ".."))
print(root_dir)

# Add the root directory to the Python path
sys.path.append(root_dir)

# Now you can print the Python path
for path in sys.path:
    pass  # print(path)
