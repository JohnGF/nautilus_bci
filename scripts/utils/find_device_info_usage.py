import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import os

search_dir = r"C:\Users\VR\Desktop\BCI gtec"
matches = []

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith(('.py', '.cs', '.txt', '.md', '.json', '.xml')):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'GetDeviceInformation' in content:
                        matches.append(path)
            except Exception:
                pass

print("Matches found in workspace:")
for m in matches:
    print(m)
