import re
import os

filepath = r'c:\\Users\\H P\\Desktop\\FinalProject_PapillaPicasso\\src\\dip_fina_project.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Add # %% before coding utf-8
content = re.sub(r'(# -\*- coding: utf-8 -\*)', r'# %%\n\1', content)

# Add # %% before any """CELL
content = re.sub(r'(\"\"\"CELL \d+:)', r'# %%\n\1', content)

# Add # %% before pip install
content = re.sub(r'\n(pip install )', r'\n\n# %%\n\1', content)
content = re.sub(r'\n(!pip install )', r'\n\n# %%\n\1', content)

# Add # %% before import gradio
content = re.sub(r'\n(import gradio as gr)', r'\n\n# %%\n\1', content)

# Remove multiple # %% if they got duplicated accidentally
content = re.sub(r'(# %%\n)+', r'# %%\n', content)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Cells added successfully.")
