# Install Jinja2 if needed: pip install jinja2
from jinja2 import Template

# Define variables
variables = {
    "seq_len": 336,
    "label_len": 336,
    "pred_len": 96,
    "stride": 1,
    "kernel_size": 25,
    "d_model": 512,
    "n_heads": 8,
    "batch_size": 8,
    "enc_in": 321,
    "udefined_v": 4,
    "t_dim": 1
}

# Load the template and inject variables
with open("./.Tmp_Key_params.md", "r") as f:
    template = Template(f.read())

# Render the final Markdown
final_markdown = template.render(**variables)

# Save to a new file
with open("./Key_params.md", "w") as f:
    f.write(final_markdown)