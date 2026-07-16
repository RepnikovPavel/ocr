import os
import sys
import json
import re

from PIL import Image
from dots_mocr.utils.image_utils import PILimage_to_base64


def has_latex_markdown(text: str) -> bool:
    """
    Checks if a string contains LaTeX markdown patterns.
    
    Args:
        text (str): The string to check.
        
    Returns:
        bool: True if LaTeX markdown is found, otherwise False.
    """
    if not isinstance(text, str):
        return False
    
    # Define regular expression patterns for LaTeX markdown
    latex_patterns = [
        r'\$\$.*?\$\$',           # Block-level math formula $$...$$
        r'\$[^$\n]+?\$',          # Inline math formula $...$
        r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environment \begin{...}...\end{...}
        r'\\[a-zA-Z]+\{.*?\}',    # LaTeX command \command{...}
        r'\\[a-zA-Z]+',           # Simple LaTeX command \command
        r'\\\[.*?\\\]',           # Display math formula \[...\]
        r'\\\(.*?\\\)',           # Inline math formula \(...\)
    ]
    
    # Check if any of the patterns match
    for pattern in latex_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    
    return False


def clean_latex_preamble(latex_text: str) -> str:
    """
    Removes LaTeX preamble commands like document class and package imports.
    
    Args:
        latex_text (str): The original LaTeX text.

    Returns:
        str: The cleaned LaTeX text without preamble commands.
    """
    # Define patterns to be removed
    patterns = [
        r'\\documentclass\{[^}]+\}',  # \documentclass{...}
        r'\\usepackage\{[^}]+\}',    # \usepackage{...}
        r'\\usepackage\[[^\]]*\]\{[^}]+\}',  # \usepackage[options]{...}
        r'\\begin\{document\}',       # \begin{document}
        r'\\end\{document\}',         # \end{document}
    ]
    
    # Apply each pattern to clean the text
    cleaned_text = latex_text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text
    

def get_formula_in_markdown(text: str) -> str:
    """
    Formats a string containing a formula into a standard Markdown block.
    
    Args:
        text (str): The input string, potentially containing a formula.

    Returns:
        str: The formatted string, ready for Markdown rendering.
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if it's already enclosed in $$
    if text.startswith('$$') and text.endswith('$$'):
        text_new = text[2:-2].strip()
        if not '$' in text_new:
            return f"$$\n{text_new}\n$$"
        else:
            return text

    # Handle \[...\] format, convert to $$...$$
    if text.startswith('\\[') and text.endswith('\\]'):
        inner_content = text[2:-2].strip()
        return f"$$\n{inner_content}\n$$"
        
    # Check if it's enclosed in \[ \]
    if len(re.findall(r'.*\\\[.*\\\].*', text)) > 0:
        return text

    # Handle inline formulas ($...$)
    pattern = r'\$([^$]+)\$'
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        # It's an inline formula, return it as is
        return text  

    # If no LaTeX markdown syntax is present, return directly
    if not has_latex_markdown(text):  
        return text

    # Handle unnecessary LaTeX formatting like \usepackage
    if 'usepackage' in text:
        text = clean_latex_preamble(text)

    if text[0] == '`' and text[-1] == '`':
        text = text[1:-1]

    # Enclose the final text in a $$ block with newlines
    text = f"$$\n{text}\n$$"
    return text 


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace.
    
    Args:
        text: The original text.
        
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple consecutive whitespace characters with a single space
    if text[:2] == '`$' and text[-2:] == '$`':
        text = text[1:-1]
    
    return text


def layoutjson2md(image: Image.Image, cells: list, text_key: str = 'text', no_page_hf: bool = False,
                  image_dir: str = None, rel_prefix: str = "images", name: str = "img") -> str:
    """
    Converts a layout JSON format to Markdown.

    In the layout JSON, formulas are LaTeX, tables are HTML, and text is Markdown.

    Picture regions are written as SEPARATE image files (never inlined as base64)
    when ``image_dir`` is given: each crop is saved to ``image_dir`` and the
    markdown gets a relative link ``![](rel_prefix/<file>.png)``. This keeps the
    markdown small and token-cheap for downstream code agents; the preview renders
    the links with standard tooling. Without ``image_dir`` it falls back to the
    legacy base64 embed (for standalone/offline markdown).

    Args:
        image: A PIL Image object.
        cells: A list of dictionaries, each representing a layout cell.
        text_key: The key for the text field in the cell dictionary.
        no_page_hf: If True, skips page headers and footers.
        image_dir: Directory to save picture crops into (created if missing).
        rel_prefix: Path prefix used in the markdown links (relative to the .md).
        name: Filename stem for saved crops (e.g. the page save-name).

    Returns:
        str: The text in Markdown format.
    """
    text_items = []
    pic_idx = 0

    if image_dir is not None:
        os.makedirs(image_dir, exist_ok=True)

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = [int(coord) for coord in cell['bbox']]

        if no_page_hf and cell['category'] in ['Page-header', 'Page-footer']:
            continue

        if cell['category'] == 'Picture':
            # clamp to image bounds; skip degenerate boxes
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(image.width, x2), min(image.height, y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            image_crop = image.crop((x1c, y1c, x2c, y2c))
            if image_dir is not None:
                fname = f"{name}_pic_{pic_idx}.png"
                image_crop.save(os.path.join(image_dir, fname))
                link = f"{rel_prefix}/{fname}" if rel_prefix else fname
                text_items.append(f"![]({link})")
                pic_idx += 1
            else:
                text_items.append(f"![]({PILimage_to_base64(image_crop)})")
        elif cell['category'] == 'Formula':
            text_items.append(get_formula_in_markdown(cell.get(text_key, "")))
        else:
            text_items.append(f"{clean_text(cell.get(text_key, ''))}")

    markdown_text = '\n\n'.join(text_items)
    return markdown_text


def fix_streamlit_formulas(md: str) -> str:
    """
    Fixes the format of formulas in Markdown to ensure they display correctly in Streamlit.
    It adds a newline after the opening $$ and before the closing $$ if they don't already exist.
    
    Args:
        md_text (str): The Markdown text to fix.
        
    Returns:
        str: The fixed Markdown text.
    """
    
    # This inner function will be used by re.sub to perform the replacement
    def replace_formula(match):
        content = match.group(1)
        # If the content already has surrounding newlines, don't add more.
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]
        return f'$$\n{content}\n$$'
    
    # Use regex to find all $$....$$ patterns and replace them using the helper function.
    return re.sub(r'\$\$(.*?)\$\$', replace_formula, md, flags=re.DOTALL)
