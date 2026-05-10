import os
import tokenize
import io
import re

def strip_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    
    try:
        tokens = list(tokenize.generate_tokens(io_obj.readline))
        for i, (toktype, ttext, (slineno, scol), (elineno, ecol), ltext) in enumerate(tokens):
            if slineno > last_lineno:
                last_col = 0
            if scol > last_col:
                out += " " * (scol - last_col)
                
            if toktype == tokenize.COMMENT:
                pass
            elif toktype == tokenize.STRING:
                # Check if it's a docstring
                # Look at the previous non-whitespace token
                is_docstring = False
                j = i - 1
                while j >= 0 and tokens[j][0] in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                    j -= 1
                
                if j < 0: # Module level docstring
                    is_docstring = True
                elif tokens[j][0] == tokenize.OP and tokens[j][1] == ':':
                    # Check if the colon is for a function/class or if/else
                    # Actually, docstrings only appear after ':' in class/def
                    # We'll assume if it's a standalone string after a colon, it's a docstring
                    is_docstring = True
                
                if is_docstring:
                    pass
                else:
                    # It's a regular string, but might contain CSS comments or emojis
                    processed_text = remove_css_comments(ttext)
                    out += processed_text
            else:
                out += ttext
            
            last_lineno = elineno
            last_col = ecol
    except Exception as e:
        print(f"Error: {e}")
        return source
    return out

def remove_css_comments(text):
    return re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U00010000-\U0010FFFF"
        "\u2600-\u27BF"
        "\u2b50\u231a"
        "🤖✅❌⚠️🎲📄💡🟦🟨🟥🔓"
        "]+", flags=re.UNICODE)
    text = re.sub(r'&#9888;', '', text)
    return emoji_pattern.sub('', text)

def process_file(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if filepath.endswith("app.py"):
        content = re.sub(r'if chosen == correct:\s+st\.balloons\(\)\s+else:', 'if chosen != correct:', content)
        content = re.sub(r'if chosen == correct:\s+st\.balloons\(\)', 'if chosen == correct:\n                pass', content)

    content = remove_emojis(content)
    content = strip_comments_and_docstrings(content)
    
    # Final cleanup of excessive blank lines
    lines = content.splitlines()
    clean_lines = []
    for line in lines:
        if line.strip():
            clean_lines.append(line)
        elif clean_lines and clean_lines[-1].strip():
            clean_lines.append("")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(clean_lines).strip() + "\n")

src_dir = r"c:\Users\Humayun\Desktop\uni\AI PROJECT\src"
for filename in os.listdir(src_dir):
    if filename.endswith(".py"):
        process_file(os.path.join(src_dir, filename))
