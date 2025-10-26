#!/usr/bin/env python3
import re
import sys
import argparse
from itertools import product
# New GUI imports
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import contextlib
import os

# Tokenizer
_TOKEN_RE = re.compile(r'''
    (?P<WS>\s+)
  | (?P<IMPL><->|->)
  | (?P<NOT>~|!|\bnot\b)
  | (?P<AND>&|\band\b)
  | (?P<OR>\||\bor\b)
  | (?P<XOR>\^|\bxor\b)
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<ID>[A-Za-z_][A-Za-z0-9_]*)
  ''', re.VERBOSE | re.IGNORECASE)

class Token:
    def __init__(self, typ, val):
        self.type = typ
        self.val = val
    def __repr__(self):
        return f"Token({self.type},{self.val})"

def tokenize(s):
    pos = 0
    tokens = []
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise SyntaxError(f"Unexpected character at {pos}: {s[pos]!r}")
        pos = m.end()
        typ = m.lastgroup
        if typ == 'WS':
            continue
        val = m.group(typ)
        tokens.append(Token(typ.upper(), val))
    tokens.append(Token('EOF', ''))
    return tokens

# AST nodes
class Node:
    def eval(self, env): raise NotImplementedError
    def vars(self): raise NotImplementedError

class Var(Node):
    def __init__(self, name): self.name = name
    def eval(self, env): return bool(env[self.name])
    def vars(self): return {self.name}
    def __repr__(self): return f"Var({self.name})"

class Not(Node):
    def __init__(self, child): self.child = child
    def eval(self, env): return not self.child.eval(env)
    def vars(self): return self.child.vars()
    def __repr__(self): return f"Not({self.child})"

class BinOp(Node):
    def __init__(self, left, right): self.left = left; self.right = right
    def vars(self): return self.left.vars() | self.right.vars()

class And(BinOp):
    def eval(self, env): return self.left.eval(env) and self.right.eval(env)
    def __repr__(self): return f"And({self.left},{self.right})"

class Or(BinOp):
    def eval(self, env): return self.left.eval(env) or self.right.eval(env)
    def __repr__(self): return f"Or({self.left},{self.right})"

class Xor(BinOp):
    def eval(self, env): return self.left.eval(env) ^ self.right.eval(env)
    def __repr__(self): return f"Xor({self.left},{self.right})"

class Impl(BinOp):
    def eval(self, env): return (not self.left.eval(env)) or self.right.eval(env)
    def __repr__(self): return f"Impl({self.left},{self.right})"

class Equiv(BinOp):
    def eval(self, env): return self.left.eval(env) == self.right.eval(env)
    def __repr__(self): return f"Equiv({self.left},{self.right})"

# Parser (recursive-descent)
class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0
    def cur(self): return self.toks[self.pos]
    def eat(self, typ=None):
        t = self.cur()
        if typ and t.type != typ:
            raise SyntaxError(f"Expected {typ} got {t.type}")
        self.pos += 1
        return t
    def parse(self):
        node = self.parse_equiv()
        if self.cur().type != 'EOF':
            raise SyntaxError("Extra input after expression")
        return node

    def parse_equiv(self):
        left = self.parse_impl()
        while self.cur().type == 'IMPL' and self.cur().val == '<->':
            self.eat('IMPL')
            right = self.parse_impl()
            left = Equiv(left, right)
        return left

    def parse_impl(self):
        left = self.parse_or()
        if self.cur().type == 'IMPL' and self.cur().val == '->':
            self.eat('IMPL')
            right = self.parse_impl()  # right-assoc
            left = Impl(left, right)
        return left

    def parse_or(self):
        left = self.parse_xor()
        while self.cur().type in ('OR',):
            self.eat('OR')
            right = self.parse_xor()
            left = Or(left, right)
        return left

    def parse_xor(self):
        left = self.parse_and()
        while self.cur().type in ('XOR',):
            self.eat('XOR')
            right = self.parse_and()
            left = Xor(left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.cur().type in ('AND',):
            self.eat('AND')
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self):
        if self.cur().type == 'NOT':
            self.eat('NOT')
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self):
        t = self.cur()
        if t.type == 'ID':
            self.eat('ID')
            return Var(t.val)
        if t.type == 'LPAREN':
            self.eat('LPAREN')
            node = self.parse_equiv()
            self.eat('RPAREN')
            return node
        raise SyntaxError(f"Unexpected token: {t}")

def parse_expr(s):
    tokens = tokenize(s)
    p = Parser(tokens)
    return p.parse()

# Utility functions
def sorted_vars(node):
    return sorted(node.vars(), key=lambda v: v.lower())

def gen_assignments(vars_list):
    for bits in product([False, True], repeat=len(vars_list)):
        yield dict(zip(vars_list, bits))

def truth_table(node):
    vars_list = sorted_vars(node)
    rows = []
    for env in gen_assignments(vars_list):
        rows.append((env, node.eval(env)))
    return vars_list, rows

def print_table(node, show_bool=False):
    vars_list, rows = truth_table(node)
    hdr = vars_list + ['Result']
    print(' | '.join(hdr))
    print('-' * (4 * len(hdr) + 3))
    for env, res in rows:
        vals = [str(int(env[v])) for v in vars_list] if not show_bool else [str(env[v]) for v in vars_list]
        vals.append(str(int(res)) if not show_bool else str(res))
        print(' | '.join(vals))

def check_equiv(expr_l, expr_r):
    n1 = parse_expr(expr_l)
    n2 = parse_expr(expr_r)
    vars_all = sorted(set(n1.vars()) | set(n2.vars()), key=lambda v: v.lower())
    counterexamples = []
    for env in gen_assignments(vars_all):
        if n1.eval(env) != n2.eval(env):
            counterexamples.append((env, n1.eval(env), n2.eval(env)))
    return counterexamples, vars_all

# New: normalize operator characters early (required by parse_mappings and GUI)
def normalize_ops(text: str) -> str:
    """
    Normalize common operator variants to the internal symbols:
      * -> &
      + -> |
      ! -> ~
      ASCII and Unicode single quotes -> ~
    Safe to call with None/empty string.
    """
    if not text:
        return text
    return (text
            .replace('*', '&')
            .replace('+', '|')
            .replace('!', '~')
            .replace("'", '~')   # ASCII apostrophe
            .replace("’", '~')   # Unicode right single quote U+2019
            .replace("‘", '~'))  # Unicode left single quote U+2018

# New: parse reference table into headers + list-of-row-dicts
def parse_ref_table_text(text):
    """
    Parse reference table text into headers and rows as dicts.
    Format:
      A,B,C,OUT
      0,0,0,1
    Returns: (headers_list, rows_list) where rows_list contains dicts {header: bool}
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    if not lines:
        raise ValueError("Empty reference table")
    headers = [h.strip() for h in lines[0].split(',')]
    if len(headers) < 2:
        raise ValueError("Reference table must have at least two columns (variables + output)")
    rows = []
    for i, ln in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) != len(headers):
            raise ValueError(f"Line {i}: expected {len(headers)} columns, got {len(parts)}")
        try:
            bits = [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"Line {i}: values must be 0 or 1")
        for b in bits:
            if b not in (0, 1):
                raise ValueError(f"Line {i}: values must be 0 or 1")
        row = dict(zip(headers, [bool(b) for b in bits]))
        rows.append(row)
    return headers, rows

def parse_ref_table_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return parse_ref_table_text(text)

def check_against_table(expr_text, headers, rows, out_name=None):
    """
    Evaluate expression against each row of the reference table.
    - headers: list of all column names
    - rows: list of dicts {col: bool}
    - out_name: chosen output column name (if None, use last header)
    Returns: (mismatches_list, input_vars)
      mismatches_list entries: (env_dict, expected_bool, actual_bool)
    """
    if out_name is None:
        out_name = headers[-1]
    if out_name not in headers:
        raise ValueError(f"Output column '{out_name}' not found in reference headers: {headers}")
    input_vars = [h for h in headers if h != out_name]
    node = parse_expr(expr_text)
    expr_vars = set(node.vars())
    unknown = expr_vars - set(input_vars)
    if unknown:
        raise ValueError(f"Expression uses unknown variable(s) not present in reference table inputs: {', '.join(sorted(unknown))}")
    mismatches = []
    for raw in rows:
        env = {v: raw[v] for v in input_vars}
        expected = raw[out_name]
        actual = node.eval(env)
        if actual != expected:
            mismatches.append((env, expected, actual))
    return mismatches, input_vars

# New: parse mappings like "S2: A0&B1&B0" or "S2=A0&B1&B0"
def parse_mappings(raw_text, default_out=None):
    """
    Parse mapping text into list of (out_name, expr) pairs.
    Accept separators: comma, semicolon, newline.
    Mapping forms:
      OUT:EXPR
      OUT=EXPR
    If an item has no OUT and default_out is provided, use default_out.
    """
    # normalize upfront so mapping parsing sees consistent operators
    raw_text = normalize_ops(raw_text)
    if not raw_text or not raw_text.strip():
        return []
    parts = [p.strip() for p in re.split(r'[,\n;]+', raw_text) if p.strip()]
    mappings = []
    for p in parts:
        if ':' in p:
            out, expr = p.split(':', 1)
        elif '=' in p:
            # prefer mapping with '=' only if it's not a variable assignment inside expression; assume mapping.
            out, expr = p.split('=', 1)
        else:
            if default_out is None:
                raise ValueError(f"No output specified for expression '{p}' and no default output available")
            out, expr = default_out, p
        out = out.strip()
        expr = expr.strip()
        if not out or not expr:
            raise ValueError(f"Invalid mapping entry: '{p}'")
        mappings.append((out, expr))
    return mappings

def check_mappings_against_table(mappings, headers, rows):
    """
    mappings: list of (out_name, expr)
    returns: list of result tuples (out_name, expr, mismatches, input_vars) 
    where mismatches is list as from check_against_table
    """
    results = []
    for out_name, expr in mappings:
        mismatches, input_vars = check_against_table(expr, headers, rows, out_name=out_name)
        results.append((out_name, expr, mismatches, input_vars))
    return results

# New: GUI
def launch_gui():
    root = tk.Tk()
    root.title("Truth-table / Reference Checker")

    # State
    state = {'ref_path': None, 'headers': [], 'rows': []}

    # Top frame: load file, output column
    top = tk.Frame(root)
    top.pack(fill='x', padx=8, pady=6)

    def load_file():
        # suppress macOS ObjC runtime warning emitted to stderr by the native file dialog
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                path = filedialog.askopenfilename(title="Open reference CSV", filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path:
            return
        try:
            headers, rows = parse_ref_table_file(path)
        except Exception as e:
            messagebox.showerror("Parse error", str(e))
            return
        state['ref_path'] = path
        state['headers'] = headers
        state['rows'] = rows
        out_var.set(headers[-1])
        out_menu['menu'].delete(0, 'end')
        for h in headers:
            out_menu['menu'].add_command(label=h, command=tk._setit(out_var, h))
        status_text.set(f"Loaded {path} ({len(rows)} rows).")
        # populate hint of input vars
        inputs = [h for h in headers if h != out_var.get()]
        vars_label.config(text="Inputs: " + ", ".join(inputs))

    load_btn = tk.Button(top, text="Load reference CSV", command=load_file)
    load_btn.pack(side='left')

    out_var = tk.StringVar(value="(none)")
    out_menu = tk.OptionMenu(top, out_var, ())
    out_menu.pack(side='left', padx=8)
    tk.Label(top, text="Output column:").pack(side='left')

    status_text = tk.StringVar(value="No reference loaded")
    status = tk.Label(root, textvariable=status_text, anchor='w')
    status.pack(fill='x', padx=8)

    vars_label = tk.Label(root, text="Inputs: (none)", anchor='w')
    vars_label.pack(fill='x', padx=8)

    # Middle: expressions input
    mid = tk.Frame(root)
    mid.pack(fill='both', expand=True, padx=8, pady=6)
    tk.Label(mid, text="Expressions (comma or newline separated):").pack(anchor='w')
    expr_box = scrolledtext.ScrolledText(mid, height=6)
    expr_box.pack(fill='both', expand=True)

    # Buttons
    btns = tk.Frame(root)
    btns.pack(fill='x', padx=8, pady=6)

    # New: normalize operators button handler
    def replace_ops():
        txt = expr_box.get("1.0", 'end')
        if not txt.strip():
            return
        new = normalize_ops(txt)
        expr_box.delete("1.0", 'end')
        expr_box.insert('end', new)
        status_text.set("Normalized operators (*,+,!,',’ ) to &,|,~ in expressions")

    def run_match():
        if not state['rows']:
            messagebox.showwarning("No reference", "Load a reference CSV first.")
            return
        out_name_default = out_var.get() if out_var.get() != "(none)" else None
        raw = expr_box.get("1.0", 'end').strip()
        if not raw:
            messagebox.showinfo("No expressions", "Enter one or more expressions.")
            return
        try:
            mappings = parse_mappings(raw, default_out=out_name_default)
        except Exception as e:
            messagebox.showerror("Parse error", str(e))
            return
        out_lines = []
        results = check_mappings_against_table(mappings, state['headers'], state['rows'])
        for out_name, expr, mismatches, input_vars in results:
            if not mismatches:
                out_lines.append(f"OK: '{expr}' matches column '{out_name}' (inputs: {', '.join(input_vars)})")
            else:
                out_lines.append(f"MISMATCH: '{expr}' does NOT match '{out_name}' (showing up to 10):")
                for env, expected, actual in mismatches[:10]:
                    row = ' '.join(f"{k}={int(v)}" for k, v in env.items())
                    out_lines.append(f"  {row} -> expected={int(expected)} got={int(actual)}")
                if len(mismatches) > 10:
                    out_lines.append(f"  ... and {len(mismatches)-10} more mismatches")
        result_box.configure(state='normal')
        result_box.delete("1.0", 'end')
        result_box.insert('end', "\n".join(out_lines))
        result_box.configure(state='disabled')

    def show_truth_table():
        raw = expr_box.get("1.0", 'end').strip()
        if not raw:
            messagebox.showinfo("No expression", "Enter an expression to show its truth table.")
            return
        expr = raw.splitlines()[0].split(',')[0].strip()
        # normalize before parsing
        expr = normalize_ops(expr)
        try:
            node = parse_expr(expr)
        except Exception as e:
            messagebox.showerror("Parse error", str(e))
            return
        vars_list, rows = truth_table(node)
        lines = []
        hdr = vars_list + ['Result']
        lines.append(' | '.join(hdr))
        lines.append('-' * (4 * len(hdr) + 3))
        for env, res in rows:
            vals = [str(int(env[v])) for v in vars_list]
            vals.append(str(int(res)))
            lines.append(' | '.join(vals))
        result_box.configure(state='normal')
        result_box.delete("1.0", 'end')
        result_box.insert('end', "\n".join(lines))
        result_box.configure(state='disabled')

    # New button for operator normalization
    replace_btn = tk.Button(btns, text="Normalize ops (*+!)", command=replace_ops)
    replace_btn.pack(side='left', padx=6)

    match_btn = tk.Button(btns, text="Match against reference", command=run_match)
    match_btn.pack(side='left')
    tt_btn = tk.Button(btns, text="Show truth table (first expr)", command=show_truth_table)
    tt_btn.pack(side='left', padx=6)

    # Results area
    tk.Label(root, text="Results:").pack(anchor='w', padx=8)
    result_box = scrolledtext.ScrolledText(root, height=12, state='disabled')
    result_box.pack(fill='both', expand=True, padx=8, pady=(0,8))

    root.mainloop()

# CLI
def repl():
    print("Truth table / equivalence checker. Type 'quit' to exit.")
    while True:
        try:
            s = input("tt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not s:
            continue
        if s.lower() in ('quit','exit'):
            return
        if '=' in s:
            left, right = s.split('=', 1)
            try:
                ce, vars_all = check_equiv(left.strip(), right.strip())
            except Exception as e:
                print("Parse error:", e)
                continue
            if not ce:
                print("Equivalent for all assignments over variables:", ', '.join(vars_all) or "(none)")
            else:
                print("Not equivalent. Counterexample(s):")
                for env, a, b in ce[:5]:
                    row = ' '.join(f"{k}={int(v)}" for k,v in env.items())
                    print(f" {row}  -> {int(a)} != {int(b)}")
        else:
            try:
                node = parse_expr(s)
            except Exception as e:
                print("Parse error:", e)
                continue
            print_table(node)

def main(argv):
    ap = argparse.ArgumentParser(description="Truth-table generator and equation checker")
    ap.add_argument('-e','--expr', help="Expression to show truth table for (e.g. 'A & (B | C)')", metavar='"EXPR"')
    ap.add_argument('-c','--check', help="Check equivalence of equations. Use commas to separate multiple like \"A&B = A, A|B = B|A\"", metavar='"EQ1 = EQ2, ..."' )
    ap.add_argument('-i','--interactive', action='store_true', help="Start interactive prompt")
    # Ref args (if not already present)
    ap.add_argument('-r','--ref-file', dest='ref_file', help="Path to reference truth-table CSV (first line headers; last column is output).")
    ap.add_argument('-R','--ref-string', dest='ref_string', help="Reference truth-table as CSV string (same format as file).")
    ap.add_argument('-m','--match', dest='match', help="Comma-separated expressions to test against the reference table (use with -r or -R).")
    ap.add_argument('-o','--out', dest='out_col', help="Name of the output column in the reference CSV to test (default: last column).")
    # New GUI flag
    ap.add_argument('--gui', action='store_true', help="Launch graphical user interface")
    args = ap.parse_args(argv)

    if args.gui:
        launch_gui()
        return 0

    # If a reference table is provided, evaluate provided expressions against it.
    if args.ref_file or args.ref_string:
        if args.ref_file and args.ref_string:
            print("Specify only one of --ref-file or --ref-string", file=sys.stderr)
            return 2
        try:
            if args.ref_file:
                headers, raw_rows = parse_ref_table_file(args.ref_file)
            else:
                headers, raw_rows = parse_ref_table_text(args.ref_string)
        except Exception as e:
            print("Failed to parse reference table:", e, file=sys.stderr)
            return 2

        # If multiple potential output columns exist and no --out was provided, warn and use last header by default.
        if args.out_col is None and len(headers) > 2:
            # It's possible there are many columns; default to last but inform user.
            print(f"No --out specified; using last column '{headers[-1]}' as expected output. If this is incorrect, rerun with --out NAME", file=sys.stderr)

        out_name = args.out_col or headers[-1]

        # build mappings
        mappings = []
        try:
            if args.match:
                mappings = parse_mappings(args.match, default_out=out_name)
            elif args.expr:
                # support pairing comma-separated lists: -e "expr1,expr2" -o "OUT1,OUT2"
                if ',' in args.expr and args.out_col and ',' in args.out_col:
                    exprs = [s.strip() for s in args.expr.split(',') if s.strip()]
                    outs = [s.strip() for s in args.out_col.split(',') if s.strip()]
                    if len(exprs) != len(outs):
                        print("When providing comma-separated -e and -o lists they must have the same length", file=sys.stderr)
                        return 2
                    mappings = list(zip(outs, exprs))
                else:
                    mappings = [(out_name, args.expr.strip())]
            else:
                print("No expressions provided to test against the reference. Use -m or -e.", file=sys.stderr)
                return 2
        except Exception as e:
            print("Failed to parse mappings:", e, file=sys.stderr)
            return 2

        results = check_mappings_against_table(mappings, headers, raw_rows)
        for out_name, expr, mismatches, input_vars in results:
            if not mismatches:
                print(f"OK: '{expr}' matches the reference column '{out_name}' (input vars: {', '.join(input_vars)})")
            else:
                print(f"MISMATCH: '{expr}' does NOT match reference column '{out_name}' (showing up to 10 counterexamples):")
                for env, expected, actual in mismatches[:10]:
                    row = ' '.join(f"{k}={int(v)}" for k, v in env.items())
                    print(f"  {row} -> expected={int(expected)} got={int(actual)}")
                if len(mismatches) > 10:
                    print(f"  ... and {len(mismatches)-10} more mismatches")
        return 0

    if args.expr:
        try:
            expr_norm = normalize_ops(args.expr)
            node = parse_expr(expr_norm)
        except Exception as e:
            print("Parse error:", e)
            return 1
        print_table(node)
        return 0

    if args.check:
        eqs = [s.strip() for s in args.check.split(',') if s.strip()]
        for eq in eqs:
            if '=' not in eq:
                print(f"Skipping invalid equation (no '='): {eq}")
                continue
            left, right = eq.split('=',1)
            try:
                ce, vars_all = check_equiv(left.strip(), right.strip())
            except Exception as e:
                print(f"Parse error in '{eq}': {e}")
                continue
            if not ce:
                print(f"OK: {left.strip()} == {right.strip()}  (vars: {', '.join(vars_all) or '(none)'})")
            else:
                print(f"NOT EQUIVALENT: {left.strip()} != {right.strip()}")
                for env, a, b in ce[:5]:
                    row = ' '.join(f"{k}={int(v)}" for k,v in env.items())
                    print(f"  {row} -> {int(a)} != {int(b)}")
                if len(ce) > 5:
                    print(f"  ... and {len(ce)-5} more counterexamples")
        return 0

    if args.interactive or (not args.expr and not args.check):
        repl()
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
