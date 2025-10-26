# Truth Table Checker

Small Python tool to generate truth tables and check Boolean expressions against a reference truth table. Includes a CLI and a lightweight Tkinter GUI.

Features
- Parse boolean expressions with operators: NOT (~), AND (&), OR (|), XOR (^), IMPLIES (->), EQUIV (<->).
- Generate truth tables for expressions.
- Check equivalence between expressions.
- Load a CSV reference truth-table and check one or more output expressions against it (supports multiple outputs).
- GUI to load CSV, pick output column(s), enter mappings, normalize common operator characters, and view results.

Quick start
1. Ensure Python 3.8+ is installed.
2. Run from project directory:
   - CLI: python tt_checker.py [options]
   - GUI:   python tt_checker.py --gui

Reference CSV format
- First line: comma-separated headers (input variables and one or more outputs). Example:
  A0,B1,B0,S2,S1,S0
- Subsequent lines: rows of 0/1 values, same column count as header. Example:
  0,0,0,0,0,0
  1,1,1,1,0,0

Mapping syntax (check multiple outputs at once)
- Provide expressions as mappings in the form `OUT:EXPR` or `OUT=EXPR`.
- Separators: comma, semicolon, or newline.
- If you provide a default output (via UI or `-o`), you may omit the `OUT:` part for entries and they'll use the default.
- Examples:
  - Single output default: `-r ref.csv -o S2 -m "A0&B1&B0"`
  - Multiple mappings: `-r ref.csv -m "S2:A0&B1&B0, S1:A0 & ~B1"`
  - CLI paired lists: `-r ref.csv -e "expr1,expr2" -o "S2,S1"`

Operator normalization
- The tool accepts common variants. The GUI has "Normalize ops" to replace:
  `* -> &`, `+ -> |`, `! -> ~`, and common single quotes (', ’, ‘) -> `~`.
  - Please note that the placement of `~` is on the right side of the input instead of a left **when it should be placed on the right**.
- Normalization is applied early during parsing, so both GUI and CLI mapping inputs are normalized before evaluation.

CLI examples
- Show truth table:
  python tt_checker.py -e "A & (B | C)"
- Check equivalence:
  python tt_checker.py -c "A & B = B & A"
- Check expressions against reference CSV (default output is last column unless `-o` used):
  python tt_checker.py -r A1_0_ref.csv -o S2 -m "A0&B1&B0"
- Multiple mappings:
  python tt_checker.py -r ref.csv -m "S2:A0&B1&B0; S1:A0&~B1"

GUI
- Launch: python tt_checker.py --gui
- Actions:
  - Load reference CSV
  - Choose output column from dropdown (defaults to last header)
  - Paste mappings (one per line or comma-separated)
  - Click "Normalize ops" (optional) then "Match against reference"
  - Results and counterexamples appear in the Results area

Troubleshooting
- macOS may emit an ObjC runtime warning when opening the native file dialog:
  "The class 'NSOpenPanel' overrides the method identifier..."
  This is benign; the GUI suppresses that stderr message when opening the dialog.
- If expressions use variable names not present in the CSV's input columns you'll get an error listing unknown variables.
- Ensure CSV rows contain only 0 or 1 and match header column count.

Example file in repo
- /Users/samstep/Documents/projects/tt_checker/A1_0_ref.csv
  A0,B1,B0,S2,S1,S0
  0,0,0,0,0,0
  0,0,1,0,0,1
  0,1,0,0,1,0
  0,1,1,0,1,1
  1,0,0,0,0,1
  1,0,1,0,1,0
  1,1,0,0,1,1
  1,1,1,1,0,0

License
- MIT-style permissive use is recommended for personal projects. (Add an explicit LICENSE file if needed.)

Contact / Contributing
- Open issues or PRs in the repository with minimal reproducer steps. Keep expressions, CSVs, and CLI command lines for debugging.
