#!/usr/bin/env python3
"""
Emoji Calculator — Dark Theme (Tkinter)
Fixed: number buttons now show emoji keycaps (0️⃣..9️⃣) and do not repeat.
"""
import ast
import operator
import tkinter as tk
from tkinter import font as tkfont

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        return super().visit(node)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        return ALLOWED_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unary operator {op_type} not allowed")
        return ALLOWED_OPERATORS[op_type](operand)

    def visit_Num(self, node):
        return node.n

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only int/float constants are allowed")

    def visit_Expr(self, node):
        return self.visit(node.value)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

def safe_eval(expr: str):
    try:
        parsed = ast.parse(expr, mode="eval")
        return SafeEval().visit(parsed)
    except Exception as e:
        raise ValueError("Invalid expression") from e

class EmojiCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔮 EmojiCalc — Dark Theme")
        self.configure(bg="#0b1220")
        self.resizable(False, False)

        # fonts
        self.large_font = tkfont.Font(family="Inter", size=22, weight="bold")
        self.small_font = tkfont.Font(family="Inter", size=12)
        # emoji-capable font
        self.emoji_font = tkfont.Font(family="Segoe UI Emoji", size=16)

        # state
        self.expr = ""

        # build UI
        self._build_screen()
        self._build_pad()

        # keyboard bindings
        self.bind("<Return>", lambda e: self.evaluate())
        self.bind("<BackSpace>", lambda e: self.backspace())
        self.bind("<Escape>", lambda e: self.clear())
        self.bind(".", lambda e: self.insert_value('.'))
        for k in "0123456789+-*/()%":
            self.bind(k, lambda e, ch=k: self.insert_value(ch))

    def _build_screen(self):
        screen = tk.Frame(self, bg="#071021", bd=0, padx=12, pady=12)
        screen.grid(row=0, column=0, padx=18, pady=(18, 8), sticky="ew")

        self.expr_label = tk.Label(
            screen,
            text="0",
            anchor="e",
            bg="#071021",
            fg="#94a3b8",
            font=self.small_font,
            wraplength=320,
        )
        self.expr_label.pack(fill="x")

        self.result_label = tk.Label(
            screen,
            text="0",
            anchor="e",
            bg="#071021",
            fg="#e6eef7",
            font=self.large_font,
        )
        self.result_label.pack(fill="x", pady=(6, 0))

    def _build_pad(self):
        pad = tk.Frame(self, bg="#0b1220")
        pad.grid(row=1, column=0, padx=18, pady=(0, 18))

        btn_cfg = {
            "width": 6,
            "height": 2,
            "bd": 0,
            "relief": "flat",
            "font": self.emoji_font,
            "fg": "#eef3f8",
            "bg": "#0f1720",
            "activebackground": "#16323b",
            "activeforeground": "#ffffff",
        }

        # use emoji keycaps for digits (0️⃣..9️⃣)
        buttons = [
            ("🧽\nClear", "C", self.clear, "#991b1b"),
            ("⌫\nBack", "BK", self.backspace, "#374151"),
            ("( )\nParen", "()", lambda: self.insert_value("()"), "#374151"),
            ("➗\nDiv", "/", lambda: self.insert_value("/"), "#1e40af"),

            ("7️⃣\n7", "7", lambda: self.insert_value("7"), None),
            ("8️⃣\n8", "8", lambda: self.insert_value("8"), None),
            ("9️⃣\n9", "9", lambda: self.insert_value("9"), None),
            ("✖️\nMul", "*", lambda: self.insert_value("*"), "#1e40af"),

            ("4️⃣\n4", "4", lambda: self.insert_value("4"), None),
            ("5️⃣\n5", "5", lambda: self.insert_value("5"), None),
            ("6️⃣\n6", "6", lambda: self.insert_value("6"), None),
            ("➖\nSub", "-", lambda: self.insert_value("-"), "#1e40af"),

            ("1️⃣\n1", "1", lambda: self.insert_value("1"), None),
            ("2️⃣\n2", "2", lambda: self.insert_value("2"), None),
            ("3️⃣\n3", "3", lambda: self.insert_value("3"), None),
            ("➕\nAdd", "+", lambda: self.insert_value("+"), "#1e40af"),

            ("0️⃣\n0", "0", lambda: self.insert_value("0"), None),
            ("🔸\n.", ".", lambda: self.insert_value("."), None),
            ("%\nMod", "%", lambda: self.insert_value("%"), None),
            ("✅\n=", "=", self.evaluate, "#059669"),
        ]

        r = 0
        c = 0
        for (label, _key, cmd, bg) in buttons:
            b = tk.Button(pad, text=label, command=cmd, **btn_cfg)
            if bg:
                b.configure(bg=bg)
            b.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

            b.bind("<Enter>", lambda e, t=label: e.widget.configure(relief="raised"))
            b.bind("<Leave>", lambda e: e.widget.configure(relief="flat"))

            c += 1
            if c > 3:
                c = 0
                r += 1

        for i in range(4):
            pad.grid_columnconfigure(i, weight=1)

    def insert_value(self, val: str):
        if val == "()":
            # insert both parentheses; user can type between or use backspace
            self.expr += "()"
        else:
            self.expr += val
        self._refresh()

    def clear(self):
        self.expr = ""
        self._refresh()

    def backspace(self):
        self.expr = self.expr[:-1]
        self._refresh()

    def evaluate(self):
        if not self.expr.strip():
            return
        try:
            safe_expr = self.expr.replace("×", "*").replace("÷", "/")
            result = safe_eval(safe_expr)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            self.result_label.config(text=str(result))
        except Exception:
            self.result_label.config(text="Error")

    def _refresh(self):
        display_expr = self.expr if self.expr else "0"
        self.expr_label.config(text=display_expr)

if __name__ == "__main__":
    app = EmojiCalculator()
    app.mainloop()
