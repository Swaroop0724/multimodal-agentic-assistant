"""
tools/calculator_tool.py
Safe mathematical expression evaluator using Python's AST module.
No eval() — fully safe from code injection.
Handles: arithmetic, trig, logs, compound interest, statistics.
"""

from langchain.tools import tool
from loguru import logger
import ast
import math
import operator
import statistics


# ── Whitelist of safe operations ────────────────────────────────────────────────
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# ── Whitelist of safe math functions ────────────────────────────────────────────
SAFE_FUNCTIONS = {
    # Basic
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    # Math module
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    # Trig
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Statistics
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "variance": statistics.variance,
    # Constants
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Num):           # Python < 3.8
        return node.n
    elif isinstance(node, ast.Constant):    # Python >= 3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function not allowed: {func_name}")
        args = [_safe_eval(a) for a in node.args]
        return SAFE_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    elif isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_safe_eval(e) for e in node.elts)
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _evaluate_expression(expr: str) -> float:
    """Parse and safely evaluate a math expression string."""
    tree = ast.parse(expr.strip(), mode="eval")
    return _safe_eval(tree.body)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions safely.
    Use this for: arithmetic, algebra, compound interest, statistics,
    unit conversions, geometry, or any numerical calculation.

    Supports: +, -, *, /, **, %, sqrt(), log(), sin(), cos(), tan(),
              mean(), stdev(), round(), abs(), pi, e, and more.

    Examples:
        "10000 * (1 + 0.07) ** 10"          → compound interest
        "sqrt(9**2 + 12**2)"                 → Pythagorean theorem
        "stdev([4, 8, 15, 16, 23, 42])"      → standard deviation
        "(98.6 - 32) * 5/9"                  → Fahrenheit to Celsius

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        The computed result as a formatted string
    """
    try:
        logger.debug(f"[calculator] expression='{expression}'")

        result = _evaluate_expression(expression)

        # Format nicely
        if isinstance(result, float):
            # Round to 6 significant figures
            if result == int(result) and abs(result) < 1e12:
                formatted = str(int(result))
            else:
                formatted = f"{result:.6g}"
        else:
            formatted = str(result)

        output = f"Result: {formatted}\nExpression: {expression}"
        logger.debug(f"[calculator] result={formatted}")
        return output

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid expression — {e}"
    except Exception as e:
        logger.error(f"[calculator] error: {e}")
        return f"Calculation failed: {str(e)}"
