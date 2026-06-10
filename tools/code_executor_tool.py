"""
tools/code_executor_tool.py
E2B-powered sandboxed Python code execution.
Runs code in an isolated cloud sandbox — safe, no local execution.
"""

from langchain.tools import tool
from e2b_code_interpreter import Sandbox
from config.settings import settings
from loguru import logger
import re


def _extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks or plain text."""
    # Match ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()
    # If no code block markers, treat entire input as code
    return text.strip()


@tool
def code_executor(code: str) -> str:
    """
    Execute Python code in a secure cloud sandbox and return the output.
    Use this for: running calculations, testing algorithms, data processing,
    generating plots (returns description), or verifying code logic.

    Args:
        code: Python code to execute (plain code or markdown code block)

    Returns:
        stdout output, stderr if any errors, and execution status
    """
    try:
        clean_code = _extract_code(code)
        logger.debug(f"[code_executor] executing {len(clean_code)} chars of code")

        with Sandbox(api_key=settings.e2b_api_key) as sandbox:
            execution = sandbox.run_code(clean_code)

            # ── Collect outputs ─────────────────────────────────────────────────
            output_parts = []

            # stdout logs
            stdout_logs = [
                log.line for log in execution.logs.stdout
                if log.line.strip()
            ]
            if stdout_logs:
                output_parts.append("Output:\n" + "\n".join(stdout_logs))

            # stderr logs
            stderr_logs = [
                log.line for log in execution.logs.stderr
                if log.line.strip()
            ]
            if stderr_logs:
                output_parts.append("Stderr:\n" + "\n".join(stderr_logs))

            # Execution error
            if execution.error:
                output_parts.append(
                    f"Error: {execution.error.name}: {execution.error.value}"
                )
                if execution.error.traceback:
                    # Only last 3 lines of traceback to keep it concise
                    tb_lines = execution.error.traceback.strip().split("\n")
                    output_parts.append("Traceback (last 3 lines):\n" + "\n".join(tb_lines[-3:]))

            # Results (return values, display objects)
            if execution.results:
                for result in execution.results:
                    if hasattr(result, "text") and result.text:
                        output_parts.append(f"Result: {result.text}")
                    elif hasattr(result, "png") and result.png:
                        output_parts.append("Result: [Plot/Image generated successfully]")

            if not output_parts:
                output_parts.append("Code executed successfully with no output.")

            final_output = "\n\n".join(output_parts)
            logger.debug(f"[code_executor] execution complete | output_len={len(final_output)}")
            return final_output

    except Exception as e:
        logger.error(f"[code_executor] error: {e}")
        return f"Code execution failed: {str(e)}"
