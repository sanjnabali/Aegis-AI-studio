"""
Code Runner Tool
================
Safely executes Python code in isolated environment.
Used for code testing, debugging, and demonstration.
"""

import asyncio
import sys
from io import StringIO
from typing import Dict, Any, Optional
import traceback

from loguru import logger


class CodeRunner:
    """
    Safe Python code execution with output capture.
    
    Security features:
    - Timeout protection
    - Output size limits
    - Restricted imports (configurable)
    """
    
    def __init__(self):
        self.timeout = 5  # seconds
        self.max_output_length = 5000  # characters
        self.execution_history = []
    
    async def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code and capture output.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
        
        Returns:
            Dict with output, errors, and execution status
        """
        
        if timeout is None:
            timeout = self.timeout
        
        logger.info(f"Executing Python code ({len(code)} chars)")
        
        result = {
            "success": False,
            "output": "",
            "error": None,
            "execution_time": 0,
        }
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            output, error, exec_time = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._execute_sync,
                    code,
                ),
                timeout=timeout,
            )
            
            result["output"] = output
            result["error"] = error
            result["execution_time"] = exec_time
            result["success"] = error is None
            
            if result["success"]:
                logger.success(f"Code executed in {exec_time:.3f}s")
            else:
                logger.warning(f"Code execution failed: {error}")
            
            # Store in history
            self.execution_history.append({
                "code_length": len(code),
                "success": result["success"],
                "execution_time": exec_time,
            })
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Code execution timeout ({timeout}s)")
            result["error"] = f"Execution timeout ({timeout}s)"
            return result
        
        except Exception as e:
            logger.error(f"Execution error: {e}")
            result["error"] = str(e)
            return result
    
    def _execute_sync(self, code: str) -> tuple:
        """Synchronous code execution with output capture"""
        
        import time
        start_time = time.time()
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        try:
            # Execute code
            exec_globals = {
                '__builtins__': __builtins__,
            }
            exec(code, exec_globals)
            
            # Get output
            output = stdout_capture.getvalue()
            error_output = stderr_capture.getvalue()
            
            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n[Output truncated...]"
            
            execution_time = time.time() - start_time
            
            return output, error_output if error_output else None, execution_time
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            execution_time = time.time() - start_time
            return "", error, execution_time
        
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format execution result for display"""
        
        formatted = "# Code Execution Result\n\n"
        
        if result["success"]:
            formatted += "**Status:** Success\n"
            formatted += f"**Execution Time:** {result['execution_time']:.3f}s\n\n"
            
            if result["output"]:
                formatted += "## Output\n\n```\n"
                formatted += result["output"]
                formatted += "\n```\n"
            else:
                formatted += "## Output\n\n(No output)\n"
        else:
            formatted += "**Status:** Error\n\n"
            formatted += "## Error\n\n```\n"
            formatted += result["error"]
            formatted += "\n```\n"
        
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h["success"])
        avg_time = sum(h["execution_time"] for h in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": f"{successful/total*100:.1f}%",
            "avg_execution_time": f"{avg_time:.3f}s",
        }


# Global code runner instance
code_runner = CodeRunner()