"""
Comprehensive debugging and timing utilities for Fashion Bot.
Provides detailed logging, timing, and error tracking for all operations.
"""
import os
import time
import functools
import json
from typing import Any, Callable, Optional, Dict
from datetime import datetime
from contextlib import contextmanager
import asyncio

# Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
LOG_TIMINGS = os.getenv("LOG_TIMINGS", "true").lower() == "true"
LOG_TOOL_CALLS = os.getenv("LOG_TOOL_CALLS", "true").lower() == "true"
LOG_API_CALLS = os.getenv("LOG_API_CALLS", "true").lower() == "true"
SLOW_THRESHOLD_MS = float(os.getenv("SLOW_THRESHOLD_MS", "1000"))


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_debug(message: str, level: str = "INFO", **kwargs):
    """Enhanced logging with colors and context."""
    if not DEBUG_MODE:
        return
    
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    color_map = {
        "INFO": Colors.OKBLUE,
        "SUCCESS": Colors.OKGREEN,
        "WARNING": Colors.WARNING,
        "ERROR": Colors.FAIL,
        "TIMING": Colors.OKCYAN,
        "API": Colors.HEADER,
    }
    
    color = color_map.get(level, Colors.ENDC)
    
    # Format the message
    log_line = f"{color}[{timestamp}] [{level}]{Colors.ENDC} {message}"
    
    # Add extra context if provided
    if kwargs:
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        log_line += f" {Colors.BOLD}({context}){Colors.ENDC}"
    
    print(log_line, flush=True)


@contextmanager
def timer(operation_name: str, log_slow_only: bool = False):
    """Context manager for timing operations."""
    start = time.perf_counter()
    
    if LOG_TIMINGS and not log_slow_only:
        log_debug(f"⏱️  Starting: {operation_name}", level="TIMING")
    
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        if LOG_TIMINGS:
            if log_slow_only and elapsed_ms < SLOW_THRESHOLD_MS:
                return
                
            emoji = "🐌" if elapsed_ms > SLOW_THRESHOLD_MS else "⚡"
            level = "WARNING" if elapsed_ms > SLOW_THRESHOLD_MS else "TIMING"
            
            log_debug(
                f"{emoji} Completed: {operation_name}",
                level=level,
                time_ms=f"{elapsed_ms:.2f}"
            )


def debug_tool(func: Callable) -> Callable:
    """Decorator for Parlant tools to add detailed logging and timing."""
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tool_name = func.__name__
        
        if LOG_TOOL_CALLS:
            # Log tool invocation
            log_debug(
                f"🔧 Tool Called: {tool_name}",
                level="INFO",
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            # Log key parameters
            if kwargs:
                for key, value in kwargs.items():
                    if key == "context":
                        continue
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    log_debug(f"   └─ {key}: {value_str}", level="INFO")
        
        start = time.perf_counter()
        error = None
        result = None
        
        try:
            result = await func(*args, **kwargs)
            
        except Exception as e:
            error = e
            log_debug(
                f"❌ Tool Error: {tool_name}",
                level="ERROR",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
            
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if LOG_TOOL_CALLS:
                if error:
                    status = "❌ FAILED"
                    level = "ERROR"
                else:
                    status = "✅ SUCCESS"
                    level = "SUCCESS" if elapsed_ms < SLOW_THRESHOLD_MS else "WARNING"
                
                log_debug(
                    f"{status}: {tool_name}",
                    level=level,
                    time_ms=f"{elapsed_ms:.2f}"
                )
                
                # Log result summary
                if result and hasattr(result, 'data'):
                    data_keys = list(result.data.keys()) if isinstance(result.data, dict) else []
                    if data_keys:
                        log_debug(f"   └─ Returned keys: {', '.join(data_keys[:5])}", level="INFO")
        
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # For synchronous functions
        tool_name = func.__name__
        
        if LOG_TOOL_CALLS:
            log_debug(f"🔧 Tool Called: {tool_name}", level="INFO")
        
        start = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if LOG_TOOL_CALLS:
                log_debug(
                    f"✅ SUCCESS: {tool_name}",
                    level="SUCCESS",
                    time_ms=f"{elapsed_ms:.2f}"
                )
            
            return result
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log_debug(
                f"❌ Tool Error: {tool_name}",
                level="ERROR",
                error=str(e),
                time_ms=f"{elapsed_ms:.2f}"
            )
            raise
    
    # Return appropriate wrapper based on whether function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def debug_api_call(api_name: str):
    """Decorator for external API calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if LOG_API_CALLS:
                log_debug(f"🌐 API Call: {api_name}", level="API")
            
            start = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                if LOG_API_CALLS:
                    log_debug(
                        f"✅ API Success: {api_name}",
                        level="SUCCESS",
                        time_ms=f"{elapsed_ms:.2f}"
                    )
                
                return result
                
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log_debug(
                    f"❌ API Error: {api_name}",
                    level="ERROR",
                    error=str(e),
                    error_type=type(e).__name__,
                    time_ms=f"{elapsed_ms:.2f}"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if LOG_API_CALLS:
                log_debug(f"🌐 API Call: {api_name}", level="API")
            
            start = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                if LOG_API_CALLS:
                    log_debug(
                        f"✅ API Success: {api_name}",
                        level="SUCCESS",
                        time_ms=f"{elapsed_ms:.2f}"
                    )
                
                return result
                
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log_debug(
                    f"❌ API Error: {api_name}",
                    level="ERROR",
                    error=str(e),
                    error_type=type(e).__name__,
                    time_ms=f"{elapsed_ms:.2f}"
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class PerformanceTracker:
    """Track performance metrics across the application."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.call_counts: Dict[str, int] = {}
    
    def record(self, operation: str, duration_ms: float):
        """Record a timing metric."""
        if operation not in self.metrics:
            self.metrics[operation] = []
            self.call_counts[operation] = 0
        
        self.metrics[operation].append(duration_ms)
        self.call_counts[operation] += 1
    
    def get_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an operation."""
        if operation not in self.metrics:
            return None
        
        timings = self.metrics[operation]
        return {
            "count": self.call_counts[operation],
            "avg_ms": sum(timings) / len(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "total_ms": sum(timings),
        }
    
    def print_summary(self):
        """Print performance summary."""
        if not DEBUG_MODE or not self.metrics:
            return
        
        log_debug("=" * 80, level="INFO")
        log_debug("📊 PERFORMANCE SUMMARY", level="INFO")
        log_debug("=" * 80, level="INFO")
        
        for operation in sorted(self.metrics.keys()):
            stats = self.get_stats(operation)
            log_debug(
                f"{operation}:",
                level="INFO",
                calls=stats["count"],
                avg=f"{stats['avg_ms']:.2f}ms",
                min=f"{stats['min_ms']:.2f}ms",
                max=f"{stats['max_ms']:.2f}ms",
                total=f"{stats['total_ms']:.2f}ms"
            )
        
        log_debug("=" * 80, level="INFO")


# Global performance tracker
perf_tracker = PerformanceTracker()


def log_function_entry(func_name: str, **params):
    """Log function entry with parameters."""
    if DEBUG_MODE:
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        log_debug(f"→ Entering: {func_name}({params_str})", level="INFO")


def log_function_exit(func_name: str, result_summary: str = ""):
    """Log function exit."""
    if DEBUG_MODE:
        log_debug(f"← Exiting: {func_name} {result_summary}", level="INFO")


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with detailed error logging."""
    try:
        # Try direct parse
        return json.loads(text)
    except json.JSONDecodeError as e:
        log_debug(f"JSON parse attempt 1 failed: {e}", level="WARNING")
        
        # Try cleaning markdown
        try:
            cleaned = text.strip()
            prefixes = ("```json", "```jsonc", "```")
            for pref in prefixes:
                if cleaned.startswith(pref):
                    cleaned = cleaned[len(pref):]
                    break
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            
            return json.loads(cleaned)
        except json.JSONDecodeError as e2:
            log_debug(f"JSON parse attempt 2 failed: {e2}", level="ERROR")
            log_debug(f"Original text: {text[:200]}...", level="ERROR")
            
            if default is not None:
                log_debug(f"Returning default value", level="WARNING")
                return default
            raise


# Export all utilities
__all__ = [
    "log_debug",
    "timer",
    "debug_tool",
    "debug_api_call",
    "Colors",
    "perf_tracker",
    "log_function_entry",
    "log_function_exit",
    "safe_json_loads",
]