import os
import re
import requests
import subprocess
import json
import time
import sys
import yaml
from pathlib import Path
from markdownify import markdownify
from requests.exceptions import RequestException
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from predefined_tools import *
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    OpenAIServerModel,
    GradioUI,
    MCPClient,
    tool,
)
from mcp import StdioServerParameters

# Environment variables management
from dotenv import load_dotenv

# Optimization imports
from functools import lru_cache, wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
from collections import deque

# Load environment variables from .env file
load_dotenv()

# Configuration constants
DEFAULT_MAX_STEPS = 50  # Increased from 20 for complex biomedical tasks

# Mem0 integration for enhanced memory management
try:
    from mem0 import Memory, MemoryClient
    MEM0_AVAILABLE = True
    print("✅ Mem0 library available - enhanced memory features enabled")
except ImportError:
    MEM0_AVAILABLE = False
    print("⚠️ Mem0 library not installed - using traditional knowledge base")
    print("💡 Install with: pip install mem0ai")

# === API Keys Management ===
def get_api_key(key_name: str, required: bool = True) -> str:
    """Get API key from environment variables with proper error handling."""
    api_key = os.getenv(key_name)
    
    if required and not api_key:
        print(f"❌ Missing required API key: {key_name}")
        print(f"💡 Please set {key_name} in your .env file")
        print(f"📋 Copy .env.example to .env and fill in your API keys")
        if key_name == "OPENROUTER_API_KEY":
            print(f"🔗 Get your OpenRouter API key at: https://openrouter.ai/")
            sys.exit(1)
    elif not api_key:
        print(f"⚠️ Optional API key not set: {key_name}")
        return ""
    
    return api_key

# Load API keys from environment variables
OPENROUTER_API_KEY_STRING = get_api_key("OPENROUTER_API_KEY", required=True)
MEM0_API_KEY = get_api_key("MEM0_API_KEY", required=False)

HTTP_REFERER_URL = "http://localhost:8000"  # Replace if you have a specific site
X_TITLE_APP_NAME = "My Smolagent Web Search System" # Replace with your app name
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"


# --- Import Knowledge Base System (legacy, kept for migration) ---
try:
    from Knowledge_base import KnowledgeBase, Mem0EnhancedKnowledgeBase, MEM0_AVAILABLE
    from memory_manager import MemoryManager
    LEGACY_KB_AVAILABLE = True
except ImportError:
    LEGACY_KB_AVAILABLE = False
    MEM0_AVAILABLE = False

# --- Import New Skill Management System ---
from skill_manager import SkillManager
from skill_schema import Skill
from tool_governance import ToolIndex

# Global skill manager instance (replaces global_memory_manager)
global_skill_manager = None

# Legacy compatibility
global_memory_manager = None
use_templates = False  # Global flag for template/skill usage

# Global custom prompt templates
custom_prompt_templates = None

# --- Lightweight Automatic Memory System ---
class AutoMemory:
    """Lightweight memory that automatically tracks agent activities"""
    def __init__(self):
        self.task_history = deque(maxlen=50)  # Recent tasks
        self.tool_usage = {}  # Tool usage statistics
        self.success_patterns = {}  # Successful task patterns
        self.error_history = deque(maxlen=20)  # Recent errors
        self.agent_performance = {}  # Agent performance metrics
        
    def record_task(self, agent_name: str, task: str, result: str, success: bool, duration: float):
        """Automatically record task execution"""
        self.task_history.append({
            'agent': agent_name,
            'task': task[:100],
            'success': success,
            'duration': duration,
            'timestamp': time.time()
        })
        
        # Update agent performance
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {'total': 0, 'success': 0, 'avg_duration': 0}
        
        stats = self.agent_performance[agent_name]
        stats['total'] += 1
        if success:
            stats['success'] += 1
        
        # Update average duration
        old_avg = stats['avg_duration']
        stats['avg_duration'] = (old_avg * (stats['total'] - 1) + duration) / stats['total']
        
    def record_tool_use(self, tool_name: str, success: bool):
        """Record tool usage"""
        if tool_name not in self.tool_usage:
            self.tool_usage[tool_name] = {'uses': 0, 'success': 0}
        
        self.tool_usage[tool_name]['uses'] += 1
        if success:
            self.tool_usage[tool_name]['success'] += 1
    
    def get_similar_tasks(self, task: str, limit: int = 3):
        """Find similar successful tasks"""
        keywords = set(task.lower().split())
        matches = []
        
        for hist in self.task_history:
            if hist['success']:
                task_keywords = set(hist['task'].lower().split())
                score = len(keywords & task_keywords)
                if score > 0:
                    matches.append((score, hist))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]
    
    def get_best_agent_for_task(self, task: str):
        """Suggest best agent based on performance"""
        similar_tasks = self.get_similar_tasks(task)
        if similar_tasks:
            # Count which agents succeeded most
            agent_counts = {}
            for t in similar_tasks:
                agent = t['agent']
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            # Return agent with most successes
            return max(agent_counts.items(), key=lambda x: x[1])[0] if agent_counts else None
        return None

# Global auto memory instance
auto_memory = AutoMemory()

# --- Self-Evolution Tools ---
# Global registry for dynamically created tools
dynamic_tools_registry = {}


def _is_tool_object(obj) -> bool:
    """Return True if an object looks like a smolagents tool instance or legacy tool function."""
    return (
        hasattr(obj, "forward") and hasattr(obj, "name")
    ) or hasattr(obj, "__smolagents_tool__")


def _discover_tools_in_module(module) -> list:
    """Find tool objects exported by a dynamically loaded module."""
    import inspect

    tools = []
    for _, obj in inspect.getmembers(module):
        if _is_tool_object(obj):
            tools.append(obj)
    return tools


def _register_tool_with_agent(agent, tool_obj) -> bool:
    """Register a tool with an agent and its Python executor if available."""
    if agent is None:
        return False

    added = False
    tool_name = getattr(tool_obj, "name", getattr(tool_obj, "__name__", None))
    if not tool_name:
        return False

    try:
        if isinstance(agent.tools, dict):
            if tool_name not in agent.tools:
                agent.tools[tool_name] = tool_obj
                added = True
        elif isinstance(agent.tools, list):
            if tool_obj not in agent.tools:
                agent.tools.append(tool_obj)
                added = True
    except Exception:
        pass

    try:
        if hasattr(agent, "python_executor") and hasattr(agent.python_executor, "custom_tools"):
            agent.python_executor.custom_tools[tool_name] = tool_obj
    except Exception:
        pass

    return added

# --- Performance Optimization: Tool Loading Cache ---
tool_loading_cache = {}
tool_loading_lock = threading.Lock()

# --- Performance Optimization: Retry mechanism ---
def retry_on_failure(max_retries=3, delay=1.0):
    """Decorator to retry failed operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function that implements retry logic with exponential backoff."""
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise last_exception
            return None
        return wrapper
    return decorator


# --- Simple Memory Tools ---
@tool
def auto_recall_experience(task_description: str) -> str:
    """Automatically recall similar past tasks and their outcomes.
    
    Args:
        task_description: Description of the current task to find similar experiences for
        
    Returns:
        List of similar successful tasks with execution times and recommended agent
    """
    similar_tasks = auto_memory.get_similar_tasks(task_description, 3)
    
    if not similar_tasks:
        return "No similar past tasks found"
    
    result = f"Found {len(similar_tasks)} similar tasks:\n"
    for i, task in enumerate(similar_tasks, 1):
        duration = task['duration']
        result += f"{i}. {task['task']} - took {duration:.1f}s\n"
    
    # Suggest best agent
    best_agent = auto_memory.get_best_agent_for_task(task_description)
    if best_agent:
        result += f"\nRecommended agent: {best_agent}"
    
    return result

@tool 
def check_agent_performance() -> str:
    """Check which agents perform best on different types of tasks.
    
    Returns:
        Performance statistics for all agents including success rates and average execution times
    """
    if not auto_memory.agent_performance:
        return "No performance data available yet"
    
    result = "Agent Performance:\n"
    for agent, stats in auto_memory.agent_performance.items():
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        result += f"- {agent}: {success_rate:.0%} success, avg {stats['avg_duration']:.1f}s ({stats['total']} tasks)\n"
    
    return result

@tool
def quick_tool_stats() -> str:
    """Quick overview of which tools work best.
    
    Returns:
        Tool effectiveness rankings showing success rates and usage counts
    """
    if not auto_memory.tool_usage:
        return "No tool usage data yet"
    
    # Sort by success rate
    tool_stats = []
    for tool, stats in auto_memory.tool_usage.items():
        if stats['uses'] > 0:
            success_rate = stats['success'] / stats['uses']
            tool_stats.append((success_rate, tool, stats['uses']))
    
    tool_stats.sort(reverse=True)
    
    result = "Top performing tools:\n"
    for rate, tool, uses in tool_stats[:5]:
        result += f"- {tool}: {rate:.0%} success ({uses} uses)\n"
    
    return result

# --- Memory-Enhanced Agent Wrapper ---
def create_memory_enabled_agent(agent, agent_name):
    """Wrap an agent to automatically record task performance"""
    original_run = agent.run
    
    def run_with_memory(*args, **kwargs):
        """Enhanced run method that automatically records task performance and suggests improvements."""
        start_time = time.time()
        success = False
        result = ""
        
        # Extract task from args or kwargs
        task = args[0] if args else kwargs.get('task', 'Unknown task')
        
        try:
            # Check for similar past tasks first
            similar = auto_memory.get_similar_tasks(str(task), 2)
            if similar:
                print(f"💡 {agent_name}: Found {len(similar)} similar successful tasks")
            
            # Execute the task with all original arguments
            result = original_run(*args, **kwargs)
            success = True
            
            # Record tool usage (simplified) - avoid errors
            try:
                tools_used = getattr(agent, 'tools', [])
                if tools_used and isinstance(tools_used, list):
                    # Only record the last few tools to avoid excessive logging
                    recent_tools = tools_used[-3:] if len(tools_used) >= 3 else tools_used
                    for tool in recent_tools:
                        tool_name = getattr(tool, '__name__', getattr(tool, 'name', str(tool)))
                        auto_memory.record_tool_use(tool_name, success)
            except Exception:
                # Silent fail - don't break the main task for tool recording
                pass
            
            return result
            
        except Exception as e:
            result = str(e)
            raise
            
        finally:
            # Always record the task attempt
            duration = time.time() - start_time
            auto_memory.record_task(agent_name, str(task), str(result)[:100], success, duration)
    
    # Replace the run method
    agent.run = run_with_memory
    return agent

@tool
def evaluate_with_critic(task_description: str, current_result: str, expected_outcome: str = "") -> str:
    """Use the critic agent to evaluate task completion and recommend improvements.
    
    Args:
        task_description: Original task description
        current_result: Current result or output achieved
        expected_outcome: Expected outcome (optional)
        
    Returns:
        Critic evaluation with tool creation recommendations
    """
    try:
        # Optimized prompt - more concise
        evaluation_prompt = f"""
Evaluate ML model task completion with focus on PERFORMANCE METRICS:

TASK: {task_description}
FULL RESULT: {current_result[:1500]}  # Include more context for performance analysis
EXPECTED: {expected_outcome if expected_outcome else "High-performance ML model with good correlation scores"}

Provide evaluation focusing on ACTUAL PERFORMANCE:
1. status: EXCELLENT/SATISFACTORY/NEEDS_IMPROVEMENT/POOR (based on performance metrics, not just completion)
2. quality_score: 1-10 (heavily weight actual performance metrics)
3. gaps: performance issues and missing optimizations (max 3)
4. should_create_tool: true/false (recommend iteration tools for poor performance)
5. recommended_tool: if performance is poor, suggest optimization tools
6. performance_analysis: brief analysis of the numerical results

Be critical of poor performance. Completion ≠ Success.
"""
        
        critic_response = critic_agent.run(evaluation_prompt)
        return critic_response
        
    except Exception as e:
        return f"Error in critic evaluation: {str(e)}"
    

@tool
def list_dynamic_tools() -> str:
    """List all dynamically created tools.
    
    Returns:
        List of created tools with their purposes
    """
    if not dynamic_tools_registry:
        return "No dynamic tools have been created yet."
    
    result = f"Dynamic Tools ({len(dynamic_tools_registry)}):\n"
    
    for tool_name, tool_info in dynamic_tools_registry.items():
        result += f"• {tool_name}: {tool_info['purpose'][:50]}...\n"
    
    return result


@tool
def create_new_tool(tool_name: str, tool_purpose: str, tool_category: str, technical_requirements: str) -> str:
    """Use the tool creation agent to create a new specialized tool.
    
    Args:
        tool_name: Name of the tool to create
        tool_purpose: Detailed description of what the tool should do
        tool_category: Category of the tool (analysis, visualization, data_processing, modeling, etc.)
        technical_requirements: Specific technical requirements and implementation details
        
    Returns:
        Result of tool creation process
    """
    try:
        creation_task = f"""
Create a new Python tool with the following specifications:

TOOL NAME: {tool_name}
PURPOSE: {tool_purpose}
CATEGORY: {tool_category}
TECHNICAL REQUIREMENTS: {technical_requirements}

Requirements:
1. Create a Python file in the ./new_tools/ directory named '{tool_name}.py'
2. The tool should be implemented as a function decorated with @tool from smolagents
3. Include proper docstrings with Args and Returns sections
4. Add error handling and input validation
5. Import all necessary dependencies at the top of the file
6. Include type hints for all function parameters and returns
7. Test the tool functionality after creation

The tool should be production-ready and immediately usable by other agents.
"""
        
        result = tool_creation_agent.run(creation_task)
        
        # Register the tool in the dynamic registry
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        dynamic_tools_registry[tool_name] = {
            'purpose': tool_purpose,
            'category': tool_category,
            'created_at': current_time,
            'file_path': f'./new_tools/{tool_name}.py'
        }

        # Generate tool manifest for governance
        if global_skill_manager is not None:
            try:
                manifest = global_skill_manager.tool_index.auto_generate_manifest(
                    function_name=tool_name,
                    module_path=f"new_tools.{tool_name}",
                    category=tool_category,
                )
                if manifest:
                    global_skill_manager.tool_index.register(manifest, save=True)
                    print(f"Tool manifest generated for {tool_name}")
            except Exception as e:
                print(f"Warning: manifest generation failed: {e}")

        # Automatically load the created tool into the agents
        load_result = load_dynamic_tool(tool_name, add_to_agents=True)
        tool_file_path = f'./new_tools/{tool_name}.py'
        tool_file_exists = os.path.exists(tool_file_path)

        if tool_file_exists and "Successfully loaded tool" in load_result:
            final_result = (
                f"Tool creation completed!\n\n{result}\n\n"
                f"Tool '{tool_name}' registered with governance manifest.\n\n"
                f"Auto-loading result: {load_result}"
            )
        else:
            final_result = (
                f"⚠️ Tool creation attempted but not fully materialized.\n\n{result}\n\n"
                f"Tool file exists: {tool_file_exists}\n"
                f"Auto-loading result: {load_result}"
            )
        
        return final_result
        
    except Exception as e:
        return f"❌ Error creating tool: {str(e)}"


@tool
@retry_on_failure(max_retries=2)
def load_dynamic_tool(tool_name: str, add_to_agents: bool = True) -> str:
    """Dynamically load a tool from the new_tools directory and optionally add it to agents.
    
    Args:
        tool_name: Name of the tool to load
        add_to_agents: Whether to add the loaded tool to dev_agent and tool_creation_agent
        
    Returns:
        Status of the loading operation
    """
    try:
        import importlib.util
        import sys
        
        # Ensure new_tools directory exists
        os.makedirs('./new_tools', exist_ok=True)
        
        tool_file_path = f'./new_tools/{tool_name}.py'
        
        if not os.path.exists(tool_file_path):
            return f"❌ Tool file '{tool_file_path}' not found."
        
        # Load the module
        spec = importlib.util.spec_from_file_location(tool_name, tool_file_path)
        if spec is None or spec.loader is None:
            return f"❌ Could not load module specification for '{tool_name}'."
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[tool_name] = module
        spec.loader.exec_module(module)
        
        result = f"✅ Successfully loaded tool '{tool_name}' from {tool_file_path}"
        
        if add_to_agents:
            tool_objects = _discover_tools_in_module(module)

            if tool_objects:
                added_agents = set()
                for tool_obj in tool_objects:
                    if _register_tool_with_agent(dev_agent, tool_obj):
                        added_agents.add("dev_agent")
                    if _register_tool_with_agent(tool_creation_agent, tool_obj):
                        added_agents.add("tool_creation_agent")
                    if _register_tool_with_agent(manager_agent, tool_obj):
                        added_agents.add("manager_agent")

                if added_agents:
                    result += (
                        f"\n🔧 Added {len(tool_objects)} tool object(s) to "
                        + ", ".join(sorted(added_agents))
                    )
                else:
                    result += f"\nℹ️ Tool object(s) already available to agents"
            else:
                result += "\n⚠️ No smolagents tool objects found in the module"
        
        return result
        
    except Exception as e:
        return f"❌ Error loading tool '{tool_name}': {str(e)}"


@tool
def execute_tools_in_parallel(tool_calls: list, max_workers: int = 3, timeout: int = 30) -> str:
    """Execute multiple tool calls in parallel to improve efficiency.
    
    Args:
        tool_calls: List of tool call dictionaries with 'tool_name' and 'args' keys
        max_workers: Maximum number of parallel workers (default: 3)
        timeout: Timeout in seconds for each tool call (default: 30)
        
    Returns:
        Formatted results from all parallel tool executions
        
    Example:
        tool_calls = [
            {"tool_name": "query_pubmed", "args": {"query": "protein research", "max_results": 5}},
            {"tool_name": "query_uniprot", "args": {"genes": ["FAST"], "fields": "function"}},
            {"tool_name": "multi_source_search", "args": {"query": "cell fusion", "sources": "google"}}
        ]
        results = execute_tools_in_parallel(tool_calls)
    """
    try:
        import concurrent.futures
        import time
        
        global manager_agent
        
        if not tool_calls:
            return "❌ No tool calls provided for parallel execution"
        
        if not isinstance(tool_calls, list):
            return "❌ tool_calls must be a list of dictionaries"
        
        # Validate tool calls format
        for i, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                return f"❌ Tool call {i+1} must be a dictionary"
            if 'tool_name' not in call:
                return f"❌ Tool call {i+1} missing 'tool_name'"
            if 'args' not in call:
                return f"❌ Tool call {i+1} missing 'args'"
            
            # Check if tool exists (flexible approach)
            tool_name = call['tool_name']
            tool_found = False
            
            try:
                # Try dictionary access first
                if hasattr(manager_agent.tools, 'get'):
                    tool_found = manager_agent.tools.get(tool_name) is not None
                elif hasattr(manager_agent.tools, '__contains__'):
                    tool_found = tool_name in manager_agent.tools
                else:
                    # List-like search
                    tool_found = any(getattr(t, 'name', getattr(t, '__name__', str(t))) == tool_name for t in manager_agent.tools)
            except:
                tool_found = False
            
            if not tool_found:
                return f"❌ Tool '{tool_name}' not found in loaded tools"
        
        def execute_single_tool(tool_call):
            """Execute a single tool call with timeout"""
            tool_name = tool_call['tool_name']
            args = tool_call['args']
            start_time = time.time()
            
            try:
                # Get tool function - flexible approach
                tool_func = None
                
                # Try different ways to access the tool
                if hasattr(manager_agent.tools, 'get'):
                    # Dictionary-like access
                    tool_func = manager_agent.tools.get(tool_name)
                elif hasattr(manager_agent.tools, '__getitem__'):
                    # Dictionary or list-like with indexing
                    try:
                        tool_func = manager_agent.tools[tool_name]
                    except (KeyError, TypeError):
                        # Maybe it's a list, search by name
                        for tool in manager_agent.tools:
                            if getattr(tool, 'name', getattr(tool, '__name__', str(tool))) == tool_name:
                                tool_func = tool
                                break
                else:
                    # List-like search
                    for tool in manager_agent.tools:
                        if getattr(tool, 'name', getattr(tool, '__name__', str(tool))) == tool_name:
                            tool_func = tool
                            break
                
                if tool_func is None:
                    raise ValueError(f"Tool '{tool_name}' not found")
                
                # Handle different tool calling conventions with debugging
                import inspect
                
                # Debug: Log tool call attempt
                if manager_agent and hasattr(manager_agent, 'verbose') and manager_agent.verbose:
                    print(f"🔧 Tool signature:")
                    try:
                        if hasattr(tool_func, 'forward'):
                            print(f"{tool_name}{inspect.signature(tool_func.forward)}")
                        elif callable(tool_func):
                            print(f"{tool_name}{inspect.signature(tool_func)}")
                        else:
                            print(f"{tool_name}(*args, sanitize_inputs_outputs: bool = False, **kwargs)")
                    except:
                        print(f"{tool_name}(unknown signature)")
                    print(f"\n📖 Documentation:")
                    print(tool_func.__doc__[:200] if tool_func.__doc__ else "No documentation available...")
                    print()
                
                # Check if this is a dynamic tool that expects positional args
                if 'args' in args and isinstance(args['args'], str):
                    # This is likely a dynamic tool that expects a positional string argument
                    try:
                        # Extract additional parameters
                        sanitize = args.get('sanitize_inputs_outputs', False)
                        kwargs = {k: v for k, v in args.items() if k not in ['args', 'sanitize_inputs_outputs']}
                        
                        # Call with positional argument first
                        if kwargs:
                            result = tool_func(args['args'], sanitize_inputs_outputs=sanitize, **kwargs)
                        else:
                            result = tool_func(args['args'], sanitize_inputs_outputs=sanitize)
                    except Exception as e:
                        # If that fails, try standard keyword arguments
                        try:
                            result = tool_func(**args)
                        except Exception as ex:
                            # Log more details about the failure
                            print(f"❌ Tool {tool_name} failed with both calling conventions")
                            print(f"   Positional args error: {e}")
                            print(f"   Keyword args error: {ex}")
                            print(f"   Args provided: {args}")
                            # Re-raise the original error
                            raise e
                else:
                    # Standard tool with keyword arguments
                    try:
                        result = tool_func(**args)
                    except TypeError as e:
                        error_str = str(e)
                        print(f"🔍 Debug: Tool {tool_name} failed with: {error_str}")
                        print(f"🔍 Debug: Tool signature: {inspect.signature(tool_func)}")
                        print(f"🔍 Debug: Args provided: {args}")
                        
                        # Try to provide helpful hints based on the error
                        if "unexpected keyword argument" in error_str:
                            print(f"💡 Hint: This tool may expect different parameters. Check the tool's documentation.")
                            # Try extracting the positional arg if present
                            if 'prompt' in args:
                                print(f"💡 Trying with prompt as positional argument...")
                                try:
                                    result = tool_func(args['prompt'], **{k: v for k, v in args.items() if k != 'prompt'})
                                    print(f"✅ Success with positional prompt!")
                                except:
                                    raise e
                            else:
                                raise
                        else:
                            raise
                duration = time.time() - start_time
                
                return {
                    'tool_name': tool_name,
                    'success': True,
                    'result': result,
                    'duration': duration,
                    'error': None
                }
            except Exception as e:
                duration = time.time() - start_time
                return {
                    'tool_name': tool_name,
                    'success': False,
                    'result': None,
                    'duration': duration,
                    'error': str(e)
                }
        
        # Execute tools in parallel
        results = []
        start_total = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tool calls
            future_to_call = {
                executor.submit(execute_single_tool, call): call 
                for call in tool_calls
            }
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_call, timeout=timeout):
                try:
                    result = future.result(timeout=5)  # Individual result timeout
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    call = future_to_call[future]
                    results.append({
                        'tool_name': call['tool_name'],
                        'success': False,
                        'result': None,
                        'duration': timeout,
                        'error': 'Timeout'
                    })
                except Exception as e:
                    call = future_to_call[future]
                    results.append({
                        'tool_name': call['tool_name'],
                        'success': False,
                        'result': None,
                        'duration': 0,
                        'error': str(e)
                    })
        
        total_duration = time.time() - start_total
        
        # Format results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        output = f"🚀 Parallel Execution Complete ({len(tool_calls)} tools, {total_duration:.1f}s total)\n"
        output += f"✅ Successful: {len(successful)} | ❌ Failed: {len(failed)}\n\n"
        
        # Show successful results
        if successful:
            output += "📋 Successful Results:\n"
            for result in successful:
                tool_name = result['tool_name']
                duration = result['duration']
                result_preview = str(result['result'])[:100] + "..." if len(str(result['result'])) > 100 else str(result['result'])
                output += f"  ✅ {tool_name} ({duration:.1f}s): {result_preview}\n"
        
        # Show failed results
        if failed:
            output += f"\n❌ Failed Results:\n"
            for result in failed:
                tool_name = result['tool_name']
                error = result['error']
                output += f"  ❌ {tool_name}: {error}\n"
        
        # Performance summary
        if successful:
            avg_duration = sum(r['duration'] for r in successful) / len(successful)
            max_duration = max(r['duration'] for r in successful)
            output += f"\n📊 Performance: Avg {avg_duration:.1f}s, Max {max_duration:.1f}s"
            
            # Calculate efficiency gain
            sequential_time = sum(r['duration'] for r in successful)
            if sequential_time > total_duration:
                speedup = sequential_time / total_duration
                output += f", {speedup:.1f}x speedup vs sequential"
        
        return output
        
    except Exception as e:
        return f"❌ Parallel execution error: {str(e)}"

@tool
def analyze_query_and_load_relevant_tools(user_query: str, max_tools: int = 10) -> str:
    """Analyze user query using LLM and intelligently load the most relevant tools from database_tools.py, virtual_screening_tools.py, and biosecurity_tools.py.
    
    Optimized version with caching and reduced token usage.
    
    Args:
        user_query: The user's task description or query
        max_tools: Maximum number of relevant tools to load (default: 10)
        
    Returns:
        Status of the tool loading operation with analysis details
    """
    global manager_agent, tool_creation_agent  # 添加全局变量声明
    try:
        # Check cache first
        query_hash = hashlib.md5(user_query.encode()).hexdigest()
        cache_key = f"{query_hash}_{max_tools}"
        
        with tool_loading_lock:
            if cache_key in tool_loading_cache:
                cached_result, cached_time = tool_loading_cache[cache_key]
                # Use cache if less than 5 minutes old
                if time.time() - cached_time < 300:
                    return f"🔄 Using cached tool selection\n{cached_result}"
        
        import inspect
        import importlib.util
        import sys
        import os
        
        # Import LLM functionality
        current_dir = os.path.dirname(os.path.abspath(__file__))
        llm_path = os.path.join(current_dir, 'new_tools')
        sys.path.insert(0, llm_path)
        from llm import json_llm_call
        
        # Define tool files to analyze (use absolute paths based on script location)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tool_files = {
            'database_tools': os.path.join(script_dir, 'new_tools', 'database_tools.py'),
            'virtual_screening_tools': os.path.join(script_dir, 'new_tools', 'virtual_screening_tools.py'),
            'enzyme_tools': os.path.join(script_dir, 'new_tools', 'enzyme_tools.py')
        }
        
        available_tools = {}
        
        # Extract tools and their descriptions from each file
        for module_name, file_path in tool_files.items():
            if not os.path.exists(file_path):
                continue
                
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Find all tools (SimpleTool objects created by @tool decorator)
                for name, obj in inspect.getmembers(module):
                    if hasattr(obj, '__class__') and 'SimpleTool' in str(type(obj)):
                        # Extract description from tool object or docstring
                        if hasattr(obj, 'description') and obj.description:
                            description = obj.description
                        else:
                            doc = inspect.getdoc(obj) or ""
                            description = doc.split('\n\n')[0].replace('\n', ' ').strip()
                        
                        available_tools[name] = {
                            'function': obj,
                            'description': description,
                            'module': module_name,
                            'file_path': file_path
                        }
                            
            except Exception as e:
                continue
        
        if not available_tools:
            return f"❌ No tools found in database_tools.py, virtual_screening_tools.py, or biosecurity_tools.py"
        
        # Create tool list for LLM analysis - OPTIMIZED
        tool_list = []
        for tool_name, tool_info in available_tools.items():
            tool_list.append({
                "name": tool_name,
                "description": tool_info['description'][:100],  # Truncate descriptions
                "module": tool_info['module']
            })
        
        # Create OPTIMIZED LLM prompt for intelligent tool selection
        llm_prompt = f"""Select relevant tools for this query: "{user_query}"

Available tools ({len(tool_list)}):
{chr(10).join([f"{i+1}. {tool['name']} [{tool['module']}]: {tool['description']}" for i, tool in enumerate(tool_list[:20])])}

Return JSON with top {max_tools} most relevant tools:
{{
    "selected_tools": [
        {{"name": "tool_name", "relevance_score": 0.95}}
    ]
}}"""
        
        # Use LLM to select tools intelligently
        try:
            llm_response = json_llm_call(llm_prompt, "gemini-3-pro")
            
            if "error" in llm_response:
                # Fallback to simple keyword matching if LLM fails
                return _fallback_tool_selection(user_query, available_tools, max_tools)
            
            selected_tool_data = llm_response.get("selected_tools", [])
            
            if not selected_tool_data:
                return f"🔍 LLM analysis found no relevant tools for query: '{user_query}'"
            
        except Exception as e:
            # Fallback to simple selection if LLM fails completely
            return _fallback_tool_selection(user_query, available_tools, max_tools)
        
        # Load selected tools into agents
        loaded_tools = []
        loaded_count = 0
        
        for tool_selection in selected_tool_data:
            tool_name = tool_selection.get("name")
            
            if tool_name not in available_tools:
                continue
                
            try:
                tool_info = available_tools[tool_name]
                tool_func = tool_info['function']
                
                # Add to manager_agent tools if not already present
                if tool_name not in manager_agent.tools:
                    manager_agent.tools[tool_name] = tool_func
                    
                    # 重要：也要更新CodeAgent的Python执行器
                    if hasattr(manager_agent, 'python_executor') and hasattr(manager_agent.python_executor, 'custom_tools'):
                        manager_agent.python_executor.custom_tools[tool_name] = tool_func
                    
                    # Add to dynamic_tools_registry so list_dynamic_tools can find it
                    dynamic_tools_registry[tool_name] = {
                        'purpose': tool_info.get('description', f'Loaded biomedical tool from {tool_info["module"]}'),
                        'module': tool_info['module'],
                        'source': 'loaded_tool'
                    }
                    
                    loaded_count += 1
                
                # Add to tool_creation_agent tools if not already present  
                if tool_name not in tool_creation_agent.tools:
                    tool_creation_agent.tools[tool_name] = tool_func
                    
                    # 重要：也要更新CodeAgent的Python执行器
                    if hasattr(tool_creation_agent, 'python_executor') and hasattr(tool_creation_agent.python_executor, 'custom_tools'):
                        tool_creation_agent.python_executor.custom_tools[tool_name] = tool_func
                
                loaded_tools.append({
                    'name': tool_name,
                    'relevance': tool_selection.get("relevance_score", 0.0),
                    'module': tool_info['module']
                })
                
            except Exception as e:
                continue
        
        # Generate enhanced result with tool signatures
        result = f"🎯 Loaded {loaded_count} tools for: '{user_query[:50]}...'\n"
        result += f"📋 Tools with signatures:\n"
        
        # Add tool signatures for immediate use
        for i, tool_data in enumerate(loaded_tools, 1):
            tool_name = tool_data['name']
            try:
                # Get tool signature inline
                if tool_name in manager_agent.tools:
                    import inspect
                    tool_func = manager_agent.tools[tool_name]
                    
                    # Try to get signature from the actual function
                    if hasattr(tool_func, 'forward'):
                        sig = inspect.signature(tool_func.forward)
                    elif callable(tool_func):
                        sig = inspect.signature(tool_func)
                    else:
                        result += f"  {i}. {tool_name}(args, sanitize_inputs_outputs: bool, kwargs)\n"
                        continue
                    
                    # Show COMPLETE signature with parameter types
                    params = []
                    for param_name, param in sig.parameters.items():
                        if param_name in ['self', 'cls']:  # Skip self/cls parameters
                            continue
                        if param.annotation != inspect.Parameter.empty:
                            param_type = getattr(param.annotation, '__name__', str(param.annotation))
                            if param.default != inspect.Parameter.empty:
                                params.append(f"{param_name}: {param_type} = {param.default}")
                            else:
                                params.append(f"{param_name}: {param_type}")
                        else:
                            if param.default != inspect.Parameter.empty:
                                params.append(f"{param_name} = {param.default}")
                            else:
                                params.append(param_name)
                    
                    param_str = ", ".join(params) if params else "no params"
                    
                    result += f"  {i}. {tool_name}({param_str})\n"
                else:
                    result += f"  {i}. {tool_name} (signature unavailable)\n"
            except Exception as e:
                # For dynamic tools, show the standard signature
                result += f"  {i}. {tool_name}(*args, sanitize_inputs_outputs: bool = False, **kwargs)\n"
        
        result += f"\n💡 All tools ready to use with correct parameter names shown above"
        
        # Cache the result
        with tool_loading_lock:
            tool_loading_cache[cache_key] = (result, time.time())
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing query and loading tools: {str(e)}"


def _fallback_tool_selection(user_query: str, available_tools: dict, max_tools: int) -> str:
    """Fallback tool selection using simple keyword matching when LLM fails"""
    global manager_agent, tool_creation_agent  # 添加全局变量声明
    query_lower = user_query.lower()
    tool_scores = []
    
    # Simple keyword matching
    for tool_name, tool_info in available_tools.items():
        tool_text = f"{tool_name.replace('_', ' ')} {tool_info['description']}".lower()
        
        # Score based on keyword matches
        score = 0
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 2 and word in tool_text:
                score += 1
        
        tool_scores.append((tool_name, score))
    
    # Sort by score and take top tools
    tool_scores.sort(key=lambda x: x[1], reverse=True)
    selected_tools = tool_scores[:max_tools]
    
    if not selected_tools or all(score == 0 for _, score in selected_tools):
        return f"🔍 No relevant tools found for query: '{user_query}' (fallback method used)"
    
    # Load tools and return summary
    loaded_count = 0
    for tool_name, score in selected_tools:
        if score > 0:
            tool_func = available_tools[tool_name]['function']
            # 修复：正确处理工具字典而不是列表
            if tool_name not in manager_agent.tools:
                manager_agent.tools[tool_name] = tool_func
                
                # 重要：也要更新CodeAgent的Python执行器
                if hasattr(manager_agent, 'python_executor') and hasattr(manager_agent.python_executor, 'custom_tools'):
                    manager_agent.python_executor.custom_tools[tool_name] = tool_func
                
                # Add to dynamic_tools_registry so list_dynamic_tools can find it
                dynamic_tools_registry[tool_name] = {
                    'purpose': f'Loaded biomedical tool (fallback method)',
                    'module': 'fallback_selection',
                    'source': 'loaded_tool'
                }
                
                loaded_count += 1
            if tool_name not in tool_creation_agent.tools:
                tool_creation_agent.tools[tool_name] = tool_func
                
                # 重要：也要更新CodeAgent的Python执行器
                if hasattr(tool_creation_agent, 'python_executor') and hasattr(tool_creation_agent.python_executor, 'custom_tools'):
                    tool_creation_agent.python_executor.custom_tools[tool_name] = tool_func
    
    # Generate enhanced fallback result with signatures
    result = f"🎯 Fallback Analysis: '{user_query}'\n✅ Loaded {loaded_count} tools using keyword matching.\n"
    result += f"📋 Tools with signatures:\n"
    
    # Add signatures for fallback loaded tools
    tool_count = 0
    for tool_name, score in selected_tools:
        if score > 0 and tool_count < 10:
            try:
                if tool_name in manager_agent.tools:
                    import inspect
                    tool_func = manager_agent.tools[tool_name]
                    sig = inspect.signature(tool_func.forward)
                    
                    # Show COMPLETE signature with parameter types
                    params = []
                    for param_name, param in sig.parameters.items():
                        if param.annotation != inspect.Parameter.empty:
                            param_type = getattr(param.annotation, '__name__', str(param.annotation))
                            params.append(f"{param_name}: {param_type}")
                        else:
                            params.append(param_name)
                    
                    param_str = ", ".join(params) if params else "no params"
                    result += f"  {tool_count+1}. {tool_name}({param_str})\n"
                    tool_count += 1
            except Exception:
                pass
    
    result += f"\n💡 Use tools with the parameter names shown above"
    return result


@tool
def refresh_agent_tools() -> str:
    """Refresh agent tools by loading all available tools from the new_tools directory.
    
    Returns:
        Status of the refresh operation
    """
    try:
        import os
        import glob
        
        new_tools_dir = './new_tools'
        if not os.path.exists(new_tools_dir):
            return "📁 new_tools directory does not exist yet."
        
        # Find all Python files in new_tools directory
        tool_files = glob.glob(os.path.join(new_tools_dir, '*.py'))
        
        if not tool_files:
            return "📁 No tool files found in new_tools directory."
        
        loaded_count = 0
        results = []
        
        for tool_file in tool_files:
            tool_name = os.path.splitext(os.path.basename(tool_file))[0]
            try:
                result = load_dynamic_tool(tool_name, add_to_agents=True)
                if "✅" in result:
                    loaded_count += 1
                results.append(f"  - {tool_name}: {'✅' if '✅' in result else '❌'}")
            except Exception as e:
                results.append(f"  - {tool_name}: ❌ {str(e)}")
        
        summary = f"🔄 Agent tools refresh completed!\n"
        summary += f"📊 Loaded {loaded_count}/{len(tool_files)} tools:\n"
        summary += "\n".join(results)
        
        return summary
        
    except Exception as e:
        return f"❌ Error refreshing agent tools: {str(e)}"


@tool
def get_tool_signature(tool_name: str) -> str:
    """Get the complete function signature of a loaded tool.
    
    Args:
        tool_name: Name of the tool to get signature for
        
    Returns:
        Complete function signature with parameter types and descriptions
    """
    global manager_agent
    
    try:
        if tool_name not in manager_agent.tools:
            return f"❌ Tool '{tool_name}' not found in loaded tools"
        
        import inspect
        tool_func = manager_agent.tools[tool_name]
        sig = inspect.signature(tool_func.forward)
        
        # Get complete signature with types
        params = []
        for param_name, param in sig.parameters.items():
            param_info = param_name
            if param.annotation != inspect.Parameter.empty:
                param_type = getattr(param.annotation, '__name__', str(param.annotation))
                param_info += f": {param_type}"
            if param.default != inspect.Parameter.empty:
                param_info += f" = {param.default}"
            params.append(param_info)
        
        signature = f"{tool_name}({', '.join(params)})"
        
        # Get docstring for parameter descriptions
        doc = inspect.getdoc(tool_func) or "No documentation available"
        
        result = f"🔧 Tool signature:\n{signature}\n\n📖 Documentation:\n{doc[:500]}..."
        
        return result
        
    except Exception as e:
        return f"❌ Error getting tool signature: {str(e)}"


@tool
def add_tool_to_agents(tool_function_name: str, module_name: str) -> str:
    """Add a specific tool function to dev_agent and tool_creation_agent.
    
    Args:
        tool_function_name: Name of the tool function to add
        module_name: Name of the module containing the tool
        
    Returns:
        Status of the operation
    """
    try:
        import sys
        
        if module_name not in sys.modules:
            return f"❌ Module '{module_name}' not loaded. Use load_dynamic_tool first."
        
        module = sys.modules[module_name]
        
        if not hasattr(module, tool_function_name):
            return f"❌ Function '{tool_function_name}' not found in module '{module_name}'."
        
        tool_func = getattr(module, tool_function_name)
        
        # Check if it's a tool function
        if not _is_tool_object(tool_func):
            return f"❌ Function '{tool_function_name}' is not a recognized smolagents tool object."
        
        # Add to agents if not already present
        added_to = []
        if _register_tool_with_agent(dev_agent, tool_func):
            added_to.append("dev_agent")
        
        if _register_tool_with_agent(tool_creation_agent, tool_func):
            added_to.append("tool_creation_agent")

        if _register_tool_with_agent(manager_agent, tool_func):
            added_to.append("manager_agent")
        
        if added_to:
            return f"✅ Tool '{tool_function_name}' added to: {', '.join(added_to)}"
        else:
            return f"ℹ️ Tool '{tool_function_name}' was already available in all agents."
        
    except Exception as e:
        return f"❌ Error adding tool to agents: {str(e)}"

# --- Skill Management Tools (replaces Knowledge Base Tools) ---
@tool
def retrieve_similar_skills(task_description: str, top_k: int = 3, domain: str = "") -> str:
    """Retrieve relevant skills (reusable workflows) for the current task.

    Uses hybrid 3-stage retrieval: tag matching, semantic similarity, and quality re-ranking.
    Each skill contains structured workflow steps, required tools, and quality metrics.

    Args:
        task_description: Description of the current task
        top_k: Number of skills to retrieve (default: 3)
        domain: Optional domain filter (e.g., "genomics", "drug_discovery")

    Returns:
        Ranked list of matching skills with workflows and tool requirements
    """
    global global_skill_manager

    if global_skill_manager is None:
        return "Skill system not initialized."

    try:
        return global_skill_manager.retrieve_skills(
            query=task_description,
            top_k=top_k,
            domain=domain if domain else None,
        )
    except Exception as e:
        return f"Error retrieving skills: {str(e)}"


@tool
def save_successful_skill(task_description: str, reasoning_process: str, solution_outcome: str, domain: str = "general", tools_used: str = "") -> str:
    """Save a successful problem-solving approach as a reusable skill.

    The skill system automatically extracts structured workflow steps,
    deduplicates against existing skills, and tracks quality metrics.

    Args:
        task_description: Description of the solved task
        reasoning_process: The reasoning process and steps that led to success
        solution_outcome: The successful outcome achieved
        domain: Domain category (default: "general")
        tools_used: Comma-separated list of tools used (default: "")

    Returns:
        Status of the save operation with skill ID
    """
    global global_skill_manager

    if global_skill_manager is None:
        return "Skill system not initialized."

    try:
        tool_list = [t.strip() for t in tools_used.split(",") if t.strip()] if tools_used else []
        return global_skill_manager.save_skill_from_run(
            query=task_description,
            reasoning_process=reasoning_process,
            result_summary=solution_outcome,
            tools_used=tool_list,
            domain=domain,
        )
    except Exception as e:
        return f"Error saving skill: {str(e)}"


@tool
def get_skill_system_status() -> str:
    """Get status and statistics of the skill management system.

    Shows skill counts, run statistics, success rates, and tool index summary.

    Returns:
        Comprehensive skill system status
    """
    global global_skill_manager

    if global_skill_manager is None:
        return "Skill system not initialized."

    try:
        return global_skill_manager.get_skill_status()
    except Exception as e:
        return f"Error getting skill status: {str(e)}"


@tool
def search_skills_by_keyword(keyword: str, limit: int = 5) -> str:
    """Search skills by keyword using hybrid retrieval.

    Args:
        keyword: Keyword to search for in skills
        limit: Maximum number of results to return (default: 5)

    Returns:
        Matching skills with relevance scores
    """
    global global_skill_manager

    if global_skill_manager is None:
        return "Skill system not initialized."

    try:
        return global_skill_manager.search_skills(keyword, limit=limit)
    except Exception as e:
        return f"Error searching skills: {str(e)}"


@tool
def check_tools_for_task(task_description: str) -> str:
    """Check which tools are needed and available for a task based on matching skills.

    Args:
        task_description: Description of the task

    Returns:
        Tool availability report for the task
    """
    global global_skill_manager

    if global_skill_manager is None:
        return "Skill system not initialized."

    try:
        return global_skill_manager.check_tools_for_query(task_description)
    except Exception as e:
        return f"Error checking tools: {str(e)}"


@tool
def export_skill_environment(skill_id: str, format: str = "requirements") -> str:
    """Export reproducible environment specification for a skill's tools.

    Args:
        skill_id: ID of the skill to export environment for
        format: Output format - "requirements" (pip) or "conda" (environment.yml)

    Returns:
        Environment specification content
    """
    global global_skill_manager

    if global_skill_manager is None:
        return "Skill system not initialized."

    try:
        return global_skill_manager.export_environment_for_skill(skill_id, format=format)
    except Exception as e:
        return f"Error exporting environment: {str(e)}"


# Legacy compatibility aliases
retrieve_similar_templates = retrieve_similar_skills
save_successful_template = save_successful_skill
list_knowledge_base_status = get_skill_system_status
search_templates_by_keyword = search_skills_by_keyword


claude_model = OpenAIServerModel(
    model_id="google/gemini-3-flash-preview",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY_STRING,
)

# Create a more capable model for manager and critic agents
gemini_model = OpenAIServerModel(
    model_id="google/gemini-3-flash-preview",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY_STRING,
    temperature=0.1,
)

gpt_model = OpenAIServerModel(
    model_id="google/gemini-3-flash-preview",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY_STRING,
    temperature=0.1,
)


# --- Tool Management Permissions ---
# 为 dev_agent 定义基础工具管理权限 - OPTIMIZED
dev_tool_management = [
    list_dynamic_tools,       # ✅ 查看可用工具
    load_dynamic_tool,        # ✅ 加载需要的工具
    refresh_agent_tools,      # ✅ 刷新自己的工具
    # Essential memory tools for dev_agent
    auto_recall_experience,   # 🧠 回忆相似任务经验
    quick_tool_stats,         # 🔧 快速工具效果统计
]

# Base manager tools (always available)
base_manager_tools = [
    multi_source_search,          # 🔍 Unified search tool (default: google,serpapi)
    search_github_repositories,
    search_github_code,
    get_github_repository_info,
    run_shell_command,  # 🖥️ Execute shell commands to read files
    analyze_query_and_load_relevant_tools,  # 🎯 Smart tool retrieval and loading
    execute_tools_in_parallel,  # 🚀 Parallel tool execution
    evaluate_with_critic,     # 🎯 Evaluate task quality
    list_dynamic_tools,       # 📋 View tool library
    load_dynamic_tool,        # 📦 Manage tool loading
    add_tool_to_agents,       # ➕ Fine-grained tool management
    get_tool_signature,       # 🔍 Get tool signatures
    # Skill management tools (replaces template/memory tools)
    retrieve_similar_skills,      # 🧠 Retrieve relevant skills with hybrid retrieval
    save_successful_skill,        # 💾 Save successful runs as reusable skills
    get_skill_system_status,      # 📊 Skill system status and metrics
    search_skills_by_keyword,     # 🔍 Search skills by keyword
    check_tools_for_task,         # 🔧 Check tool availability for tasks
    export_skill_environment,     # 📦 Export reproducible environments
    # Legacy memory tools (still useful for agent-level tracking)
    auto_recall_experience,   # 🧠 Recall similar task experiences
    check_agent_performance,  # 📊 Check agent performance
    quick_tool_stats,         # 🔧 Quick tool statistics
    extract_url_content,      # ✅ Specialized content extraction
    query_arxiv,
    query_pubmed,
    extract_pdf_content,
    fetch_supplementary_info_from_doi,
]

# Tool creation tools (only included when enabled)
tool_creation_tools_for_manager = [
    create_new_tool,          # 🛠️ 决定创建新工具
]

# Function to build manager tools based on configuration
def build_manager_tools(enable_tool_creation=False):
    """Build manager agent tools based on configuration."""
    tools = base_manager_tools.copy()
    if enable_tool_creation:
        tools.extend(tool_creation_tools_for_manager)
        print("🛠️ Tool creation enabled - added tool creation capabilities")
    else:
        print("⚡ Tool creation disabled - optimized for performance")
    return tools

# Create the web search and development agent (ToolCallingAgent)
base_tools = [
    # Core web and search tools - OPTIMIZED: Keep only most reliable tools
    extract_url_content,          # ✅ Specialized content extraction with BeautifulSoup
    query_arxiv,
    query_scholar,
    query_pubmed,
    extract_pdf_content,
    fetch_supplementary_info_from_doi,
    
    # Unified Search Tool for dev_agent
    multi_source_search,          # 🔍 Unified search tool - supports all search needs
    
    # GitHub tools
    search_github_repositories,
    search_github_code,
    get_github_repository_info,
    
    # Development environment tools
    run_shell_command,
    create_conda_environment,
    install_packages_conda,
    install_packages_pip,
    check_gpu_status,
    create_and_run_script,
    run_script,
    create_requirements_file,
    monitor_training_logs,
    
    # Basic tool management for dev_agent (read-only + basic operations)
] + dev_tool_management

dev_tools = base_tools

dev_agent = ToolCallingAgent(
    tools=dev_tools,
    model=claude_model,
    max_steps=30,  # Increased for complex tasks
    name="dev_agent",
    description="""A specialist agent for code execution and environment management through tool-based programming.
    
    Core Tool Capabilities:
    🔧 Code Execution & Script Management:
    - create_and_run_script: Create and execute Python code directly with no import restrictions
    - run_script: Run existing script files
    - run_shell_command: Execute system commands and shell operations
    
    📦 Package Management & Environment Setup:
    - install_packages_pip: Install Python packages using pip
    - install_packages_conda: Install packages and manage environments with conda
    - create_conda_environment: Create isolated conda environments
    - create_requirements_file: Generate project dependency files
    
    🔍 Research & Code Discovery:
    - search_github_repositories: Search GitHub repos to find best implementation solutions
    - search_github_code: Search for specific code snippets and implementations
    - multi_source_search: Comprehensive search across academic papers and technical resources
    
    📁 File & Data Management:
    - Complete file operations support: read/write, CSV processing, data loading
    - Multiple format support: JSON, CSV, pickle, HDF5, etc.
    
    Use Case Examples:
    - 'Analyze this CSV file and create visualizations' → use create_and_run_script for data analysis
    - 'Install fair-esm package and run protein embeddings' → install_packages_pip + create_and_run_script
    - 'Search for best ESM protein model implementations' → search_github_repositories
    - 'Create complete machine learning pipeline script' → create_and_run_script
    - 'Setup bioinformatics analysis environment' → create_conda_environment + package installation
    """,
)

dev_agent.prompt_templates["managed_agent"]["task"] += """
Save Files and Data to the './agent_outputs' directory.

RESEARCH-FIRST DEVELOPMENT APPROACH:
- ALWAYS search for existing open-source implementations on GitHub before coding from scratch
- Look for established libraries, pre-trained models, and proven methodologies
- Use search_github_repositories, search_github_code, and multi_source_search tools
- Prioritize building upon and adapting existing solutions over reinventing
- Document the source and reasoning for chosen approaches"""

# Enable automatic memory recording for dev_agent
dev_agent = create_memory_enabled_agent(dev_agent, "dev_agent")


# Create tool creation agent for writing new tools
tool_creation_tools = [
    # OPTIMIZED: Core web and search tools - keep only most reliable
    extract_url_content,          # ✅ Specialized content extraction
    query_arxiv,
    query_scholar,
    query_pubmed,
    # Unified Search Tool for research and best practices
    multi_source_search,          # 🔍 Unified search tool - supports comprehensive research and best practices
    # GitHub tools for research and code examples
    search_github_repositories,
    search_github_code,
    get_github_repository_info,
    run_shell_command,
    create_conda_environment,
    install_packages_conda,
    install_packages_pip,
    check_gpu_status,
    create_and_run_script,
    run_script,
    create_requirements_file,
    monitor_training_logs,
]

tool_creation_agent = ToolCallingAgent(
    tools=tool_creation_tools,
    model=gpt_model,
    max_steps=25,  # Reasonable for tool creation tasks
    name="tool_creation_agent",
    description="""A specialized agent for creating new Python tools and utilities.
    
    Responsibilities:
    1. Write production-ready Python code for new tools
    2. Research best practices and existing solutions via web search and GitHub
    3. Test and validate tool functionality
    4. Create comprehensive documentation and error handling
    5. Follow proper software engineering practices
    
    Expertise areas:
    - Python programming and best practices
    - Tool architecture and design patterns
    - Error handling and input validation
    - Code testing and quality assurance
    - Integration with smolagents framework
    
    When creating tools:
    - Always save files to the ./new_tools/ directory
    - Use the @tool decorator from smolagents
    - Include comprehensive docstrings with Args and Returns
    - Add proper type hints for all parameters
    - Implement robust error handling
    - Test the tool after creation
    """,
)

tool_creation_agent.prompt_templates["managed_agent"]["task"] += """
You are a expert tool creator to write production-redady code for new tools.
Create all new tools in the './new_tools/' directory.
Use the @tool decorator from smolagents for all new tools.

MANDATORY CODE DISCOVERY PROCESS:
1. ALWAYS search GitHub repositories for existing implementations before writing new code
2. Search for academic papers and open-source projects on the topic
3. Look for established libraries, frameworks, and pre-trained models
4. Build upon existing solutions rather than creating from scratch
5. Use search_github_repositories, search_github_code, and multi_source_search tools

Research the best practices and existing solutions via web search and GitHub.
Always test your created tools to ensure they work correctly."""

# Enable automatic memory recording for tool_creation_agent
tool_creation_agent = create_memory_enabled_agent(tool_creation_agent, "tool_creation_agent")


# Create critic agent for intelligent evaluation (OPTIMIZED)
critic_tools = [
    extract_url_content,     # ✅ For reference materials - specialized extraction
    run_shell_command,       # For verification tasks
]

critic_agent = ToolCallingAgent(
    tools=critic_tools,  # ✅ Fixed: Added necessary tools
    model=gemini_model,
    max_steps=10,  # Reasonable for quality evaluation tasks
    name="critic_agent", 
    description="""Expert critic agent that evaluates task completion quality and determines if specialized tools are needed.
    
    Enhanced Responsibilities:
    1. Analyze task completion quality objectively with proper tools
    2. Identify gaps or areas for improvement through research
    3. Recommend specific specialized tools when beneficial
    4. Provide clear rationale for tool creation decisions
    5. Verify claims through web search and validation
    
    Evaluation criteria:
    - Task completion accuracy and completeness
    - Quality of output and analysis depth
    - Efficiency and methodology used
    - Potential for improvement with specialized tools
    - Comparison with industry best practices
    """
)

critic_agent.prompt_templates["managed_agent"]["task"] += """
You are a expert critic agent to evaluate task completion quality and determine if specialized tools are needed.
Analyze task completion quality objectively and identify gaps or areas for improvement.
Recommend specific specialized tools when beneficial.
Provide clear rationale for tool creation decisions.

CRITICAL PERFORMANCE STANDARDS:
- For ML/AI tasks, focus on ACTUAL PERFORMANCE METRICS, not just task completion
- Spearman correlation benchmarks: < 0.1 = POOR, 0.1-0.3 = NEEDS_IMPROVEMENT, 0.3-0.6 = SATISFACTORY, > 0.6 = EXCELLENT
- Never give high scores for low-performing models, even if they "run successfully"
- Always recommend iteration and optimization tools when performance is subpar
- Be especially critical of correlation scores near zero or negative values"""

# Enable automatic memory recording for critic_agent
critic_agent = create_memory_enabled_agent(critic_agent, "critic_agent")


# Manager agent will be created in main() function after loading custom prompts
manager_agent = None


# --- Launch Gradio Interface ---
def main():
    """Launch the Gradio interface for interactive agent communication with optional knowledge base."""
    global global_memory_manager, use_templates, gemini_model
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stella - Self-Evolving AI Assistant with Enhanced Memory")
    parser.add_argument("--use_template", action="store_true", default=True,
                       help="Enable knowledge base template usage for learning from past successes (default: True)")
    parser.add_argument("--no_template", action="store_true",
                       help="Disable template usage (overrides --use_template)")
    parser.add_argument("--use_mem0", action="store_true",
                       help="Enable Mem0 enhanced memory system for better semantic understanding")
    parser.add_argument("--use_default_prompts", action="store_true",
                       help="Force use of default smolagents prompts instead of custom prompts/code_agent.yaml")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run Gradio interface (default: 7860)")
    parser.add_argument("--enable_tool_creation", action="store_true", default=False,
                       help="Enable tool creation agent (default: True for full functionality)")
    
    args = parser.parse_args()
    
    # Set global template usage flag with override handling
    use_templates = args.use_template and not args.no_template
    
    # Handle tool creation flag with override
    enable_tool_creation = args.enable_tool_creation and not args.disable_tool_creation
    
    # Load custom prompt templates for manager_agent (default behavior)
    global custom_prompt_templates
    if args.use_default_prompts:
        print("📋 Using default smolagents prompts")
        custom_prompt_templates = None
    else:
        # 默认尝试加载自定义提示词
        try:
            prompt_templates_path = os.path.join(os.path.dirname(__file__), "prompts", "Stella_prompt_bioml.yaml")
            with open(prompt_templates_path, 'r', encoding='utf-8') as stream:
                custom_prompt_templates = yaml.safe_load(stream)
            print(f"✅ Custom prompts loaded: {prompt_templates_path}")
        except FileNotFoundError:
            print(f"📋 Custom prompts not found: {prompt_templates_path}")
            print("🔄 Using default smolagents prompts")
            custom_prompt_templates = None
        except Exception as e:
            print(f"⚠️ Error loading custom prompts: {str(e)}")
            print("🔄 Using default smolagents prompts")
            custom_prompt_templates = None
    
    # Build manager tools based on configuration
    manager_tool_management = build_manager_tools(enable_tool_creation=enable_tool_creation)
    
    # Build managed agents list based on configuration  
    managed_agents_list = [dev_agent, critic_agent]
    if enable_tool_creation:
        managed_agents_list.append(tool_creation_agent)
        print(f"🤖 Managed agents: dev_agent, critic_agent, tool_creation_agent")
    else:
        print(f"🤖 Managed agents: dev_agent, critic_agent (tool_creation_agent disabled)")
    
    # Create the manager agent AFTER loading custom prompts
    global manager_agent
    print("🚀 Creating manager agent with custom prompts...")
    print(f"📋 Available tools: {len(manager_tool_management)}")
    
    try:
        # 为自定义模板提供Jinja模板变量支持
        if custom_prompt_templates:
            # 渲染自定义模板，提供必要的模板变量
            from jinja2 import Template
            template_variables = {
                'code_block_opening_tag': '```python',
                'code_block_closing_tag': '```',
                'custom_instructions': '',  # 可以根据需要添加自定义指令
                'authorized_imports': ', '.join([
                    "time", "datetime", "os", "sys", "json", "csv", "pickle", "pathlib",
                    "math", "statistics", "random", "numpy", "pandas",
                    "collections", "itertools", "functools", "operator",
                    "typing", "dataclasses", "enum", "xml", "xml.etree", "xml.etree.ElementTree",
                    "requests", "urllib", "urllib.parse", "http", "re", "unicodedata", "string"
                ]),
                'managed_agents': {agent.name if hasattr(agent, 'name') else f'agent_{i}': agent for i, agent in enumerate(managed_agents_list)},
                'tools': {tool.name if hasattr(tool, 'name') else str(tool): tool for tool in manager_tool_management}
            }
            
            # 渲染模板
            rendered_templates = {}
            for key, template_content in custom_prompt_templates.items():
                if isinstance(template_content, str):
                    template = Template(template_content)
                    rendered_templates[key] = template.render(**template_variables)
                elif isinstance(template_content, dict):
                    # 对于嵌套的模板字典，递归渲染
                    rendered_sub_templates = {}
                    for sub_key, sub_content in template_content.items():
                        if isinstance(sub_content, str):
                            template = Template(sub_content)
                            rendered_sub_templates[sub_key] = template.render(**template_variables)
                        else:
                            rendered_sub_templates[sub_key] = sub_content
                    rendered_templates[key] = rendered_sub_templates
                else:
                    rendered_templates[key] = template_content
            
            print("✅ Custom prompt templates rendered with Jinja variables")
            manager_agent = CodeAgent(
                tools=manager_tool_management,  # 使用完整的工具管理权限
                model=gpt_model,
                managed_agents=managed_agents_list,
                additional_authorized_imports=[
        # 基于环境检测的已安装模块
        "sys", "os", "subprocess", "pathlib", "shutil", "glob", "fnmatch", "tempfile",
        "io", "json", "csv", "pickle", "sqlite3", "gzip", "zipfile", "tarfile",
        "math", "statistics", "random", "decimal", "fractions", "cmath",
        "time", "datetime", "calendar", "locale", "re", "string", "unicodedata", 
        "difflib", "textwrap", "collections", "itertools", "functools", "operator", 
        "heapq", "bisect", "queue", "threading", "multiprocessing", "concurrent", 
        "asyncio", "typing", "dataclasses", "enum", "abc", "contextlib", "inspect", 
        "argparse", "configparser", "logging", "warnings", "traceback", "xml", 
        "html", "urllib", "http", "socket", "ssl", "ftplib", "email", "numpy", 
        "pandas", "matplotlib", "scipy", "sklearn", "requests", "Bio", "yaml", 
        "tqdm", "joblib", "torch", "torchvision", "huggingface_hub", "seaborn", 
        "plotly", "PIL", "rdkit", "statsmodels",
    ],
                executor_kwargs={
                    "additional_functions": {
                        "globals": lambda: {"__name__": "__main__"},  # Safe globals replacement
                    }
                },
                name="manager_agent", 
                description=f"""STELLA Manager Agent - Problem Analysis & Task Coordination Center

                Core Responsibilities:
                1. Problem Analysis - Deep understanding of user requirements
                2. Skill Retrieval - Find reusable skills (retrieve_similar_skills)
                3. Strategic Planning - Plan solution strategies with skill workflows
                4. Tool Management - Load tools (analyze_query_and_load_relevant_tools), check availability (check_tools_for_task)
                5. Task Delegation - Assign tasks to specialized agents
                6. Quality & Learning - Evaluate with critic, save successful runs as skills

                Search: multi_source_search, search_github_repositories, PubMed, ArXiv
                Skills: retrieve_similar_skills, save_successful_skill, check_tools_for_task, export_skill_environment
                Agents: dev_agent (code/env), critic_agent (evaluation) {'tool_creation_agent (new tools)' if enable_tool_creation else ''}
                Workflow: Analyze -> Retrieve Skills -> Load Tools -> Delegate -> Evaluate -> Save Skill
                """,
                prompt_templates=rendered_templates,  # Use rendered templates
            )
        else:
            # Use default templates
            manager_agent = CodeAgent(
                tools=manager_tool_management,  
                model=gpt_model,
                managed_agents=managed_agents_list,
                additional_authorized_imports=[
        # 基于环境检测的已安装模块
        "sys", "os", "subprocess", "pathlib", "shutil", "glob", "fnmatch", "tempfile",
        "io", "json", "csv", "pickle", "sqlite3", "gzip", "zipfile", "tarfile",
        "math", "statistics", "random", "decimal", "fractions", "cmath",
        "time", "datetime", "calendar", "locale", "re", "string", "unicodedata", 
        "difflib", "textwrap", "collections", "itertools", "functools", "operator", 
        "heapq", "bisect", "queue", "threading", "multiprocessing", "concurrent", 
        "asyncio", "typing", "dataclasses", "enum", "abc", "contextlib", "inspect", 
        "argparse", "configparser", "logging", "warnings", "traceback", "xml", 
        "html", "urllib", "http", "socket", "ssl", "ftplib", "email", "numpy", 
        "pandas", "matplotlib", "scipy", "sklearn", "requests", "Bio", "yaml", 
        "tqdm", "joblib", "torch", "torchvision", "huggingface_hub", "seaborn", 
        "plotly", "PIL", "rdkit", "statsmodels", "esm", "transformers", "posixpath"
    ],
                executor_kwargs={
                    "additional_functions": {
                        "globals": lambda: {"__name__": "__main__"},  # Safe globals replacement
                    }
                },
                name="manager_agent", 
                description="""The main coordinator agent with self-evolution capabilities and tool management.""",
            )

        # Enable automatic memory recording for manager_agent
        manager_agent = create_memory_enabled_agent(manager_agent, "manager_agent")
        
        # Debug: Verify manager agent is created
        print(f"✅ Manager agent created: {type(manager_agent).__name__}")
        print(f"🔧 Manager agent has {len(manager_agent.tools)} tools")
        
    except Exception as e:
        print(f"❌ Error creating manager agent: {e}")
        print("🔄 Creating basic manager agent without custom prompts...")
        manager_agent = CodeAgent(
            tools=manager_tool_management,
            model=gpt_model,
            managed_agents=managed_agents_list,
            executor_kwargs={
                "additional_functions": {
                    "globals": lambda: {"__name__": "__main__"},  # Safe globals replacement
                }
            },
            name="manager_agent",
            description="Basic manager agent"
        )
        manager_agent = create_memory_enabled_agent(manager_agent, "manager_agent")
    
    # Initialize Skill Management System
    global global_skill_manager
    if use_templates:
        print("🧠 Initializing Skill Management System...")
        try:
            # Create LLM call function for skill summarizer
            from new_tools.llm import simple_llm_call
            llm_call_fn = lambda prompt: simple_llm_call(prompt, model_name="gemini-3-pro")

            global_skill_manager = SkillManager(llm_call=llm_call_fn)
            print(f"✅ Skill system initialized: {len(global_skill_manager.store.list_all())} skills, "
                  f"{len(global_skill_manager.tool_index.list_all())} tools indexed")

            # Migrate old templates if they exist and skill store is empty
            old_kb_path = "/home/ubuntu/agent_outputs/agent_knowledge_base.json"
            if (os.path.exists(old_kb_path)
                and len(global_skill_manager.store.list_all()) <= 10):
                print("📋 Migrating old templates to skill system...")
                migrated = global_skill_manager.migrate_from_knowledge_base(old_kb_path)
                if migrated > 0:
                    print(f"✅ Migrated {migrated} templates to skills")

        except Exception as e:
            print(f"⚠️ Skill system initialization failed: {str(e)}")
            print("⚠️ Continuing without skill system")
            global_skill_manager = None
            use_templates = False
    else:
        print("📋 Skill system disabled, use --use_template to enable")

    print(f"   Prompt Templates: {'Custom' if custom_prompt_templates else 'Default'}")

    if use_templates and global_skill_manager:
        print(f"   Skill System: Active")
    
    # Final check before creating Gradio UI
    if manager_agent is None:
        print("❌ CRITICAL ERROR: manager_agent is None!")
        print("🔧 Creating emergency fallback manager agent...")
        manager_agent = CodeAgent(
            tools=manager_tool_management[:3],  # Use only first 3 tools to avoid issues
            model=gpt_model,
            managed_agents=[dev_agent, critic_agent],  # Minimal agents (no tool creation)
            executor_kwargs={
                "additional_functions": {
                    "globals": lambda: {"__name__": "__main__"},  # Safe globals replacement
                }
            },
            name="emergency_manager",
            description="Emergency fallback manager agent"
        )
        print(f"✅ Emergency manager created: {type(manager_agent)}")
    
    print(f"🚀 Creating Gradio UI with manager_agent: {type(manager_agent)}")
    
    # Create and launch the Gradio UI
    gradio_ui = GradioUI(agent=manager_agent)
    
    # Launch with settings based on arguments
    gradio_ui.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=args.port,
        share=True,            # Set to True if you want a public link
    )

# --- Initialize function for external usage ---
def initialize_stella(use_template=True, use_mem0=True, enable_tool_creation=True):
    """Initialize Stella without launching Gradio interface - for use by other UIs"""
    global global_memory_manager, use_templates, custom_prompt_templates, manager_agent, gemini_model
    
    use_templates = use_template
    
    # Load custom prompt templates
    if not use_template:
        custom_prompt_templates = None
    else:
        try:
            prompt_templates_path = os.path.join(os.path.dirname(__file__), "prompts", "Stella_prompt_bioml.yaml")
            with open(prompt_templates_path, 'r', encoding='utf-8') as stream:
                custom_prompt_templates = yaml.safe_load(stream)
            print(f"✅ Custom prompts loaded: {prompt_templates_path}")
        except FileNotFoundError:
            print(f"📋 Custom prompts not found")
            custom_prompt_templates = None
        except Exception as e:
            print(f"⚠️ Error loading custom prompts: {str(e)}")
            custom_prompt_templates = None
    
    # Build manager tools and agents list based on configuration
    manager_tool_management = build_manager_tools(enable_tool_creation=enable_tool_creation)
    
    managed_agents_list = [dev_agent, critic_agent]
    if enable_tool_creation:
        managed_agents_list.append(tool_creation_agent)
        print(f"🤖 Managed agents: dev_agent, critic_agent, tool_creation_agent")
    else:
        print(f"🤖 Managed agents: dev_agent, critic_agent (tool_creation_agent disabled)")
    
    # Create the manager agent
    print("🚀 Creating manager agent with custom prompts...")
    print(f"📋 Available tools: {len(manager_tool_management)}")
    
    try:
        if custom_prompt_templates:
            from jinja2 import Template
            template_variables = {
                'code_block_opening_tag': '```python',
                'code_block_closing_tag': '```',
                'custom_instructions': '',
                'authorized_imports': ', '.join([
                    "time", "datetime", "os", "sys", "json", "csv", "pickle", "pathlib",
                    "math", "statistics", "random", "numpy", "pandas",
                    "collections", "itertools", "functools", "operator",
                    "typing", "dataclasses", "enum", "xml", "xml.etree", "xml.etree.ElementTree",
                    "requests", "urllib", "urllib.parse", "http", "re", "unicodedata", "string"
                ]),
                'managed_agents': {agent.name if hasattr(agent, 'name') else f'agent_{i}': agent for i, agent in enumerate(managed_agents_list)},
                'tools': {tool.name if hasattr(tool, 'name') else str(tool): tool for tool in manager_tool_management}
            }
            
            # 渲染模板
            rendered_templates = {}
            for key, template_content in custom_prompt_templates.items():
                if isinstance(template_content, str):
                    template = Template(template_content)
                    rendered_templates[key] = template.render(**template_variables)
                elif isinstance(template_content, dict):
                    rendered_sub_templates = {}
                    for sub_key, sub_content in template_content.items():
                        if isinstance(sub_content, str):
                            template = Template(sub_content)
                            rendered_sub_templates[sub_key] = template.render(**template_variables)
                        else:
                            rendered_sub_templates[sub_key] = sub_content
                    rendered_templates[key] = rendered_sub_templates
                else:
                    rendered_templates[key] = template_content
            
            print("✅ Custom prompt templates rendered with Jinja variables")
            manager_agent = CodeAgent(
                tools=manager_tool_management,
                model=gpt_model,
                managed_agents=managed_agents_list,
                additional_authorized_imports=[
        # 基于环境检测的已安装模块
        "sys", "os", "subprocess", "pathlib", "shutil", "glob", "fnmatch", "tempfile",
        "io", "json", "csv", "pickle", "sqlite3", "gzip", "zipfile", "tarfile",
        "math", "statistics", "random", "decimal", "fractions", "cmath",
        "time", "datetime", "calendar", "locale", "re", "string", "unicodedata", 
        "difflib", "textwrap", "collections", "itertools", "functools", "operator", 
        "heapq", "bisect", "queue", "threading", "multiprocessing", "concurrent", 
        "asyncio", "typing", "dataclasses", "enum", "abc", "contextlib", "inspect", 
        "argparse", "configparser", "logging", "warnings", "traceback", "xml", 
        "html", "urllib", "http", "socket", "ssl", "ftplib", "email", "numpy", 
        "pandas", "matplotlib", "scipy", "sklearn", "requests", "Bio", "yaml", 
        "tqdm", "joblib", "torch", "torchvision", "huggingface_hub", "seaborn", 
        "plotly", "PIL", "rdkit", "statsmodels",
    ],
                executor_kwargs={
                    "additional_functions": {
                        "globals": lambda: {"__name__": "__main__"},  # Safe globals replacement
                    }
                },
                name="manager_agent", 
                description=f"""STELLA Manager Agent - Problem Analysis & Task Coordination Center

                Core Responsibilities:
                1. Problem Analysis - Deep understanding of user requirements
                2. Skill Retrieval - Find reusable skills (retrieve_similar_skills)
                3. Strategic Planning - Plan solution strategies with skill workflows
                4. Tool Management - Load tools (analyze_query_and_load_relevant_tools), check availability (check_tools_for_task)
                5. Task Delegation - Assign tasks to specialized agents
                6. Quality & Learning - Evaluate with critic, save successful runs as skills

                Search: multi_source_search, search_github_repositories, PubMed, ArXiv
                Skills: retrieve_similar_skills, save_successful_skill, check_tools_for_task, export_skill_environment
                Agents: dev_agent (code/env), critic_agent (evaluation) {'tool_creation_agent (new tools)' if enable_tool_creation else ''}
                Workflow: Analyze -> Retrieve Skills -> Load Tools -> Delegate -> Evaluate -> Save Skill
                """,
                prompt_templates=rendered_templates,
            )
        else:
            # Use default templates
            manager_agent = CodeAgent(
                tools=manager_tool_management,
                model=gpt_model,
                managed_agents=managed_agents_list,
                additional_authorized_imports=[
        # 基于环境检测的已安装模块
        "sys", "os", "subprocess", "pathlib", "shutil", "glob", "fnmatch", "tempfile",
        "io", "json", "csv", "pickle", "sqlite3", "gzip", "zipfile", "tarfile",
        "math", "statistics", "random", "decimal", "fractions", "cmath",
        "time", "datetime", "calendar", "locale", "re", "string", "unicodedata", 
        "difflib", "textwrap", "collections", "itertools", "functools", "operator", 
        "heapq", "bisect", "queue", "threading", "multiprocessing", "concurrent", 
        "asyncio", "typing", "dataclasses", "enum", "abc", "contextlib", "inspect", 
        "argparse", "configparser", "logging", "warnings", "traceback", "xml", 
        "html", "urllib", "http", "socket", "ssl", "ftplib", "email", "numpy", 
        "pandas", "matplotlib", "scipy", "sklearn", "requests", "Bio", "yaml", 
        "tqdm", "joblib", "torch", "torchvision", "huggingface_hub", "seaborn", 
        "plotly", "PIL", "rdkit", "statsmodels",
    ],
                executor_kwargs={
                    "additional_functions": {
                        "globals": lambda: {"__name__": "__main__"},  # Safe globals replacement
                    }
                },
                name="manager_agent", 
                description="""The main coordinator agent with self-evolution capabilities and tool management.""",
            )

        # Enable automatic memory recording
        manager_agent = create_memory_enabled_agent(manager_agent, "manager_agent")
        
        print(f"✅ Manager agent created: {type(manager_agent).__name__}")
        print(f"🔧 Manager agent has {len(manager_agent.tools)} tools")
        
    except Exception as e:
        print(f"❌ Error creating manager agent: {e}")
        manager_agent = None
        return False
    
    # Initialize Skill Management System
    global global_skill_manager
    if use_template:
        print("🧠 Initializing Skill Management System...")
        try:
            from new_tools.llm import simple_llm_call
            llm_call_fn = lambda prompt: simple_llm_call(prompt, model_name="gemini-3-pro")
            global_skill_manager = SkillManager(llm_call=llm_call_fn)
            print(f"✅ Skill system initialized: {len(global_skill_manager.store.list_all())} skills")
        except Exception as e:
            print(f"⚠️ Skill system initialization failed: {str(e)}")
            global_skill_manager = None

    return manager_agent

if __name__ == "__main__":
    main()
