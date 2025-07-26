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
    WebSearchTool,
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
PHOENIX_API_KEY = get_api_key("PHOENIX_API_KEY", required=False)
TAVILY_API_KEY = get_api_key("TAVILY_API_KEY", required=False)
GEMINI_API_KEY = get_api_key("GEMINI_API_KEY", required=False)

HTTP_REFERER_URL = "http://localhost:8000"  # Replace if you have a specific site
X_TITLE_APP_NAME = "My Smolagent Web Search System" # Replace with your app name

# --- Phoenix Configuration ---
# Set the Phoenix endpoint (assuming Phoenix is running on localhost:6006)
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"


# --- Import Knowledge Base System ---
from Knowledge_base import KnowledgeBase, Mem0EnhancedKnowledgeBase, MEM0_AVAILABLE
from memory_manager import MemoryManager



# Global memory manager instance (replaces global_knowledge_base)
global_memory_manager = None
use_templates = False  # Global flag for template usage

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
Evaluate task completion:

TASK: {task_description}
RESULT: {current_result[:500]}...  # Truncate long results
EXPECTED: {expected_outcome if expected_outcome else "Not specified"}

Provide brief evaluation:
1. status: EXCELLENT/SATISFACTORY/NEEDS_IMPROVEMENT/POOR
2. quality_score: 1-10
3. gaps: key missing areas (max 3)
4. should_create_tool: true/false
5. recommended_tool: if needed, name and purpose

Be concise.
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
        
        # Automatically load the created tool into the agents
        load_result = load_dynamic_tool(tool_name, add_to_agents=True)
        
        final_result = f"✅ Tool creation completed!\n\n{result}\n\n🔧 Tool '{tool_name}' has been registered in the dynamic tools registry.\n\n📦 Auto-loading result: {load_result}"
        
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
        import inspect
        
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
            # Find all functions decorated with @tool in the loaded module
            tool_functions = []
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, '__smolagents_tool__'):
                    tool_functions.append(obj)
            
            if tool_functions:
                # Add to dev_agent tools
                for tool_func in tool_functions:
                    if tool_func not in dev_agent.tools:
                        dev_agent.tools.append(tool_func)
                    if tool_func not in tool_creation_agent.tools:
                        tool_creation_agent.tools.append(tool_func)
                
                result += f"\n🔧 Added {len(tool_functions)} tool function(s) to dev_agent and tool_creation_agent"
            else:
                result += "\n⚠️ No @tool decorated functions found in the module"
        
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
            if call['tool_name'] not in manager_agent.tools:
                return f"❌ Tool '{call['tool_name']}' not found in loaded tools"
        
        def execute_single_tool(tool_call):
            """Execute a single tool call with timeout"""
            tool_name = tool_call['tool_name']
            args = tool_call['args']
            start_time = time.time()
            
            try:
                tool_func = manager_agent.tools[tool_name]
                result = tool_func(**args)
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
            'biosecurity_tools': os.path.join(script_dir, 'new_tools', 'biosecurity_tools.py')
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
            llm_response = json_llm_call(llm_prompt, "gemini-2.5-pro")
            
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
        for i, tool_data in enumerate(loaded_tools[:5], 1):
            tool_name = tool_data['name']
            try:
                # Get tool signature inline
                if tool_name in manager_agent.tools:
                    import inspect
                    tool_func = manager_agent.tools[tool_name]
                    sig = inspect.signature(tool_func)
                    
                    # Show COMPLETE signature with parameter types
                    params = []
                    for param_name, param in sig.parameters.items():
                        if param.annotation != inspect.Parameter.empty:
                            param_type = getattr(param.annotation, '__name__', str(param.annotation))
                            params.append(f"{param_name}: {param_type}")
                        else:
                            params.append(param_name)
                    
                    param_str = ", ".join(params) if params else "no params"
                    
                    result += f"  {i}. {tool_name}({param_str})\n"
                else:
                    result += f"  {i}. {tool_name} (signature unavailable)\n"
            except Exception:
                result += f"  {i}. {tool_name} (signature error)\n"
        
        if len(loaded_tools) > 5:
            result += f"  ... (+{len(loaded_tools)-5} more tools loaded)\n"
        
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
        if score > 0 and tool_count < 5:
            try:
                if tool_name in manager_agent.tools:
                    import inspect
                    tool_func = manager_agent.tools[tool_name]
                    sig = inspect.signature(tool_func)
                    
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
        sig = inspect.signature(tool_func)
        
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
        import inspect
        
        if module_name not in sys.modules:
            return f"❌ Module '{module_name}' not loaded. Use load_dynamic_tool first."
        
        module = sys.modules[module_name]
        
        if not hasattr(module, tool_function_name):
            return f"❌ Function '{tool_function_name}' not found in module '{module_name}'."
        
        tool_func = getattr(module, tool_function_name)
        
        # Check if it's a tool function
        if not hasattr(tool_func, '__smolagents_tool__'):
            return f"❌ Function '{tool_function_name}' is not decorated with @tool."
        
        # Add to agents if not already present
        added_to = []
        if tool_func not in dev_agent.tools:
            dev_agent.tools.append(tool_func)
            added_to.append("dev_agent")
        
        if tool_func not in tool_creation_agent.tools:
            tool_creation_agent.tools.append(tool_func)
            added_to.append("tool_creation_agent")
        
        if added_to:
            return f"✅ Tool '{tool_function_name}' added to: {', '.join(added_to)}"
        else:
            return f"ℹ️ Tool '{tool_function_name}' was already available in all agents."
        
    except Exception as e:
        return f"❌ Error adding tool to agents: {str(e)}"

# --- Knowledge Base Tools with Optimization ---
@tool
@lru_cache(maxsize=16)  # Cache template retrievals
def retrieve_similar_templates(task_description: str, top_k: int = 3, user_id: str = "default") -> str:
    """Retrieve similar problem-solving templates from the knowledge base.
    
    Optimized with caching and reduced output.
    
    Args:
        task_description: Description of the current task
        top_k: Number of similar templates to retrieve (default: 3)
        user_id: User ID for personalized memory retrieval (default: "default")
        
    Returns:
        List of similar templates with reasoning approaches
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    try:
        # 使用新的知识记忆组件
        result = global_memory_manager.knowledge.search_templates(task_description, top_k, user_id)
        if result["success"]:
            similar_templates = result["templates"]
        else:
            similar_templates = []
        
        if not similar_templates:
            return "📚 No similar templates found in knowledge base."
        
        # Optimized output - more concise
        result = f"📚 Found {len(similar_templates)} templates:\n"
        
        for i, template in enumerate(similar_templates, 1):
            similarity = template.get('similarity', 0.0)
            task = template.get('task', '')[:80]  # Reduced from 150
            result += f"{i}. {task}... (Sim: {similarity:.2f})\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error retrieving templates: {str(e)}"


@tool
def save_successful_template(task_description: str, reasoning_process: str, solution_outcome: str, domain: str = "general", user_id: str = "default") -> str:
    """Save a successful problem-solving approach to the knowledge base.
    
    Args:
        task_description: Description of the solved task
        reasoning_process: The reasoning process that led to success
        solution_outcome: The successful outcome achieved
        domain: Domain category (default: "general")
        user_id: User ID for personalized memory storage (default: "default")
        
    Returns:
        Status of the save operation
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    try:
        # 使用新的知识记忆组件保存模板
        result = global_memory_manager.knowledge.add_template(task_description, reasoning_process, solution_outcome, domain, user_id)
        
        if result.get("success", False):
            # 获取统计信息
            stats = global_memory_manager.knowledge.get_stats(user_id)
            total_templates = stats.get('total_templates', 0)
            return f"✅ Template saved! Total: {total_templates}"
        else:
            return f"❌ Failed to save template: {result.get('message', 'Unknown error')}"
        
    except Exception as e:
        return f"❌ Error saving template: {str(e)}"


@tool
def list_knowledge_base_status(user_id: str = "default") -> str:
    """Get status and statistics of the knowledge base.
    
    Optimized to return concise information.
    
    Args:
        user_id: User ID for personalized memory statistics (default: "default")
    
    Returns:
        Knowledge base status and statistics
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    try:
        # 获取知识记忆组件的统计信息
        stats = global_memory_manager.knowledge.get_stats(user_id)
        
        result = f"📚 Memory Status: {stats.get('backend', 'Unknown')} - {stats.get('total_templates', 0)} templates"
        
        return result
        
    except Exception as e:
        return f"❌ Error getting memory system status: {str(e)}"


@tool
def search_templates_by_keyword(keyword: str, user_id: str = "default", limit: int = 5) -> str:
    """Search templates in the knowledge base by keyword.
    
    Args:
        keyword: Keyword to search for in templates
        user_id: User ID for personalized search (default: "default")
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        Matching templates containing the keyword
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    try:
        # 使用知识记忆组件进行语义搜索
        result = global_memory_manager.knowledge.search_templates(keyword, limit, user_id)
        if result["success"]:
            matching_results = result["templates"]
        else:
            matching_results = []
            
        if not matching_results:
            return f"🔍 No memories found containing keyword '{keyword}'."
            
        result = f"🔍 Found {len(matching_results)} memories:\n"
        
        for i, memory_result in enumerate(matching_results, 1):
            memory_text = memory_result.get('memory', str(memory_result))
            result += f"{i}. {memory_text[:100]}...\n"
            
        return result
        
    except Exception as e:
        return f"❌ Error searching templates: {str(e)}"


# --- New Mem0-specific Tools ---
@tool
def get_user_memories(user_id: str = "default", limit: int = 10) -> str:
    """Get all memories for a specific user (Mem0 enhanced feature).
    
    Args:
        user_id: User ID to retrieve memories for (default: "default")
        limit: Maximum number of memories to display (default: 10)
        
    Returns:
        List of user memories
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'session') or global_memory_manager.session is None:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        memories = global_memory_manager.session.get_user_memories(user_id)
        
        if not memories:
            return f"📚 No memories found for user '{user_id}'."
        
        result = f"📚 Memories for user '{user_id}' ({min(len(memories), limit)} of {len(memories)}):\n"
        
        for i, memory in enumerate(memories[:limit], 1):
            memory_text = memory.get('memory', str(memory))[:150]  # Reduced from 300
            result += f"{i}. {memory_text}...\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error retrieving user memories: {str(e)}"


# --- User session and context management functions have been removed ---
# Focus on agent team collaboration and shared knowledge base instead

# --- Multi-Agent Collaboration Tools ---

@tool
def create_shared_workspace(workspace_id: str, task_description: str, participating_agents: str = "dev_agent,manager_agent,critic_agent") -> str:
    """Create a shared workspace for agent team collaboration.
    
    Args:
        workspace_id: Unique identifier for the workspace
        task_description: Description of the collaborative task
        participating_agents: Comma-separated list of agent names
        
    Returns:
        Workspace creation status and details
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable collaboration features."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'collaboration') or global_memory_manager.collaboration is None:
        return "❌ This feature requires enhanced memory system. Use --use_template to enable."
    
    try:
        agents_list = [agent.strip() for agent in participating_agents.split(',')]
        
        result = global_memory_manager.collaboration.create_shared_workspace(
            workspace_id=workspace_id,
            task_description=task_description,
            participating_agents=agents_list
        )
        
        if result["success"]:
            return f"✅ Workspace '{workspace_id}' created with {len(result['participating_agents'])} agents"
        else:
            return f"❌ Failed to create workspace: {result['message']}"
            
    except Exception as e:
        return f"❌ Error creating shared workspace: {str(e)}"


@tool
def add_workspace_memory(workspace_id: str, agent_name: str, content: str, memory_type: str = "observation") -> str:
    """Add memory/observation to a shared workspace.
    
    Args:
        workspace_id: ID of the target workspace
        agent_name: Name of the contributing agent  
        content: The observation, discovery, or result to share
        memory_type: Type of memory (observation, discovery, result, question)
        
    Returns:
        Status of memory addition
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'collaboration') or global_memory_manager.collaboration is None:
        return "❌ This feature requires enhanced memory system. Use --use_template to enable."
    
    try:
        result = global_memory_manager.collaboration.add_workspace_memory(
            workspace_id=workspace_id,
            agent_name=agent_name,
            content=content,
            memory_type=memory_type
        )
        
        if result["success"]:
            return f"✅ Memory added to workspace '{workspace_id}'"
        else:
            return f"❌ Failed to add workspace memory: {result['message']}"
            
    except Exception as e:
        return f"❌ Error adding workspace memory: {str(e)}"


@tool
def get_workspace_memories(workspace_id: str, memory_type: str = "all", limit: int = 10) -> str:
    """Retrieve memories from a shared workspace.
    
    Optimized to return concise information.
    
    Args:
        workspace_id: ID of the target workspace
        memory_type: Type filter (all, observation, discovery, result, question)
        limit: Maximum number of memories to retrieve
        
    Returns:
        Formatted list of workspace memories
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'collaboration') or global_memory_manager.collaboration is None:
        return "❌ This feature requires enhanced memory system. Use --use_template to enable."
    
    try:
        result = global_memory_manager.collaboration.get_workspace_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            limit=limit
        )
        
        if result["success"]:
            if not result["memories"]:
                return f"📭 No memories found in workspace '{workspace_id}'"
            
            output = f"🏢 Workspace '{workspace_id}' ({result['total_found']} memories):\n"
            
            for i, memory in enumerate(result["memories"], 1):
                metadata = memory.get('metadata', {})
                agent = metadata.get('agent_name', 'Unknown')
                mem_type = metadata.get('memory_type', 'unknown')
                content = memory.get('memory', str(memory))[:100]  # Reduced from 200
                
                output += f"{i}. [{mem_type}] {agent}: {content}...\n"
            
            return output
        else:
            return f"❌ Failed to retrieve workspace memories: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error retrieving workspace memories: {str(e)}"


@tool
def create_task_breakdown(task_id: str, main_task: str, subtasks: str, agent_assignments: str = "") -> str:
    """Create a task breakdown with tracking for complex collaborative tasks.
    
    Args:
        task_id: Unique identifier for the task
        main_task: Description of the main task
        subtasks: JSON array string of subtask descriptions  
        agent_assignments: JSON object string mapping subtask indices to agent names
        
    Returns:
        Task breakdown creation status
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'collaboration') or global_memory_manager.collaboration is None:
        return "❌ This feature requires enhanced memory system. Use --use_template to enable."
    
    try:
        import json
        
        # Parse subtasks
        try:
            subtasks_list = json.loads(subtasks)
        except json.JSONDecodeError:
            # Fallback: split by line or comma
            subtasks_list = [task.strip() for task in subtasks.replace('\n', ',').split(',') if task.strip()]
        
        # Parse agent assignments
        assignments_dict = {}
        if agent_assignments:
            try:
                assignments_dict = json.loads(agent_assignments)
            except json.JSONDecodeError:
                # Ignore malformed assignments
                pass
        
        result = global_memory_manager.collaboration.create_task_breakdown(
            task_id=task_id,
            main_task=main_task,
            subtasks=subtasks_list,
            agent_assignments=assignments_dict
        )
        
        if result["success"]:
            return f"✅ Task '{task_id}' created with {result['subtasks_created']} subtasks"
        else:
            return f"❌ Failed to create task breakdown: {result['message']}"
            
    except Exception as e:
        return f"❌ Error creating task breakdown: {str(e)}"


@tool  
def update_subtask_status(task_id: str, subtask_index: int, new_status: str, agent_name: str, progress_notes: str = "") -> str:
    """Update the status of a specific subtask.
    
    Args:
        task_id: ID of the parent task
        subtask_index: Index of the subtask (0-based)
        new_status: New status (pending, in_progress, completed, blocked)
        agent_name: Name of the agent updating the status
        progress_notes: Optional notes about the progress
        
    Returns:
        Status update confirmation
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'collaboration') or global_memory_manager.collaboration is None:
        return "❌ This feature requires enhanced memory system. Use --use_template to enable."
    
    try:
        result = global_memory_manager.collaboration.update_subtask_status(
            task_id=task_id,
            subtask_index=subtask_index,
            new_status=new_status,
            agent_name=agent_name,
            progress_notes=progress_notes
        )
        
        if result["success"]:
            return f"✅ Subtask #{subtask_index} updated to {new_status}"
        else:
            return f"❌ Failed to update subtask status: {result['message']}"
            
    except Exception as e:
        return f"❌ Error updating subtask status: {str(e)}"


@tool
def get_task_progress(task_id: str) -> str:
    """Get comprehensive progress overview for a collaborative task.
    
    Optimized to return concise progress information.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        Detailed progress report with statistics and recent updates
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'collaboration') or global_memory_manager.collaboration is None:
        return "❌ This feature requires enhanced memory system. Use --use_template to enable."
    
    try:
        result = global_memory_manager.collaboration.get_task_progress(task_id)
        
        if result["success"]:
            progress = result["progress"]
            
            if not progress.get("main_task"):
                return f"❌ Task '{task_id}' not found"
            
            # Concise output
            output = f"📊 Task '{task_id}': {progress['progress_percentage']}% complete\n"
            output += f"✅ {progress['completed']} / {progress['total_subtasks']} done"
            
            return output
        else:
            return f"❌ Failed to get task progress: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error getting task progress: {str(e)}"


@tool
def get_agent_contributions(agent_name: str) -> str:
    """Get statistics about an agent's contributions to team collaboration.
    
    Args:
        agent_name: Name of the agent to analyze
        
    Returns:
        Summary of the agent's collaboration statistics
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    if not hasattr(global_memory_manager, 'collaboration') or global_memory_manager.collaboration is None:
        return "❌ This feature requires enhanced memory system. Use --use_template to enable."
    
    try:
        result = global_memory_manager.collaboration.get_agent_contributions(agent_name)
        
        if result["success"]:
            contrib = result["contributions"]
            
            output = f"📊 {agent_name} contributions: "
            output += f"{contrib['total_contributions']} total "
            output += f"({contrib['discoveries_shared']} discoveries, "
            output += f"{contrib['workspace_contributions']} workspace, "
            output += f"{contrib['task_updates']} updates)"
            
            return output
        else:
            return f"❌ Failed to get agent contributions: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error getting agent contributions: {str(e)}"


claude_model = OpenAIServerModel(
    model_id="anthropic/claude-sonnet-4",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY_STRING,
)

# Create a more capable model for manager and critic agents
gemini_model = OpenAIServerModel(
    model_id="google/gemini-2.5-pro",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY_STRING,
    temperature=0.1,  # Lower temperature for more consistent analysis
)

grok_model = OpenAIServerModel(
    model_id="x-ai/grok-4",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY_STRING,
    temperature=0.1,  # Lower temperature for more consistent analysis
)


# --- MCP Server Configuration ---
def setup_mcp_tools():
    """为生物医学和科学研究设置MCP工具。"""
    mcp_tools = []
    
    # --- PubMed MCP Server (proven to work) ---
    try:
        pubmed_server_params = StdioServerParameters(
            command="uvx",
            args=["--quiet", "pubmedmcp@0.1.3"],
            env={"UV_PYTHON": "3.12", **os.environ},
        )
        
        print("🔬 正在连接PubMed MCP服务器...")
        pubmed_client = MCPClient(pubmed_server_params)
        pubmed_tools = pubmed_client.get_tools()
        mcp_tools.extend(pubmed_tools)
        print(f"✅ 成功连接PubMed MCP服务器，获得 {len(pubmed_tools)} 个工具")
        
    except Exception as e:
        print(f"⚠️ PubMed MCP服务器连接失败: {e}")
    
    return mcp_tools

mcp_tools = setup_mcp_tools()

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

# 为 manager_agent 定义完整工具管理权限 - OPTIMIZED
manager_tool_management = [
    analyze_query_and_load_relevant_tools,  # 🎯 智能工具检索和加载（集成工具签名）
    execute_tools_in_parallel,  # 🚀 并行工具执行（提高效率）
    evaluate_with_critic,     # 🎯 评估任务质量
    list_dynamic_tools,       # 📋 查看工具库
    create_new_tool,          # 🛠️ 决定创建新工具
    load_dynamic_tool,        # 📦 管理工具加载
    refresh_agent_tools,      # 🔄 系统级刷新
    add_tool_to_agents,       # ➕ 细粒度工具管理
    get_tool_signature,       # 🔍 获取工具完整签名
    # Simplified memory tools that actually get used
    auto_recall_experience,   # 🧠 智能回忆相似任务
    check_agent_performance,  # 📊 检查智能体性能
    quick_tool_stats,         # 🔧 快速工具统计
    # OPTIMIZED: Keep only most reliable web tools
    WebSearchTool(),          # ✅ Most reliable web search
    extract_url_content,      # ✅ Specialized content extraction
    query_arxiv,
    query_scholar,
    query_pubmed,
    extract_pdf_content,
    fetch_supplementary_info_from_doi,
    
    # Unified Search Tool (Simplified Integration)
    multi_source_search,          # 🔍 Unified search tool (default: google,serpapi) - replaces all other search tools
    
    # GitHub tools
    search_github_repositories,
    search_github_code,
    get_github_repository_info,
    
    # Knowledge Base Tools (Templates disabled by default)
    # retrieve_similar_templates,     # Disabled for simplified workflow
    # save_successful_template,       # Disabled for simplified workflow
    list_knowledge_base_status,
    # search_templates_by_keyword,    # Disabled for simplified workflow
    
    # Mem0-specific Tools
    get_user_memories,
    
    # Multi-Agent Collaboration Tools
    create_shared_workspace,
    add_workspace_memory,
    get_workspace_memories,
    create_task_breakdown,
    update_subtask_status,
    get_task_progress,
    get_agent_contributions,
]

# Create the web search and development agent (ToolCallingAgent)
base_tools = [
    # Core web and search tools - OPTIMIZED: Keep only most reliable tools
    WebSearchTool(),              # ✅ Most reliable web search (DuckDuckGo/Bing)
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
    create_script,
    run_script,
    create_requirements_file,
    monitor_training_logs,
    
    # Basic tool management for dev_agent (read-only + basic operations)
] + dev_tool_management

# Combine base tools with MCP tools
all_tools = base_tools + mcp_tools


dev_agent = ToolCallingAgent(
    tools=all_tools,
    model=gemini_model,
    max_steps=15,  # Reduced from 20 to improve performance
    name="dev_agent",
    description="""A specialist agent for code execution and environment management.
    It uses tools for complex tasks like creating conda environments or generating scripts from templates.
    For file operations, prefer using the create_file tool, but basic Python functions like open() are also available.
    Give it specific, self-contained coding tasks like 'Analyze this CSV and plot a histogram' or 'Install the `numpy` library and verify the installation.'""",
)

dev_agent.prompt_templates["managed_agent"]["task"] += """
Save Files and Data to the './agent_outputs' directory."""

# Enable automatic memory recording for dev_agent
dev_agent = create_memory_enabled_agent(dev_agent, "dev_agent")


# Create tool creation agent for writing new tools
tool_creation_tools = [
    # OPTIMIZED: Core web and search tools - keep only most reliable
    WebSearchTool(),              # ✅ Most reliable web search
    extract_url_content,          # ✅ Specialized content extraction
    query_arxiv,
    query_scholar,
    query_pubmed,
    extract_pdf_content,
    fetch_supplementary_info_from_doi,
    
    # Unified Search Tool for research and best practices
    multi_source_search,          # 🔍 Unified search tool - supports comprehensive research and best practices
    
    # GitHub tools for research and code examples
    search_github_repositories,
    search_github_code,
    get_github_repository_info,
    
    # Development environment tools
    run_shell_command,
    create_conda_environment,
    install_packages_conda,
    install_packages_pip,
    check_gpu_status,
    create_script,
    run_script,
    create_requirements_file,
    monitor_training_logs,
]

tool_creation_agent = ToolCallingAgent(
    tools=tool_creation_tools,
    model=claude_model,
    max_steps=20,  # Reduced from 25
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
Research the best practices and existing solutions via web search and GitHub.
Always test your created tools to ensure they work correctly."""

# Enable automatic memory recording for tool_creation_agent
tool_creation_agent = create_memory_enabled_agent(tool_creation_agent, "tool_creation_agent")


# Create critic agent for intelligent evaluation (OPTIMIZED)
critic_tools = [
    WebSearchTool(),         # ✅ For querying best practices - most reliable
    extract_url_content,     # ✅ For reference materials - specialized extraction
    run_shell_command,       # For verification tasks
]

critic_agent = ToolCallingAgent(
    tools=critic_tools,  # ✅ Fixed: Added necessary tools
    model=claude_model,
    max_steps=5,  # Reduced from 8
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
Provide clear rationale for tool creation decisions."""

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
    parser.add_argument("--use_template", action="store_true", 
                       help="Enable knowledge base template usage for learning from past successes")
    parser.add_argument("--use_mem0", action="store_true",
                       help="Enable Mem0 enhanced memory system for better semantic understanding")
    parser.add_argument("--use_default_prompts", action="store_true",
                       help="Force use of default smolagents prompts instead of custom prompts/code_agent.yaml")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run Gradio interface (default: 7860)")
    
    args = parser.parse_args()
    
    # Set global template usage flag
    use_templates = args.use_template
    
    # Load custom prompt templates for manager_agent (default behavior)
    global custom_prompt_templates
    if args.use_default_prompts:
        print("📋 Using default smolagents prompts")
        custom_prompt_templates = None
    else:
        # 默认尝试加载自定义提示词
        try:
            prompt_templates_path = os.path.join(os.path.dirname(__file__), "prompts", "Stella_prompt_modified.yaml")
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
    
    # Create the manager agent AFTER loading custom prompts
    global manager_agent
    print("🚀 Creating manager agent with custom prompts...")
    print(f"📋 Available tools: {len(manager_tool_management)}")
    print(f"🤖 Managed agents: dev_agent={type(dev_agent)}, critic_agent={type(critic_agent)}, tool_creation_agent={type(tool_creation_agent)}")
    
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
                'managed_agents': {
                    'dev_agent': dev_agent,
                    'critic_agent': critic_agent,
                    'tool_creation_agent': tool_creation_agent
                },
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
                model=grok_model,
                managed_agents=[dev_agent, critic_agent, tool_creation_agent],
                additional_authorized_imports=[
                    # Basic Python modules
                    "time", "datetime", "os", "sys", "json", "csv", "pickle", "pathlib",
                    # Math and science
                    "math", "statistics", "random", 
                    # Data science core (only if installed)
                    "numpy", "pandas",
                    # Collections and utilities
                    "collections", "itertools", "functools", "operator",
                    "typing", "dataclasses", "enum",
                    # File formats
                    "xml", "xml.etree", "xml.etree.ElementTree",
                    # Networking
                    "requests", "urllib", "urllib.parse", "http",
                    # Text processing
                    "re", "unicodedata", "string"
                ],
                name="manager_agent", 
                description="""STELLA - Self-Evolving Laboratory Assistant.

                🎯 SIMPLIFIED WORKFLOW (MANDATORY):
                1. Task Planning: Create detailed action plan with clear objectives
                2. Tool Preparation: Use analyze_query_and_load_relevant_tools() after planning (includes signatures)
                3. Execution: Use tools with exact parameter names; use execute_tools_in_parallel() for independent calls
                4. Quality Evaluation: Assess results with critic_agent
                5. Self-Evolution: Create new tools if needed (templates disabled)
                6. Knowledge Storage: Save successful approaches to memory (when enabled)
                
                🔍 UNIFIED SEARCH CAPABILITY:
                - multi_source_search: Unified search tool with flexible source combinations
                  • "google": Basic Google search (0.3s)
                  • "google,serpapi": Enhanced Google search (1-2s, DEFAULT)
                  • "google,knowledge": Deep research search (30s)
                  • "google,knowledge,serpapi": Full-featured search combination (45s)
                
                🤖 Available agents:
                - dev_agent: Code execution and environment management (with unified search)
                - critic_agent: Quality evaluation  
                - tool_creation_agent: New tool creation (with unified search)
                
                📋 60+ specialized tools available on-demand (PubMed, UniProt, ChEMBL, KEGG, etc.)
                💡 Template retrieval disabled for simplified workflow - focus on direct problem solving
                """,
                prompt_templates=rendered_templates,  # Use rendered templates
            )
        else:
            # Use default templates
            manager_agent = CodeAgent(
                tools=manager_tool_management,  
                model=grok_model,
                managed_agents=[dev_agent, critic_agent, tool_creation_agent],
                additional_authorized_imports=[
                    "time", "datetime", "os", "sys", "json", "csv", "pickle", "pathlib",
                    "math", "statistics", "random", "numpy", "pandas",
                    "collections", "itertools", "functools", "operator",
                    "typing", "dataclasses", "enum", "xml", "xml.etree", "xml.etree.ElementTree",
                    "requests", "urllib", "urllib.parse", "http", "re", "unicodedata", "string"
                ],
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
            model=grok_model,
            managed_agents=[dev_agent, critic_agent, tool_creation_agent],
            name="manager_agent",
            description="Basic manager agent"
        )
        manager_agent = create_memory_enabled_agent(manager_agent, "manager_agent")
    
    # Initialize knowledge base if templates are enabled
    if use_templates:
        print("📚 Initializing knowledge base...")
        try:
            # 初始化新的统一内存管理系统
            print("🧠 Initializing memory system...")
            global_memory_manager = MemoryManager(
                gemini_model=gemini_model,
                use_mem0=args.use_mem0,
                mem0_api_key=MEM0_API_KEY,
                openrouter_api_key=OPENROUTER_API_KEY_STRING
            )
            
            # 获取整体统计
            stats = global_memory_manager.get_overall_stats()
            print(f"📊 System status: Knowledge({stats['knowledge']['backend']}) | "
                  f"Collaboration({'enabled' if stats['collaboration_enabled'] else 'disabled'}) | "
                  f"Session({'enabled' if stats['session_enabled'] else 'disabled'})")
            
            print("🧠 Keyword extraction with Gemini enhancement enabled")
        except Exception as e:
            print(f"❌ Knowledge base initialization failed: {str(e)}")
            print("⚠️ Continuing without knowledge base")
            use_templates = False
    else:
        print("📋 Knowledge base disabled, use --use_template to enable")
        if args.use_mem0:
            print("💡 Mem0 option requires --use_template")
    
    print(f"   Prompt Templates: {'Custom' if custom_prompt_templates else 'Default'}")
    
    if use_templates:
        backend = "Mem0 Enhanced" if args.use_mem0 else "Traditional KB"
        print(f"   Memory System: {backend}")
    
    # Final check before creating Gradio UI
    if manager_agent is None:
        print("❌ CRITICAL ERROR: manager_agent is None!")
        print("🔧 Creating emergency fallback manager agent...")
        manager_agent = CodeAgent(
            tools=manager_tool_management[:3],  # Use only first 3 tools to avoid issues
            model=grok_model,
            managed_agents=[dev_agent],  # Minimal agents
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
def initialize_stella(use_template=True, use_mem0=True):
    """Initialize Stella without launching Gradio interface - for use by other UIs"""
    global global_memory_manager, use_templates, custom_prompt_templates, manager_agent, gemini_model
    
    use_templates = use_template
    
    # Load custom prompt templates
    if not use_template:
        custom_prompt_templates = None
    else:
        try:
            prompt_templates_path = os.path.join(os.path.dirname(__file__), "prompts", "Stella_prompt_modified.yaml")
            with open(prompt_templates_path, 'r', encoding='utf-8') as stream:
                custom_prompt_templates = yaml.safe_load(stream)
            print(f"✅ Custom prompts loaded: {prompt_templates_path}")
        except FileNotFoundError:
            print(f"📋 Custom prompts not found")
            custom_prompt_templates = None
        except Exception as e:
            print(f"⚠️ Error loading custom prompts: {str(e)}")
            custom_prompt_templates = None
    
    # Create the manager agent
    print("🚀 Creating manager agent with custom prompts...")
    print(f"📋 Available tools: {len(manager_tool_management)}")
    
    try:
        # 为自定义模板提供Jinja模板变量支持（initialize_stella版本）
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
                'managed_agents': {
                    'dev_agent': dev_agent,
                    'critic_agent': critic_agent,
                    'tool_creation_agent': tool_creation_agent
                },
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
                model=grok_model,
                managed_agents=[dev_agent, critic_agent, tool_creation_agent],
                additional_authorized_imports=[
                    "time", "datetime", "os", "sys", "json", "csv", "pickle", "pathlib",
                    "math", "statistics", "random", 
                    "numpy", "pandas",
                    "collections", "itertools", "functools", "operator",
                    "typing", "dataclasses", "enum",
                    "xml", "xml.etree", "xml.etree.ElementTree",
                    "requests", "urllib", "urllib.parse", "http",
                    "re", "unicodedata", "string"
                ],
                name="manager_agent", 
                description="""STELLA - Self-Evolving Laboratory Assistant with Simplified Workflow.

                🎯 SIMPLIFIED WORKFLOW (MANDATORY):
                1. Task Planning: Create detailed action plan with clear objectives
                2. Tool Preparation: Use analyze_query_and_load_relevant_tools() after planning (includes signatures)
                3. Execution: Use tools with exact parameter names; use execute_tools_in_parallel() for parallel calls
                4. Quality Evaluation: Assess results with critic_agent
                5. Self-Evolution: Create new tools if needed (templates disabled)
                6. Knowledge Storage: Save successful approaches to memory (when enabled)
                
                🔍 UNIFIED SEARCH CAPABILITY:
                - multi_source_search: Unified search tool with flexible source combinations
                  • "google": Basic Google search (0.3s)
                  • "google,serpapi": Enhanced Google search (1-2s, DEFAULT)
                  • "google,knowledge": Deep research search (30s)
                  • "google,knowledge,serpapi": Full-featured search combination (45s)
                
                🤖 Available agents:
                - dev_agent: Code execution and environment management (with unified search)
                - critic_agent: Quality evaluation  
                - tool_creation_agent: New tool creation (with unified search)
                
                📋 60+ specialized tools available on-demand (PubMed, UniProt, ChEMBL, KEGG, etc.)
                💡 Template retrieval disabled for simplified workflow - focus on direct problem solving
                """,
                prompt_templates=rendered_templates,
            )
        else:
            # Use default templates
            manager_agent = CodeAgent(
                tools=manager_tool_management,
                model=grok_model,
                managed_agents=[dev_agent, critic_agent, tool_creation_agent],
                additional_authorized_imports=[
                    "time", "datetime", "os", "sys", "json", "csv", "pickle", "pathlib",
                    "math", "statistics", "random", "numpy", "pandas",
                    "collections", "itertools", "functools", "operator",
                    "typing", "dataclasses", "enum", "xml", "xml.etree", "xml.etree.ElementTree",
                    "requests", "urllib", "urllib.parse", "http", "re", "unicodedata", "string"
                ],
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
    
    # Initialize memory system if requested
    if use_template:
        print("📚 Initializing knowledge base...")
        try:
            global_memory_manager = MemoryManager(
                gemini_model=gemini_model,
                use_mem0=use_mem0,
                mem0_api_key=MEM0_API_KEY,
                openrouter_api_key=OPENROUTER_API_KEY_STRING
            )
            print("✅ Memory system initialized")
        except Exception as e:
            print(f"❌ Memory system initialization failed: {str(e)}")
    
    return manager_agent is not None

if __name__ == "__main__":
    main()