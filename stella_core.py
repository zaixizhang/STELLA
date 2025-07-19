import os
import re
import requests
import subprocess
import json
import time
import sys
from pathlib import Path
from markdownify import markdownify
from requests.exceptions import RequestException
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from predefined_tools import visit_webpage, search_github_repositories, search_github_code, get_github_repository_info, run_shell_command, create_conda_environment, install_packages_conda, install_packages_pip, check_gpu_status, create_script, run_script, create_requirements_file, monitor_training_logs
# from new_tools.virtual_screening_tools import *
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

# Mem0 integration for enhanced memory management
try:
    from mem0 import Memory, MemoryClient
    MEM0_AVAILABLE = True
    print("✅ Mem0 library available - enhanced memory features enabled")
except ImportError:
    MEM0_AVAILABLE = False
    print("⚠️ Mem0 library not installed - using traditional knowledge base")
    print("💡 Install with: pip install mem0ai")

OPENROUTER_API_KEY_STRING = "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"


openrouter_model_id = "anthropic/claude-sonnet-4"  # Example model, change as needed

HTTP_REFERER_URL = "http://localhost:8000"  # Replace if you have a specific site
X_TITLE_APP_NAME = "My Smolagent Web Search System" # Replace with your app name

# --- Phoenix Configuration ---
# Set the Phoenix endpoint (assuming Phoenix is running on localhost:6006)
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"


# --- Import Knowledge Base System ---
from Knowledge_base import KnowledgeBase, Mem0EnhancedKnowledgeBase, MEM0_AVAILABLE
from memory_manager import MemoryManager
# Mem0EnhancedKnowledgeBase class is now imported from Knowledge_base.py


# Global memory manager instance (replaces global_knowledge_base)
global_memory_manager = None
use_templates = False  # Global flag for template usage

# --- Self-Evolution Tools ---
# Global registry for dynamically created tools
dynamic_tools_registry = {}



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
        evaluation_prompt = f"""
Please evaluate the following task completion:

ORIGINAL TASK: {task_description}

CURRENT RESULT: {current_result}

EXPECTED OUTCOME: {expected_outcome if expected_outcome else "Not specified"}

Evaluate this completion and provide a JSON response with:
1. "status": "EXCELLENT" | "SATISFACTORY" | "NEEDS_IMPROVEMENT" | "POOR"
2. "quality_score": number from 1-10
3. "completion_assessment": detailed analysis of what was accomplished
4. "gaps_identified": list of specific areas lacking or could be improved
5. "should_create_tool": boolean - whether a specialized tool would significantly help
6. "recommended_tool": if should_create_tool is true, suggest specific tool details:
   - "tool_name": descriptive name
   - "tool_purpose": specific functionality needed
   - "tool_category": "analysis" | "visualization" | "data_processing" | "modeling"
7. "rationale": clear explanation of the recommendation

Focus on practical improvements that would meaningfully enhance future similar tasks.
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
    
    result = f"Dynamic Tools Registry ({len(dynamic_tools_registry)} tools):\n\n"
    
    for tool_name, tool_info in dynamic_tools_registry.items():
        result += f"🔧 {tool_name}\n"
        result += f"   Purpose: {tool_info['purpose']}\n"
        result += f"   Category: {tool_info['category']}\n"
        result += f"   Created: {tool_info['created_at']}\n\n"
    
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
6. Follow Python best practices and PEP 8 style guide
7. Include type hints for all function parameters and returns
8. Test the tool functionality after creation

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
def analyze_query_and_load_relevant_tools(user_query: str, max_tools: int = 10) -> str:
    """Analyze user query using LLM and intelligently load the most relevant tools from literature_tools.py, database_tools.py, and virtual_screening_tools.py.
    
    Args:
        user_query: The user's task description or query
        max_tools: Maximum number of relevant tools to load (default: 10)
        
    Returns:
        Status of the tool loading operation with analysis details
    """
    try:
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
            'literature_tools': os.path.join(script_dir, 'new_tools', 'literature_tools.py'),
            'database_tools': os.path.join(script_dir, 'new_tools', 'database_tools.py'),
            'virtual_screening_tools': os.path.join(script_dir, 'new_tools', 'virtual_screening_tools.py')
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
            return f"❌ No tools found in literature_tools.py, database_tools.py, or virtual_screening_tools.py"
        
        # Create tool list for LLM analysis
        tool_list = []
        for tool_name, tool_info in available_tools.items():
            tool_list.append({
                "name": tool_name,
                "description": tool_info['description'],
                "module": tool_info['module']
            })
        
        # Create LLM prompt for intelligent tool selection
        llm_prompt = f"""You are an expert AI assistant that helps select the most relevant biomedical research tools based on user queries.

TASK: Analyze the user query and select the most relevant tools from the available tool library.

USER QUERY: "{user_query}"

AVAILABLE TOOLS ({len(tool_list)} total):
{chr(10).join([f"{i+1}. {tool['name']} [{tool['module']}]: {tool['description']}" for i, tool in enumerate(tool_list)])}

INSTRUCTIONS:
1. Carefully analyze the user query to understand the research task
2. Select up to {max_tools} most relevant tools that would help accomplish this task
3. Consider tools from different categories when appropriate (literature search, database queries, analysis)
4. Prioritize tools that directly match the query requirements
5. For Chinese queries, consider both Chinese terms and their English equivalents

Please respond with a JSON object in this exact format:
{{
    "selected_tools": [
        {{
            "name": "tool_name",
            "relevance_score": 0.95,
            "reasoning": "Why this tool is relevant to the query"
        }}
    ],
    "analysis": "Brief analysis of the query and tool selection strategy"
}}

Select tools with relevance_score between 0.0-1.0, ordered by relevance."""

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
                    loaded_count += 1
                
                # Add to tool_creation_agent tools if not already present  
                if tool_name not in tool_creation_agent.tools:
                    tool_creation_agent.tools[tool_name] = tool_func
                
                loaded_tools.append({
                    'name': tool_name,
                    'relevance': tool_selection.get("relevance_score", 0.0),
                    'reasoning': tool_selection.get("reasoning", ""),
                    'description': tool_info['description'][:100] + "..." if len(tool_info['description']) > 100 else tool_info['description'],
                    'module': tool_info['module']
                })
                
            except Exception as e:
                continue
        
        # Generate result summary
        result = f"🎯 LLM Analysis: '{user_query}'\n\n"
        result += f"🤖 {llm_response.get('analysis', 'Intelligent tool selection completed')}\n\n"
        result += f"🔍 Found {len(available_tools)} total tools, selected {len(loaded_tools)} most relevant:\n\n"
        
        for i, tool in enumerate(loaded_tools, 1):
            relevance_bar = "█" * int(tool['relevance'] * 10)
            result += f"{i:2}. 🛠️ {tool['name']} [{tool['module']}]\n"
            result += f"     📊 Relevance: {relevance_bar} ({tool['relevance']:.3f})\n"
            result += f"     🎯 Reasoning: {tool['reasoning']}\n"
            result += f"     📝 {tool['description']}\n\n"
        
        result += f"✅ Successfully loaded {loaded_count} new tools into manager_agent and tool_creation_agent\n"
        result += f"🎯 Ready to execute task with LLM-selected domain-specific tools!"
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing query and loading tools: {str(e)}"

def _fallback_tool_selection(user_query: str, available_tools: dict, max_tools: int) -> str:
    """Fallback tool selection using simple keyword matching when LLM fails"""
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
            if tool_func not in manager_agent.tools:
                manager_agent.tools.append(tool_func)
                loaded_count += 1
            if tool_func not in tool_creation_agent.tools:
                tool_creation_agent.tools.append(tool_func)
    
    return f"🎯 Fallback Analysis: '{user_query}'\n✅ Loaded {loaded_count} tools using keyword matching."


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

# --- Knowledge Base Tools ---
@tool
def retrieve_similar_templates(task_description: str, top_k: int = 3, user_id: str = "default") -> str:
    """Retrieve similar problem-solving templates from the knowledge base.
    
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
        
        result = f"📚 Found {len(similar_templates)} similar templates:\n\n"
        
        for i, template in enumerate(similar_templates, 1):
            similarity = template.get('similarity', 0.0)
            result += f"🔍 Template {i} (Similarity: {similarity:.2f}):\n"
            
            # 处理不同的数据格式
            task = template.get('task', '')[:150]
            if not task and 'memory' in template:
                task = template.get('memory', '')[:150]
            
            result += f"   Task: {task}...\n"
            result += f"   Domain: {template.get('domain', 'unknown')}\n"
            result += f"   Key Reasoning: {template.get('key_reasoning', 'N/A')}\n"
            
            # 处理关键词
            keywords = template.get('keywords', [])
            if isinstance(keywords, list):
                result += f"   Keywords: {', '.join(keywords)}\n"
            else:
                result += f"   Keywords: {keywords}\n"
                
            result += f"   Created: {template.get('timestamp', 'Unknown')}\n"
            
            # 显示memory_id（如果有）
            if 'memory_id' in template:
                result += f"   Memory ID: {template['memory_id']}\n"
            
            result += "\n"
        
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
            backend = stats.get('backend', 'Knowledge Memory')
            return f"✅ Successfully saved template to {backend}!\n📊 Total templates: {total_templates}"
        else:
            return f"❌ Failed to save template: {result.get('message', 'Unknown error')}"
        
    except Exception as e:
        return f"❌ Error saving template: {str(e)}"


@tool
def list_knowledge_base_status(user_id: str = "default") -> str:
    """Get status and statistics of the knowledge base.
    
    Args:
        user_id: User ID for personalized memory statistics (default: "default")
    
    Returns:
        Knowledge base status and statistics
    """
    global global_memory_manager, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_memory_manager is None:
        return "❌ Memory manager not initialized."
    
    try:
        # 获取知识记忆组件的统计信息
        stats = global_memory_manager.knowledge.get_stats(user_id)
        overall_stats = global_memory_manager.get_overall_stats()
        
        result = f"📚 Memory System Status:\n"
        result += f"   Knowledge Backend: {stats.get('backend', 'Unknown')}\n"
        result += f"   Total Templates: {stats.get('total_templates', 0)}\n"
        result += f"   User ID: {user_id}\n"
        result += f"   Collaboration Memory: {'✅' if overall_stats.get('collaboration_enabled', False) else '❌'}\n"
        result += f"   Session Memory: {'✅' if overall_stats.get('session_enabled', False) else '❌'}\n"
        
        # 检查是否有传统知识库作为后备
        if hasattr(global_memory_manager.knowledge, 'fallback_kb') and global_memory_manager.knowledge.fallback_kb:
            result += f"   Storage File: {global_memory_manager.knowledge.fallback_kb.knowledge_file}\n"
        
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
            
            result = f"🔍 Found {len(matching_results)} memories containing '{keyword}':\n\n"
            
            for i, memory_result in enumerate(matching_results, 1):
                result += f"📋 Memory {i}:\n"
                memory_text = memory_result.get('memory', str(memory_result))
                result += f"   Content: {memory_text[:200]}...\n"
                
                metadata = memory_result.get('metadata', {})
                if metadata:
                    result += f"   Domain: {metadata.get('domain', 'unknown')}\n"
                    result += f"   Created: {metadata.get('timestamp', 'Unknown')}\n"
                
                if 'id' in memory_result:
                    result += f"   Memory ID: {memory_result['id']}\n"
                
                result += "\n"
                
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
        
        result = f"📚 Memories for user '{user_id}' (showing {min(len(memories), limit)} of {len(memories)}):\n\n"
        
        for i, memory in enumerate(memories[:limit], 1):
            result += f"🧠 Memory {i}:\n"
            memory_text = memory.get('memory', str(memory))
            result += f"   Content: {memory_text[:300]}...\n"
            
            metadata = memory.get('metadata', {})
            if metadata:
                result += f"   Domain: {metadata.get('domain', 'unknown')}\n"
                result += f"   Task Type: {metadata.get('task_type', 'unknown')}\n"
                result += f"   Created: {metadata.get('timestamp', 'Unknown')}\n"
            
            if 'id' in memory:
                result += f"   Memory ID: {memory['id']}\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error retrieving user memories: {str(e)}"


@tool
def delete_memory_by_id(memory_id: str, user_id: str = "default") -> str:
    """Delete a specific memory by ID (Mem0 enhanced feature).
    
    Args:
        memory_id: ID of the memory to delete
        user_id: User ID who owns the memory (default: "default")
        
    Returns:
        Status of the delete operation
    """
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        success = global_knowledge_base.delete_memory(memory_id, user_id)
        
        if success:
            return f"✅ Successfully deleted memory '{memory_id}' for user '{user_id}'."
        else:
            return f"❌ Failed to delete memory '{memory_id}'. Memory may not exist or belong to another user."
        
    except Exception as e:
        return f"❌ Error deleting memory: {str(e)}"


@tool
def update_memory_by_id(memory_id: str, new_content: str) -> str:
    """Update a specific memory by ID (Mem0 enhanced feature).
    
    Args:
        memory_id: ID of the memory to update
        new_content: New content for the memory
        
    Returns:
        Status of the update operation
    """
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        result = global_knowledge_base.update_memory(memory_id, new_content)
        
        if result:
            return f"✅ Successfully updated memory '{memory_id}' with new content."
        else:
            return f"❌ Failed to update memory '{memory_id}'. Memory may not exist."
        
    except Exception as e:
        return f"❌ Error updating memory: {str(e)}"


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
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable collaboration features."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        agents_list = [agent.strip() for agent in participating_agents.split(',')]
        
        result = global_knowledge_base.create_shared_workspace(
            workspace_id=workspace_id,
            task_description=task_description,
            participating_agents=agents_list
        )
        
        if result["success"]:
            return f"""✅ Shared workspace created successfully!
🏢 Workspace ID: {workspace_id}
📋 Task: {task_description}
🤖 Participating agents: {', '.join(result['participating_agents'])}
🆔 Memory ID: {result['memory_id']}

🔧 Usage:
- Use add_workspace_memory() to contribute observations and findings
- Use get_workspace_memories() to review team progress
- All agents can access and contribute to this shared space"""
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
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        result = global_knowledge_base.add_workspace_memory(
            workspace_id=workspace_id,
            agent_name=agent_name,
            content=content,
            memory_type=memory_type
        )
        
        if result["success"]:
            return f"""✅ Memory added to workspace successfully!
🏢 Workspace: {workspace_id}
🤖 Agent: {agent_name}
📝 Type: {memory_type}
🆔 Memory ID: {result['memory_id']}
💭 Content preview: {content[:100]}{'...' if len(content) > 100 else ''}"""
        else:
            return f"❌ Failed to add workspace memory: {result['message']}"
            
    except Exception as e:
        return f"❌ Error adding workspace memory: {str(e)}"


@tool
def get_workspace_memories(workspace_id: str, memory_type: str = "all", limit: int = 10) -> str:
    """Retrieve memories from a shared workspace.
    
    Args:
        workspace_id: ID of the target workspace
        memory_type: Type filter (all, observation, discovery, result, question)
        limit: Maximum number of memories to retrieve
        
    Returns:
        Formatted list of workspace memories
    """
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        result = global_knowledge_base.get_workspace_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            limit=limit
        )
        
        if result["success"]:
            if not result["memories"]:
                return f"📭 No memories found in workspace '{workspace_id}'"
            
            output = f"🏢 Workspace '{workspace_id}' memories ({result['total_found']} found):\n\n"
            
            for i, memory in enumerate(result["memories"], 1):
                metadata = memory.get('metadata', {})
                agent = metadata.get('agent_name', 'Unknown')
                mem_type = metadata.get('memory_type', 'unknown')
                timestamp = metadata.get('timestamp', '')
                content = memory.get('memory', str(memory))
                
                output += f"💭 Memory {i} [{mem_type}] by {agent} ({timestamp}):\n"
                output += f"   {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
            
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
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
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
        
        result = global_knowledge_base.create_task_breakdown(
            task_id=task_id,
            main_task=main_task,
            subtasks=subtasks_list,
            agent_assignments=assignments_dict
        )
        
        if result["success"]:
            output = f"""✅ Task breakdown created successfully!
📋 Task ID: {task_id}
🎯 Main Task: {main_task}
📊 Subtasks: {result['subtasks_created']} created
🆔 Memory ID: {result['memory_id']}

📝 Subtasks breakdown:"""
            
            for i, subtask in enumerate(subtasks_list):
                assigned_agent = assignments_dict.get(str(i), "unassigned")
                output += f"\n   {i+1}. {subtask} [→ {assigned_agent}]"
            
            output += f"\n\n🔧 Usage:\n- Use update_subtask_status() to update progress\n- Use get_task_progress() to check overall status"
            
            return output
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
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        result = global_knowledge_base.update_subtask_status(
            task_id=task_id,
            subtask_index=subtask_index,
            new_status=new_status,
            agent_name=agent_name,
            progress_notes=progress_notes
        )
        
        if result["success"]:
            output = f"""✅ Subtask status updated successfully!
📋 Task: {task_id}
📝 Subtask: #{subtask_index}
📊 Status: {new_status}
🤖 Updated by: {agent_name}
🆔 Update ID: {result['memory_id']}"""
            
            if progress_notes:
                output += f"\n📝 Notes: {progress_notes}"
            
            return output
        else:
            return f"❌ Failed to update subtask status: {result['message']}"
            
    except Exception as e:
        return f"❌ Error updating subtask status: {str(e)}"


@tool
def get_task_progress(task_id: str) -> str:
    """Get comprehensive progress overview for a collaborative task.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        Detailed progress report with statistics and recent updates
    """
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        result = global_knowledge_base.get_task_progress(task_id)
        
        if result["success"]:
            progress = result["progress"]
            
            if not progress.get("main_task"):
                return f"❌ Task '{task_id}' not found"
            
            output = f"""📊 Task Progress Report: {task_id}
🎯 Main Task: {progress['main_task']}

📈 Overall Progress: {progress['progress_percentage']}%
📋 Total Subtasks: {progress['total_subtasks']}
✅ Completed: {progress['completed']}
🔄 In Progress: {progress['in_progress']}  
⏳ Pending: {progress['pending']}

🔄 Recent Updates:"""
            
            if progress['recent_updates']:
                for update in progress['recent_updates']:
                    status = update.get('new_status', 'unknown')
                    agent = update.get('updated_by', 'unknown')
                    timestamp = update.get('timestamp', '')
                    subtask_idx = update.get('subtask_index', '?')
                    notes = update.get('progress_notes', '')
                    
                    output += f"\n   🔸 Subtask #{subtask_idx}: {status} by {agent} ({timestamp})"
                    if notes:
                        output += f"\n      📝 {notes[:100]}{'...' if len(notes) > 100 else ''}"
            else:
                output += "\n   No recent updates"
            
            return output
        else:
            return f"❌ Failed to get task progress: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error getting task progress: {str(e)}"


@tool
def share_agent_discovery(agent_name: str, discovery_title: str, discovery_content: str, tags: str = "", related_task: str = "") -> str:
    """Share a discovery or valuable experience with the agent team.
    
    Args:
        agent_name: Name of the agent sharing the discovery
        discovery_title: Brief title of the discovery
        discovery_content: Detailed content of the discovery
        tags: Comma-separated tags for categorization
        related_task: Optional task ID this discovery relates to
        
    Returns:
        Discovery sharing confirmation
    """
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
        
        result = global_knowledge_base.share_discovery(
            agent_name=agent_name,
            discovery_title=discovery_title,
            discovery_content=discovery_content,
            tags=tags_list,
            related_task=related_task
        )
        
        if result["success"]:
            output = f"""✅ Discovery shared successfully!
🤖 Shared by: {agent_name}
🔍 Title: {discovery_title}
🆔 Discovery ID: {result['discovery_id']}"""
            
            if tags_list:
                output += f"\n🏷️ Tags: {', '.join(tags_list)}"
            
            if related_task:
                output += f"\n📋 Related Task: {related_task}"
            
            output += f"\n💭 Content preview: {discovery_content[:150]}{'...' if len(discovery_content) > 150 else ''}"
            output += f"\n\n🔧 Other agents can now search for this discovery using search_agent_discoveries()"
            
            return output
        else:
            return f"❌ Failed to share discovery: {result['message']}"
            
    except Exception as e:
        return f"❌ Error sharing discovery: {str(e)}"


@tool
def search_agent_discoveries(query: str = "", agent_name: str = "", tags: str = "", limit: int = 5) -> str:
    """Search for discoveries and experiences shared by other agents.
    
    Args:
        query: Search query for discovery content
        agent_name: Filter by specific agent name
        tags: Comma-separated tags to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of relevant discoveries with details and relevance scores
    """
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else None
        
        result = global_knowledge_base.search_discoveries(
            query=query,
            agent_name=agent_name,
            tags=tags_list,
            limit=limit
        )
        
        if result["success"]:
            discoveries = result["discoveries"]
            
            if not discoveries:
                search_desc = f"query '{query}'" if query else "your criteria"
                return f"🔍 No discoveries found matching {search_desc}"
            
            output = f"🔍 Found {len(discoveries)} relevant discoveries (total: {result['total_found']}):\n\n"
            
            for i, discovery in enumerate(discoveries, 1):
                output += f"🧠 Discovery {i} [Score: {discovery['relevance_score']:.3f}]\n"
                output += f"   🤖 Agent: {discovery['agent_name']}\n"
                output += f"   🔍 Title: {discovery['title']}\n"
                output += f"   🕒 When: {discovery['timestamp']}\n"
                
                if discovery['tags']:
                    output += f"   🏷️ Tags: {', '.join(discovery['tags'])}\n"
                
                if discovery['related_task']:
                    output += f"   📋 Task: {discovery['related_task']}\n"
                
                content = discovery['content']
                output += f"   💭 Content: {content[:200]}{'...' if len(content) > 200 else ''}\n"
                output += f"   🆔 ID: {discovery['discovery_id']}\n\n"
            
            return output
        else:
            return f"❌ Failed to search discoveries: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error searching discoveries: {str(e)}"


@tool
def get_agent_contributions(agent_name: str) -> str:
    """Get statistics about an agent's contributions to team collaboration.
    
    Args:
        agent_name: Name of the agent to analyze
        
    Returns:
        Summary of the agent's collaboration statistics
    """
    global global_knowledge_base, use_templates
    
    if not use_templates:
        return "📋 Template usage is disabled. Use --use_template to enable."
    
    if global_knowledge_base is None:
        return "❌ Knowledge base not initialized."
    
    if not hasattr(global_knowledge_base, 'mem0_enabled') or not global_knowledge_base.mem0_enabled:
        return "❌ This feature requires Mem0 enhanced memory system. Use --use_mem0 to enable."
    
    try:
        result = global_knowledge_base.get_agent_contributions(agent_name)
        
        if result["success"]:
            contrib = result["contributions"]
            
            output = f"""📊 Collaboration Statistics for {agent_name}:

🧠 Discoveries Shared: {contrib['discoveries_shared']}
🏢 Workspace Contributions: {contrib['workspace_contributions']}  
📋 Task Updates: {contrib['task_updates']}
📈 Total Contributions: {contrib['total_contributions']}

🎯 Contribution Breakdown:
   • Knowledge Sharing: {contrib['discoveries_shared']} discoveries
   • Team Collaboration: {contrib['workspace_contributions']} workspace memories
   • Project Management: {contrib['task_updates']} status updates

{"🌟 This agent is an active team contributor!" if contrib['total_contributions'] > 10 else "📝 This agent is building their collaboration history."}"""
            
            return output
        else:
            return f"❌ Failed to get agent contributions: {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"❌ Error getting agent contributions: {str(e)}"



# --- Initialize the Model ---
# The OpenAIServerModel can be used for any OpenAI-compatible API, including OpenRouter.

# Check if the placeholder key is still there and warn the user.
if "YOUR_ACTUAL_OPENROUTER_KEY_HERE" in OPENROUTER_API_KEY_STRING:
    print("🔴 WARNING: You are using a placeholder API key.")
    print("Please replace 'sk-or-v1-YOUR_ACTUAL_OPENROUTER_KEY_HERE' with your actual OpenRouter API key.")
    # You might want to exit here if the key isn't real, to prevent errors.
    # exit()


print("🔍 Phoenix telemetry disabled (to avoid connection errors)")
print("💡 To enable monitoring, start Phoenix server and uncomment telemetry lines")
print()
print("🤖 Model Configuration:")
print(f"   Dev Agent: {openrouter_model_id}")
print(f"   Manager & Critic: google/gemini-2.5-pro")

model = OpenAIServerModel(
    model_id=openrouter_model_id,
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

    # # --- Alternative ArXiv MCP Server (for scientific papers) ---
    # try:
    #     arxiv_server_params = StdioServerParameters(
    #         command="uvx",
    #         args=["--quiet", "mcp-simple-arxiv@latest"],
    #         env={"UV_PYTHON": "3.12", **os.environ},
    #     )
    #     print("📚 正在连接ArXiv MCP服务器...")
    #     arxiv_client = MCPClient(arxiv_server_params)
    #     arxiv_tools = arxiv_client.get_tools()
    #     mcp_tools.extend(arxiv_tools)
    #     print(f"✅ 成功连接ArXiv MCP服务器，获得 {len(arxiv_tools)} 个工具")
    # except Exception as e:
    #     print(f"⚠️ ArXiv MCP服务器连接失败: {e}")
    
    return mcp_tools

mcp_tools = setup_mcp_tools()

# --- Tool Management Permissions ---
# 为 dev_agent 定义基础工具管理权限
dev_tool_management = [
    list_dynamic_tools,       # ✅ 查看可用工具
    load_dynamic_tool,        # ✅ 加载需要的工具
    refresh_agent_tools,      # ✅ 刷新自己的工具
]

# 为 manager_agent 定义完整工具管理权限
manager_tool_management = [
    analyze_query_and_load_relevant_tools,  # 🎯 智能工具检索和加载
    evaluate_with_critic,     # 🎯 评估任务质量
    list_dynamic_tools,       # 📋 查看工具库
    create_new_tool,          # 🛠️ 决定创建新工具
    load_dynamic_tool,        # 📦 管理工具加载
    refresh_agent_tools,      # 🔄 系统级刷新
    add_tool_to_agents,       # ➕ 细粒度工具管理
    # Knowledge base tools (enhanced with Mem0 support for agent team collaboration)
    retrieve_similar_templates,    # 🧠 检索相似模板
    save_successful_template,      # 💾 保存成功模板
    list_knowledge_base_status,    # 📊 知识库状态
    search_templates_by_keyword,   # 🔍 关键词搜索模板
    # Mem0-specific enhanced tools (for agent collaboration memory)
    get_user_memories,            # 📚 获取智能体记忆
    delete_memory_by_id,          # 🗑️ 删除特定记忆
    update_memory_by_id,          # ✏️ 更新记忆内容
    # Multi-Agent Collaboration Tools (for agent team collaboration)
    create_shared_workspace,      # 📁 创建共享工作空间
    add_workspace_memory,         # 📝 添加共享记忆
    get_workspace_memories,      # 📚 查看共享记忆
    create_task_breakdown,       # �� 创建任务分解
    update_subtask_status,       # ✏️ 更新任务状态
    get_task_progress,           # 📊 获取任务进度
    share_agent_discovery,      # 🤖 分享发现
    search_agent_discoveries,    # 🔍 搜索发现
    get_agent_contributions,    # 📊 获取贡献统计
    # Note: User session management tools have been removed - focus on agent team collaboration
]

# Create the web search and development agent (ToolCallingAgent)
base_tools = [
    # Core web and search tools
    WebSearchTool(),
    visit_webpage,
    
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
    model=model,
    max_steps=20,  # 增加步数以支持自我进化
    name="dev_agent",
    description="""A specialist agent for code execution and environment management.
    It uses tools for complex tasks like creating conda environments or generating scripts from templates.
    For file operations, prefer using the create_file tool, but basic Python functions like open() are also available.
    Give it specific, self-contained coding tasks like 'Analyze this CSV and plot a histogram' or 'Install the `numpy` library and verify the installation.'""",
)

dev_agent.prompt_templates["managed_agent"]["task"] += """
Save Files and Data to the '/home/ubuntu/agent_outputs' directory."""


# Create tool creation agent for writing new tools
tool_creation_tools = [
    # Core web and search tools  
    WebSearchTool(),
    visit_webpage,
    
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
    model=model,
    max_steps=25,  # Allow more steps for complex tool creation
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


# Create critic agent for intelligent evaluation
critic_agent = ToolCallingAgent(
    tools=[],
    model=gemini_model,
    max_steps=3,
    name="critic_agent", 
    description="""Expert critic agent that evaluates task completion quality and determines if specialized tools are needed.
    
    Responsibilities:
    1. Analyze task completion quality objectively
    2. Identify gaps or areas for improvement
    3. Recommend specific specialized tools when beneficial
    4. Provide clear rationale for tool creation decisions
    
    Evaluation criteria:
    - Task completion accuracy and completeness
    - Quality of output and analysis depth
    - Efficiency and methodology used
    - Potential for improvement with specialized tools
    """
)

critic_agent.prompt_templates["managed_agent"]["task"] += """
You are a expert critic agent to evaluate task completion quality and determine if specialized tools are needed.
Analyze task completion quality objectively and identify gaps or areas for improvement.
Recommend specific specialized tools when beneficial.
Provide clear rationale for tool creation decisions."""


# Create the manager agent (CodeAgent) and Stella Manager
manager_agent = CodeAgent(
    tools=manager_tool_management,  # 使用完整的工具管理权限
    model=gemini_model,
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
    description="""The main coordinator agent with advanced self-evolution capabilities and FULL tool management authority. An Biomedical Expert Agent for problem solving with intelligent tool selection.

    🎯 CRITICAL FIRST STEP - TOOL PREPARATION:
    ALWAYS start by using analyze_query_and_load_relevant_tools(user_query) to:
    1. Analyze the user's task and identify domain-specific requirements
    2. Automatically load the top 10 most relevant tools from literature_tools.py and database_tools.py
    3. Ensure both manager_agent and tool_creation_agent have access to domain-specific tools
    4. This enables access to 30+ specialized biomedical and literature tools (PubMed, UniProt, arXiv, etc.)
    
    🧠 Strategic Responsibilities:
    1. Manages and delegates tasks to specialized agents (dev_agent, critic_agent, tool_creation_agent)
    2. Uses critic agent to intelligently evaluate task completion quality
    3. Makes strategic decisions on when to create new specialized tools
    4. Continuously improves system capabilities through dynamic tool generation
    5. Maintains complete oversight of the dynamic tools registry
    6. Leverages knowledge base for learning from past successful approaches
    
    🛠️ Tool Management Authority (FULL ACCESS):
    - analyze_query_and_load_relevant_tools: FIRST PRIORITY - Intelligent domain tool selection
    - evaluate_with_critic: Assess task completion quality
    - create_new_tool: Decide when to create new tools based on needs
    - load_dynamic_tool: Load and distribute tools to other agents  
    - refresh_agent_tools: System-wide tool refresh operations
    - add_tool_to_agents: Fine-grained tool distribution control
    - list_dynamic_tools: Monitor tool library status
    
    📚 Knowledge Base Authority (FULL ACCESS):
    - retrieve_similar_templates: Find similar problem-solving approaches
    - save_successful_template: Store successful reasoning patterns
    - list_knowledge_base_status: Monitor knowledge base statistics
    - search_templates_by_keyword: Search for specific approaches
    
    🤖 Available agents to delegate to:
    - dev_agent: Has basic tool discovery/loading capabilities for task execution
    - critic_agent: For objective task quality evaluation  
    - tool_creation_agent: For creating new specialized Python tools
    
    🎯 WORKFLOW - Always follow this sequence:
    1. **FIRST**: Use analyze_query_and_load_relevant_tools() to prepare domain-specific tools
    2. **THEN**: Delegate task execution to dev_agent with now-available specialized tools
    3. **EVALUATE**: Use critic_agent for post-task quality assessment
    4. **EVOLVE**: Make strategic tool creation decisions based on critic recommendations
    5. **LEARN**: Save successful approaches when knowledge base is enabled
    
    📋 Available Domain Tools (loaded dynamically based on query):
    - Literature: query_arxiv, query_pubmed, query_scholar, search_google, extract_pdf_content
    - Databases: query_uniprot, query_pdb, query_kegg, query_ensembl, query_stringdb, etc.
    - 30+ specialized biomedical and scientific research tools available on-demand
    """,
)

# Add intelligent tool selection instruction to manager_agent
if hasattr(manager_agent, 'system_prompt'):
    manager_agent.system_prompt += """

🎯 CRITICAL WORKFLOW INSTRUCTION:
For EVERY new user task, you MUST start by calling analyze_query_and_load_relevant_tools(user_query) to:
1. Analyze the task requirements and domain
2. Automatically load the most relevant specialized tools
3. Ensure optimal tool availability for task execution

This is MANDATORY before any task execution to maximize success rate with domain-specific tools."""


# --- Launch Gradio Interface ---
def main():
    """Launch the Gradio interface for interactive agent communication with optional knowledge base."""
    global global_memory_manager, use_templates
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stella - Self-Evolving AI Assistant with Enhanced Memory")
    parser.add_argument("--use_template", action="store_true", 
                       help="Enable knowledge base template usage for learning from past successes")
    parser.add_argument("--use_mem0", action="store_true",
                       help="Enable Mem0 enhanced memory system for better semantic understanding")
    parser.add_argument("--mem0_platform", action="store_true",
                       help="Use Mem0 managed platform instead of self-hosted (requires --mem0_api_key)")
    parser.add_argument("--mem0_api_key", type=str,
                       help="Mem0 API key for managed platform usage")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run Gradio interface (default: 7860)")
    
    args = parser.parse_args()
    
    # Set global template usage flag
    use_templates = args.use_template
    
    # Initialize knowledge base if templates are enabled
    if use_templates:
        print("📚 初始化知识库系统...")
        try:
            # 先初始化 Gemini 模型
            gemini_model = OpenAIServerModel(
                model_id="google/gemini-2.5-pro",
                api_base="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY_STRING,
                temperature=0.1,
            )
            
            # 初始化新的统一内存管理系统
            print("🧠 初始化统一内存管理系统...")
            global_memory_manager = MemoryManager(
                gemini_model=gemini_model,
                use_mem0=args.use_mem0,
                mem0_platform=args.mem0_platform,
                mem0_api_key=args.mem0_api_key,
                openrouter_api_key=OPENROUTER_API_KEY_STRING
            )
            
            # 获取整体统计
            stats = global_memory_manager.get_overall_stats()
            print(f"📊 系统状态: 知识记忆({stats['knowledge']['backend']}) | "
                  f"协作记忆({'启用' if stats['collaboration_enabled'] else '禁用'}) | "
                  f"会话记忆({'启用' if stats['session_enabled'] else '禁用'})")
            
            print("🧠 关键词提取功能已启用 Gemini 模型增强")
        except Exception as e:
            print(f"❌ 知识库初始化失败: {str(e)}")
            print("⚠️ 继续运行但知识库功能将不可用")
            use_templates = False
    else:
        print("📋 知识库功能已禁用，使用 --use_template 启用")
        if args.use_mem0:
            print("💡 Mem0 选项需要配合 --use_template 使用")
    
    print("🌟 Launching Stella - Self-Evolving AI Agent")
    print("=" * 80)
    print("🧠 Stella具备自我进化能力，可以:")
    print("   🎯 智能分析查询并自动加载最相关的专业工具")
    print("   📚 从60+生物医学和文献工具中动态选择最适合的工具")
    print("   ✨ 自动评估任务完成质量")
    print("   🛠️ 动态创建专业工具提升性能") 
    print("   🔄 持续学习和优化解决方案")
    print("   �� 多智能体协作完成复杂任务")
    print("")
    print("🔧 Agent权限分配:")
    print("   ✏️ Dev Agent (基础权限): edit_file, list_dir, read_file, run_terminal_cmd")
    print("   🎨 UI Agent (界面权限): 以上 + create_new_tool, load_dynamic_tool")  
    print("   🔍 Critic Agent (评估权限): 以上 + evaluate_with_critic")
    print("   🧠 Manager Agent (完整权限): 以上 + analyze_query_and_load_relevant_tools (核心)")
    print("   🎯 Manager Agent (进化权限): evaluate_with_critic, create_new_tool, add_tool_to_agents")
    print("   📚 可用专业工具: 60+ (PubMed, UniProt, KEGG, arXiv, Google Scholar, etc.)")
    
    if use_templates:
        if args.use_mem0:
            print("   🧠 manager_agent: 具备完整Mem0增强记忆管理权限")
            print("   🤝 智能体团队支持: 协作记忆空间+任务追踪+知识传递")
        else:
            print("   📚 manager_agent: 同时具备完整知识库管理权限")
    
    print()
    print("🔬 生物医学任务示例:")
    print("   📚 文献检索: '搜索PubMed中关于CRISPR-Cas9在癌症治疗中的应用'")
    print("   🧬 分子生物学: '设计用于克隆特定基因的PCR引物'")
    print("   🧪 生化分析: '分析蛋白质序列的结构特征和分子量'")
    print("   🦠 微生物学: '分析细菌生长曲线数据'")
    print("   💊 药理学: '预测化合物的ADMET性质'")
    print("   🔬 数据库查询: '从UniProt获取特定蛋白质的详细信息'")
    print("   🧮 基因组学: '分析单细胞RNA-seq数据的基因表达模式'")
    print("   🏥 病理学: '量化组织学图像中的形态学特征'")
    print()
    print("💡 通用任务示例:")
    print("   🐙 GitHub搜索: '搜索最受欢迎的生物信息学Python包'")
    print("   🖥️ 环境管理: '创建生物信息学分析的conda环境'")
    print("   📊 数据可视化: '绘制基因表达数据的热图'")
    print()
    print("🎯 智能工具选择演示示例:")
    print("   📚 输入: '查找关于CRISPR-Cas9在癌症治疗中的最新研究'")
    print("   🔍 系统分析: 识别关键词 'CRISPR', 'cancer', 'research'")
    print("   🛠️ 自动加载: query_pubmed, query_arxiv, search_google, extract_pdf_content...")
    print("   ✅ 结果: Manager具备完整的文献检索工具链")
    print()
    print("🔧 Self-Evolution演示示例:")
    print("   1️⃣ 智能准备: 自动加载相关专业工具")
    print("   2️⃣ 正常任务: '分析这个蛋白质序列数据'")
    print("   3️⃣ 系统自动评估完成质量")
    print("   4️⃣ 如需改进，自动创建专用工具: 'advanced_protein_analyzer'")
    print("   5️⃣ 新工具自动集成到所有agent，立即可用")
    print("   6️⃣ 下次类似任务使用新工具，性能提升")
    
    if use_templates:
        print("   6️⃣ 成功方案自动保存到知识库")
        print("   7️⃣ 相似任务时自动检索历史经验")
    
    if use_templates and args.use_mem0:
        print()
        print("🤖 Multi-Agent协作演示示例:")
        print("   1️⃣ Manager创建工作空间: create_shared_workspace('biodata_analysis', '分析基因表达数据')")
        print("   2️⃣ Manager分解任务: create_task_breakdown('task001', '数据预处理+统计分析+可视化')")
        print("   3️⃣ Dev_agent添加发现: add_workspace_memory('biodata_analysis', 'dev_agent', '数据包含缺失值')")
        print("   4️⃣ Critic_agent更新状态: update_subtask_status('task001', 0, 'completed', 'critic_agent')")
        print("   5️⃣ 智能体分享经验: share_agent_discovery('dev_agent', '处理缺失值技巧', tags='data,preprocessing')")
        print("   6️⃣ 团队查看进度: get_task_progress('task001') 和 get_workspace_memories('biodata_analysis')")
        print("   7️⃣ 跨任务学习: search_agent_discoveries(query='缺失值', tags='preprocessing')")
    
    print()
    print("🛠️ 工具管理权限分离:")
    print("   📋 Dev Agent (基础权限): list_dynamic_tools, load_dynamic_tool, refresh_agent_tools")
    print("   🧠 Manager Agent (完整权限): 以上 + analyze_query_and_load_relevant_tools (核心)")
    print("   🎯 Manager Agent (进化权限): evaluate_with_critic, create_new_tool, add_tool_to_agents")
    print("   📚 可用专业工具: 60+ (PubMed, UniProt, KEGG, arXiv, Google Scholar, etc.)")
    
    if use_templates:
        if args.use_mem0:
            print("   🧠 Manager Agent (Mem0权限): retrieve_templates, save_templates, search_templates")
            print("   🤝 Manager Agent (协作权限): get_memories, update_memory, delete_memory (团队共享)")
            print("   📁 Manager Agent (工作空间): create_workspace, add_memory, get_memories")
            print("   📋 Manager Agent (任务追踪): create_breakdown, update_status, get_progress")
            print("   🔍 Manager Agent (知识传递): share_discovery, search_discoveries, get_contributions")
        else:
            print("   📚 Manager Agent (知识库权限): retrieve_templates, save_templates, search_templates")
    
    print()
    print("💡 智能工作流程:")
    if args.use_mem0 and use_templates:
        print("   0️⃣ 智能工具选择：分析查询 → 自动加载前10个最相关专业工具")
        print("   1️⃣ 智能记忆：语义检索用户偏好 → 个性化响应")
        print("   2️⃣ 任务执行：使用专业工具 → 高质量完成任务")
        print("   3️⃣ 任务评估：Manager评估质量 → 决定是否创建新工具")
        print("   4️⃣ 工具创建：自动生成专用工具 → 加载到所有Agent")
        print("   5️⃣ 记忆更新：保存成功经验 → 积累知识模式")
        print("   6️⃣ 上下文学习：检索相似经验 → 应用成功模式 → 保存新经验")
    else:
        print("   0️⃣ 智能工具选择：分析查询 → 自动加载前10个最相关专业工具")
        print("   1️⃣ 任务执行：Manager使用专业工具执行任务")
        print("   2️⃣ 质量评估：评估完成质量 → 决定是否创建新工具")
        print("   3️⃣ 工具进化：创建完成 → 自动加载到所有Agent")
        print("   4️⃣ 持续改进：系统级管理由Manager独家处理")
        if use_templates:
            print("   5️⃣ 知识库学习：检索相似经验 → 应用成功模式 → 保存新经验")
    
    print()
    print(f"🌐 启动参数:")
    print(f"   端口: {args.port}")
    print(f"   知识库: {'启用' if use_templates else '禁用'}")
    
    if use_templates:
        if args.use_mem0:
            print(f"   记忆系统: Mem0 增强记忆")
            print(f"   记忆模式: {'托管平台' if args.mem0_platform else '自托管'}")
            if args.mem0_platform and args.mem0_api_key:
                print(f"   API密钥: {args.mem0_api_key[:8]}...{args.mem0_api_key[-4:]}")
            
            # 获取记忆统计
            if hasattr(global_memory_manager, 'get_overall_stats'):
                stats = global_memory_manager.get_overall_stats()
                if stats.get('knowledge', {}).get('status') == 'enabled':
                    print(f"   记忆后端: {stats.get('backend', 'Unknown')}")
                    if 'total_memories' in stats:
                        print(f"   当前记忆数: {stats['total_memories']}")
                    else:
                        print(f"   当前模板数: {stats.get('total_templates', 0)}")
        else:
            print(f"   记忆系统: 传统知识库")
            if hasattr(global_memory_manager, 'knowledge') and hasattr(global_memory_manager.knowledge, 'fallback_kb'):
                kb = global_memory_manager.knowledge.fallback_kb
                if hasattr(kb, 'knowledge_file'):
                    print(f"   知识库文件: {kb.knowledge_file}")
                if hasattr(kb, 'templates'):
                    print(f"   已有模板数: {len(kb.templates)}")
    
    print()
    print("🚀 Mem0 增强功能说明:")
    if args.use_mem0 and use_templates:
        print("   ✅ 启用语义记忆搜索 - 基于内容相似性检索")
        print("   ✅ 启用用户个性化记忆 - 每用户独立记忆空间")
        print("   ✅ 启用会话上下文管理 - 对话连续性保持")
        print("   ✅ 启用偏好学习分析 - 自动学习用户偏好")
        print("   ✅ 启用智能记忆更新 - 自动去重和优化")
    else:
        print("   ❌ 使用 --use_template --use_mem0 启用Mem0增强功能")
        print("   💡 安装: pip install mem0ai")
        print("   🏠 自托管: 使用本地Chroma向量数据库")
        print("   ☁️  托管版: 使用 --mem0_platform --mem0_api_key")
    
    print("=" * 80)
    
    # Create and launch the Gradio UI
    gradio_ui = GradioUI(agent=manager_agent)
    
    # Launch with settings based on arguments
    gradio_ui.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
    )

if __name__ == "__main__":
    main() 
