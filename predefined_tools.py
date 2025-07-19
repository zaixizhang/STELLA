import requests
from markdownify import markdownify
import re
from pathlib import Path
import subprocess
from requests.exceptions import RequestException
from smolagents import tool

# --- Custom Tools ---
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def search_github_repositories(query: str, language: str = "", sort: str = "stars", order: str = "desc", per_page: int = 10) -> str:
    """Search GitHub repositories for packages, models, or projects.
    
    Args:
        query: Search query (e.g., "transformer model", "pytorch CNN", "machine learning")
        language: Programming language filter (e.g., "Python", "JavaScript")
        sort: Sort results by "stars", "forks", "updated", or "created"
        order: Order results "desc" or "asc"
        per_page: Number of results to return (max 100)
        
    Returns:
        Formatted list of repository information including name, description, stars, and URL
    """
    try:
        # GitHub API endpoint for repository search
        url = "https://api.github.com/search/repositories"
        
        # Build search query
        search_query = query
        if language:
            search_query += f" language:{language}"
            
        params = {
            "q": search_query,
            "sort": sort,
            "order": order,
            "per_page": min(per_page, 100)  # GitHub API limit
        }
        
        # Make request to GitHub API
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        repositories = data.get("items", [])
        
        if not repositories:
            return f"No repositories found for query: {query}"
        
        # Format results
        result = f"GitHub搜索结果 (查询: '{query}'):\n\n"
        
        for i, repo in enumerate(repositories, 1):
            name = repo.get("name", "N/A")
            full_name = repo.get("full_name", "N/A")
            description = repo.get("description", "No description available")
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            language_used = repo.get("language", "N/A")
            html_url = repo.get("html_url", "N/A")
            updated_at = repo.get("updated_at", "N/A")
            
            result += f"{i}. **{name}** ({full_name})\n"
            result += f"   描述: {description}\n"
            result += f"   语言: {language_used} | ⭐ {stars} | 🍴 {forks}\n"
            result += f"   更新时间: {updated_at[:10]}\n"
            result += f"   链接: {html_url}\n\n"
        
        result += f"总共找到 {data.get('total_count', 0)} 个仓库"
        return result
        
    except RequestException as e:
        return f"GitHub API请求失败: {str(e)}"
    except Exception as e:
        return f"搜索GitHub仓库时发生错误: {str(e)}"

@tool
def search_github_code(query: str, language: str = "", filename: str = "", extension: str = "", per_page: int = 10) -> str:
    """Search for code snippets in GitHub repositories.
    
    Args:
        query: Code search query (e.g., "def train_model", "class CNN")
        language: Programming language filter
        filename: Specific filename to search in
        extension: File extension filter (e.g., "py", "js")
        per_page: Number of results to return (max 100)
        
    Returns:
        Code search results with file paths, repository info, and code snippets
    """
    try:
        url = "https://api.github.com/search/code"
        
        # Build search query
        search_query = query
        if language:
            search_query += f" language:{language}"
        if filename:
            search_query += f" filename:{filename}"
        if extension:
            search_query += f" extension:{extension}"
            
        params = {
            "q": search_query,
            "per_page": min(per_page, 100)
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        code_items = data.get("items", [])
        
        if not code_items:
            return f"未找到相关代码: {query}"
        
        result = f"GitHub代码搜索结果 (查询: '{query}'):\n\n"
        
        for i, item in enumerate(code_items, 1):
            name = item.get("name", "N/A")
            path = item.get("path", "N/A")
            repo_name = item.get("repository", {}).get("full_name", "N/A")
            html_url = item.get("html_url", "N/A")
            
            result += f"{i}. **{name}**\n"
            result += f"   仓库: {repo_name}\n"
            result += f"   路径: {path}\n"
            result += f"   链接: {html_url}\n\n"
        
        result += f"总共找到 {data.get('total_count', 0)} 个代码文件"
        return result
        
    except RequestException as e:
        return f"GitHub代码搜索失败: {str(e)}"
    except Exception as e:
        return f"搜索GitHub代码时发生错误: {str(e)}"

@tool
def get_github_repository_info(repo_owner: str, repo_name: str) -> str:
    """Get detailed information about a specific GitHub repository.
    
    Args:
        repo_owner: Repository owner/organization name
        repo_name: Repository name
        
    Returns:
        Detailed repository information including README, releases, and installation instructions
    """
    try:
        # Get repository information
        repo_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        response = requests.get(repo_url)
        response.raise_for_status()
        
        repo_data = response.json()

        # Get README content
        readme_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/readme"
        try:
            readme_response = requests.get(readme_url)
            readme_response.raise_for_status()
            readme_data = readme_response.json()
            readme_content = requests.get(readme_data["download_url"]).text
        except:
            readme_content = "README不可用"
        
        # Get latest release
        releases_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
        try:
            release_response = requests.get(releases_url)
            release_response.raise_for_status()
            latest_release = release_response.json()
            release_info = f"最新版本: {latest_release.get('tag_name', 'N/A')} ({latest_release.get('published_at', 'N/A')[:10]})"
        except:
            release_info = "无可用版本信息"
        
        # Format result
        result = f"GitHub仓库详细信息: {repo_owner}/{repo_name}\n\n"
        result += f"描述: {repo_data.get('description', 'N/A')}\n"
        result += f"语言: {repo_data.get('language', 'N/A')}\n"
        result += f"⭐ Stars: {repo_data.get('stargazers_count', 0)}\n"
        result += f"🍴 Forks: {repo_data.get('forks_count', 0)}\n"
        result += f"📁 Size: {repo_data.get('size', 0)} KB\n"
        result += f"📅 创建时间: {repo_data.get('created_at', 'N/A')[:10]}\n"
        result += f"🔄 更新时间: {repo_data.get('updated_at', 'N/A')[:10]}\n"
        result += f"{release_info}\n"
        result += f"🔗 链接: {repo_data.get('html_url', 'N/A')}\n"
        result += f"📄 Clone URL: {repo_data.get('clone_url', 'N/A')}\n\n"
        
        # Add topics/tags if available
        topics = repo_data.get('topics', [])
        if topics:
            result += f"🏷️ 标签: {', '.join(topics)}\n\n"
        
        # Add README content (first 1000 characters)
        if readme_content and readme_content != "README不可用":
            result += "📖 README预览:\n"
            result += "=" * 50 + "\n"
            result += readme_content[:1000]
            if len(readme_content) > 1000:
                result += "\n... (截取前1000字符)"
            result += "\n" + "=" * 50 + "\n"
        
        return result
        
    except RequestException as e:
        return f"获取GitHub仓库信息失败: {str(e)}"
    except Exception as e:
        return f"处理GitHub仓库信息时发生错误: {str(e)}"

@tool
def run_shell_command(command: str, working_directory: str = None) -> str:
    """Execute a shell command and return the output.
    
    Args:
        command: The shell command to execute
        working_directory: Optional working directory for the command
        
    Returns:
        Command output or error message
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_directory,
            timeout=300  # 5 minute timeout
        )
        
        output = f"Command: {command}\n"
        output += f"Return code: {result.returncode}\n"
        output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
            
        return output
        
    except subprocess.TimeoutExpired:
        return f"Command timed out after 5 minutes: {command}"
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}"

@tool
def create_conda_environment(env_name: str, python_version: str = "3.9") -> str:
    """Create a new conda environment.
    
    Args:
        env_name: Name of the conda environment
        python_version: Python version for the environment (default: 3.9)
        
    Returns:
        Result of the conda environment creation
    """
    command = f"conda create -n {env_name} python={python_version} -y"
    return run_shell_command(command)

@tool
def install_packages_conda(env_name: str, packages: str) -> str:
    """Install packages in a conda environment.
    
    Args:
        env_name: Name of the conda environment
        packages: Space-separated list of packages to install
        
    Returns:
        Result of the package installation
    """
    command = f"conda activate {env_name} && conda install {packages} -y"
    return run_shell_command(command)

@tool
def install_packages_pip(env_name: str, packages: str) -> str:
    """Install pip packages in a conda environment.
    
    Args:
        env_name: Name of the conda environment
        packages: Space-separated list of pip packages to install
        
    Returns:
        Result of the pip installation
    """
    command = f"conda activate {env_name} && pip install {packages}"
    return run_shell_command(command)

@tool
def check_gpu_status(dummy_param: str = "") -> str:
    """Check GPU status and availability using nvidia-smi.
    
    Args:
        dummy_param: Unused parameter to handle smolagents' automatic parameter passing
    
    Returns:
        GPU status information or error if no GPUs available
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return f"GPU Status:\n{result.stdout}"
        else:
            return f"Error checking GPU status: {result.stderr}"
            
    except FileNotFoundError:
        return "nvidia-smi not found. No NVIDIA GPUs available or drivers not installed."
    except Exception as e:
        return f"Error checking GPU status: {str(e)}"

@tool
def create_script(script_name: str, script_content: str, directory: str = ".", script_type: str = "python") -> str:
    """Create a script file.
    
    Args:
        script_name: Name of the script file (should include appropriate extension)
        script_content: Content of the script
        directory: Directory to create the script in (default: current directory)
        script_type: Type of script (python, bash, etc.) for informational purposes
        
    Returns:
        Result of script creation
    """
    try:
        script_path = Path(directory) / script_name
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make script executable if it's a bash script
        if script_name.endswith('.sh') or script_type.lower() in ['bash', 'shell']:
            import os
            os.chmod(script_path, 0o755)
            
        return f"Successfully created {script_type} script: {script_path}"
        
    except Exception as e:
        return f"Error creating script '{script_name}': {str(e)}"

@tool
def run_script(script_path: str, env_name: str = None, working_directory: str = None, interpreter: str = "python") -> str:
    """Run a script with optional conda environment activation.
    
    Args:
        script_path: Path to the script to run
        env_name: Name of the conda environment (optional)
        working_directory: Working directory for the script execution (optional)
        interpreter: Script interpreter (python, bash, etc.) - default: python
        
    Returns:
        Output from the script execution
    """
    # Determine the appropriate command based on file extension and interpreter
    script_name = Path(script_path).name
    
    if script_name.endswith('.sh') or interpreter.lower() in ['bash', 'shell']:
        base_command = f"bash {script_path}"
    elif script_name.endswith('.py') or interpreter.lower() == 'python':
        base_command = f"python {script_path}"
    else:
        # Try to run with specified interpreter
        base_command = f"{interpreter} {script_path}"
    
    # Add conda environment activation if specified
    if env_name:
        command = f"conda activate {env_name} && {base_command}"
    else:
        command = base_command
        
    return run_shell_command(command, working_directory)

@tool
def create_requirements_file(requirements: str, directory: str = ".") -> str:
    """Create a requirements.txt file.
    
    Args:
        requirements: Content of the requirements file (one package per line)
        directory: Directory to create the file in
        
    Returns:
        Result of file creation
    """
    try:
        req_path = Path(directory) / "requirements.txt"
        req_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write(requirements)
            
        return f"Successfully created requirements.txt: {req_path}"
        
    except Exception as e:
        return f"Error creating requirements.txt: {str(e)}"

@tool
def monitor_training_logs(log_file_path: str, lines: int = 50) -> str:
    """Monitor training logs by reading the last N lines of a log file.
    
    Args:
        log_file_path: Path to the log file
        lines: Number of lines to read from the end (default: 50)
        
    Returns:
        Last N lines of the log file
    """
    try:
        command = f"tail -n {lines} {log_file_path}"
        return run_shell_command(command)
        
    except Exception as e:
        return f"Error reading log file '{log_file_path}': {str(e)}"
