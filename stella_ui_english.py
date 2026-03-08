"""
Stella AI Assistant - Professional English Interface
A bright, concise UI for AI-powered development assistance
"""

import gradio as gr
import os
import re
import time
import json
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import from stella_core.py
import stella_core

class StellaEnglishUI:
    def __init__(self):
        self.conversation_history = []
        self.created_files = []
        self.execution_steps = []
        self.current_execution = None
        
        # Ensure output directory exists
        self.output_dir = Path("/home/ubuntu/agent_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup debug mode - can be enabled via environment variable
        self.debug_mode = os.getenv('STELLA_DEBUG', 'false').lower() == 'true'
        
        print("🌟 Stella UI initialized")
        print(f"📁 Output directory: {self.output_dir}")
        
    def parse_agent_output(self, output: str) -> Dict:
        """Parse agent output to extract steps, tools, and results"""
        steps = []
        
        # More robust parsing to capture the exact terminal output format
        # Split by step markers with unicode box drawing characters
        step_pattern = r'━+\s*Step\s+(\d+)\s*━+'
        step_blocks = re.split(step_pattern, output)
        
        # Process each step block
        for i in range(1, len(step_blocks), 2):  # Skip first empty block, then take every other
            if i + 1 >= len(step_blocks):
                break
                
            step_number = int(step_blocks[i])
            block_content = step_blocks[i + 1] if i + 1 < len(step_blocks) else ""
            
            step_info = {
                'step_number': step_number,
                'tools': [],
                'observations': [],
                'content': [],
                'duration': None,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'raw_content': block_content
            }
            
            # Extract tool calls with box drawing characters
            tool_box_pattern = r'╭[─]*╮\n│\s*Calling tool:\s*\'([^\']+)\'\s*with arguments:\s*({[^}]*})[^│]*│\n╰[─]*╯'
            tool_matches = re.findall(tool_box_pattern, block_content, re.DOTALL)
            
            if not tool_matches:
                # Fallback to simpler pattern
                tool_simple_pattern = r'Calling tool:\s*\'([^\']+)\'\s*with arguments:\s*({[^}]*})'
                tool_matches = re.findall(tool_simple_pattern, block_content, re.DOTALL)
            
            for tool_name, args_str in tool_matches:
                try:
                    # Clean up the arguments string
                    args_str = args_str.strip()
                    args = json.loads(args_str)
                except:
                    # If JSON parsing fails, keep as string
                    args = args_str
                
                step_info['tools'].append({
                    'name': tool_name,
                    'arguments': args
                })
            
            # Extract observations - everything after "Observations:"
            obs_pattern = r'Observations?:\s*(.*?)(?=\[Step|\n━|$)'
            obs_matches = re.findall(obs_pattern, block_content, re.DOTALL)
            for obs in obs_matches:
                if obs.strip():
                    step_info['observations'].append(obs.strip())
            
            # Extract duration from the end of the block
            duration_pattern = r'\[Step\s+\d+:\s*Duration\s+([\d.]+)\s*seconds[^]]*\]'
            duration_match = re.search(duration_pattern, block_content)
            if duration_match:
                step_info['duration'] = float(duration_match.group(1))
            
            # Extract token information
            token_pattern = r'Input tokens:\s*([0-9,]+).*?Output tokens:\s*([0-9,]+)'
            token_match = re.search(token_pattern, block_content)
            if token_match:
                step_info['input_tokens'] = token_match.group(1)
                step_info['output_tokens'] = token_match.group(2)
            
            # Store all content for debugging
            step_info['content'] = [line.strip() for line in block_content.split('\n') if line.strip()]
            
            steps.append(step_info)
        
        return {
            'steps': steps,
            'total_steps': len(steps),
            'raw_output': output
        }
    
    def extract_created_files(self, output: str) -> List[Dict]:
        """Extract information about created files from agent output"""
        files = []
        
        # Common patterns for file creation
        patterns = [
            r'Successfully created.*?:\s*([^\n]+)',
            r'Created file:\s*([^\n]+)',
            r'Saved to:\s*([^\n]+)',
            r'Writing to:\s*([^\n]+)',
            r'Output file:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                file_path = match.strip()
                if os.path.exists(file_path):
                    try:
                        stat = os.stat(file_path)
                        files.append({
                            'path': file_path,
                            'name': os.path.basename(file_path),
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'type': self.get_file_type(file_path)
                        })
                    except:
                        files.append({
                            'path': file_path,
                            'name': os.path.basename(file_path),
                            'size': 0,
                            'modified': 'Unknown',
                            'type': self.get_file_type(file_path)
                        })
        
        return files
    
    def get_file_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        ext = Path(file_path).suffix.lower()
        type_map = {
            '.py': 'Python Script',
            '.txt': 'Text File',
            '.csv': 'CSV Data',
            '.json': 'JSON Data',
            '.pkl': 'Pickle Data',
            '.pth': 'PyTorch Model',
            '.h5': 'HDF5 Data',
            '.png': 'Image',
            '.jpg': 'Image',
            '.pdf': 'PDF Document',
            '.md': 'Markdown',
            '.yaml': 'YAML Config',
            '.yml': 'YAML Config',
        }
        return type_map.get(ext, 'File')
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def format_steps_display(self, parsed_output: Dict) -> str:
        """Format execution steps for display"""
        if not parsed_output or not parsed_output.get('steps'):
            return "**Execution Steps**\n\nNo detailed steps available."
        
        display = "**Execution Steps**\n\n"
        
        for step in parsed_output['steps']:
            step_num = step.get('step_number', '?')
            timestamp = step.get('timestamp', '')
            duration = step.get('duration')
            input_tokens = step.get('input_tokens', '')
            output_tokens = step.get('output_tokens', '')
            
            # Step header with duration and token info
            header = f"### 🔄 Step {step_num}"
            if duration:
                header += f" (Duration: {duration:.2f}s)"
            header += f" [{timestamp}]\n\n"
            display += header
            
            # Show tools called with box formatting like terminal
            if step.get('tools'):
                display += "**🔧 Tool Execution:**\n"
                for tool in step['tools']:
                    display += "```\n"
                    display += "╭─────────────────────────────────────────────────────────────╮\n"
                    display += f"│ Calling tool: '{tool['name']}'\n"
                    
                    if isinstance(tool['arguments'], dict):
                        display += "│ with arguments: {\n"
                        for k, v in tool['arguments'].items():
                            val_str = str(v)
                            if len(val_str) > 80:
                                val_str = val_str[:80] + "..."
                            display += f"│   '{k}': '{val_str}'\n"
                        display += "│ }\n"
                    else:
                        args_str = str(tool['arguments'])
                        if len(args_str) > 80:
                            args_str = args_str[:80] + "..."
                        display += f"│ with arguments: {args_str}\n"
                    
                    display += "╰─────────────────────────────────────────────────────────────╯\n"
                    display += "```\n\n"
            
            # Show observations with proper formatting
            if step.get('observations'):
                display += "**📊 Observations:**\n"
                for obs in step['observations']:
                    if obs.strip():
                        # Format observations to match terminal output
                        obs_text = obs
                        
                        # Check if it contains search results or structured data
                        if "## Search Results" in obs or "|" in obs:
                            # Keep search results in code blocks for better readability
                            display += f"```\n{obs_text}\n```\n\n"
                        else:
                            # Regular text observations
                            display += f"{obs_text}\n\n"
            
            # Show performance metrics
            if duration or input_tokens or output_tokens:
                display += "**⏱️ Performance Metrics:**\n"
                if duration:
                    display += f"- Duration: {duration:.2f} seconds\n"
                if input_tokens:
                    display += f"- Input tokens: {input_tokens}\n"
                if output_tokens:
                    display += f"- Output tokens: {output_tokens}\n"
                display += "\n"
            
            # Show raw content for debugging if needed
            if step.get('raw_content') and len(step['raw_content'].strip()) > 0:
                # Only show if there's significant content not already displayed
                raw_lines = [line for line in step['raw_content'].split('\n') if line.strip()]
                if len(raw_lines) > 10:  # Only show for substantial content
                    display += "**📝 Additional Details:**\n"
                    display += "```\n"
                    # Show first few and last few lines
                    for line in raw_lines[:3]:
                        if line.strip():
                            display += f"{line}\n"
                    if len(raw_lines) > 6:
                        display += "...\n"
                        for line in raw_lines[-3:]:
                            if line.strip():
                                display += f"{line}\n"
                    display += "```\n\n"
            
            display += "---\n\n"
        
        return display
    
    def format_files_display(self, files: List[Dict]) -> str:
        """Format created files for display with download links"""
        if not files:
            return "**Created Files**\n\nNo files created yet."
        
        display = f"**Created Files** ({len(files)} files)\n\n"
        
        for file_info in files:
            name = file_info['name']
            size = self.format_file_size(file_info['size'])
            file_type = file_info['type']
            modified = file_info['modified']
            
            display += f"📄 **{name}**\n"
            display += f"   Type: {file_type} | Size: {size}\n"
            display += f"   Modified: {modified}\n"
            display += f"   Path: `{file_info['path']}`\n\n"
        
        return display
    
    def parse_realtime_steps(self, output: str) -> List[Dict]:
        """Parse steps from real-time output stream with improved patterns"""
        steps = []
        
        # Debug: Print raw output to understand format (only if debug mode is enabled)
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"🔍 DEBUG: Parsing output of length {len(output)}")
            if output:
                print(f"🔍 DEBUG: First 500 chars: {repr(output[:500])}")
        
        # Multiple patterns to match different step formats
        step_patterns = [
            r'━+\s*Step\s+(\d+)\s*━+',  # Unicode box drawing
            r'=+\s*Step\s+(\d+)\s*=+',  # ASCII equals
            r'-+\s*Step\s+(\d+)\s*-+',  # ASCII dashes
            r'\[Step\s+(\d+)\]',        # Square brackets
            r'Step\s+(\d+):',           # Simple colon format
        ]
        
        # Try each pattern until we find matches
        step_blocks = None
        used_pattern = None
        
        for pattern in step_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                step_blocks = re.split(pattern, output, flags=re.IGNORECASE)
                used_pattern = pattern
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"🔍 DEBUG: Found {len(matches)} steps using pattern: {pattern}")
                break
        
        if not step_blocks or len(step_blocks) < 2:
            # If no formal steps found, create a single step from content
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print("🔍 DEBUG: No formal steps found, creating synthetic step")
            if output.strip():
                synthetic_step = {
                    'step_number': 1,
                    'tools': [],
                    'observations': [],
                    'duration': None,
                    'input_tokens': '',
                    'output_tokens': '',
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'status': 'completed',
                    'content': output.strip()
                }
                
                # Try to extract some basic info from content
                tool_mentions = re.findall(r'(run_shell_command|create_training_script|visit_webpage|search_github_repositories|WebSearchTool)', output, re.IGNORECASE)
                for tool in tool_mentions:
                    synthetic_step['tools'].append({
                        'name': tool,
                        'arguments': 'parsing...'
                    })
                
                steps.append(synthetic_step)
            
            return steps
        
        # Process each step block
        for i in range(1, len(step_blocks), 2):
            if i + 1 >= len(step_blocks):
                break
                
            try:
                step_number = int(step_blocks[i])
            except (ValueError, IndexError):
                continue
                
            block_content = step_blocks[i + 1] if i + 1 < len(step_blocks) else ""
            
            step_info = {
                'step_number': step_number,
                'tools': [],
                'observations': [],
                'duration': None,
                'input_tokens': '',
                'output_tokens': '',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'status': 'completed' if '[Step' in block_content or 'Duration' in block_content else 'in_progress',
                'content': block_content.strip()[:500]  # Keep first 500 chars for debugging
            }
            
            # Multiple tool call patterns
            tool_patterns = [
                r'╭[─]*╮\n│\s*Calling tool:\s*[\'"]?([^\'"\n]+)[\'"]?\s*with arguments:\s*({[^}]*})[^│]*│\n╰[─]*╯',
                r'Calling tool:\s*[\'"]?([^\'"\n]+)[\'"]?\s*with arguments:\s*({[^}]*})',
                r'Tool:\s*([^\s]+)\s*Args:\s*({[^}]*})',
                r'Using tool:\s*([^\s]+)',
                r'→\s*([a-zA-Z_][a-zA-Z0-9_]*)\(',  # Function call pattern
            ]
            
            for pattern in tool_patterns:
                tool_matches = re.findall(pattern, block_content, re.DOTALL)
                for match in tool_matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        tool_name, args_str = match[0], match[1]
                        try:
                            args_str = args_str.strip()
                            args = json.loads(args_str) if args_str.startswith('{') else args_str
                        except:
                            args = args_str
                    else:
                        tool_name = match if isinstance(match, str) else match[0]
                        args = "arguments not captured"
                
                    step_info['tools'].append({
                        'name': tool_name.strip(),
                        'arguments': args
                    })
                
                if tool_matches:
                    break  # Stop at first successful pattern
            
            # Enhanced observation patterns
            obs_patterns = [
                r'Observations?:\s*(.*?)(?=\[Step|\n━|\n=|\n-|$)',
                r'Result:\s*(.*?)(?=\[Step|\n━|\n=|\n-|$)',
                r'Output:\s*(.*?)(?=\[Step|\n━|\n=|\n-|$)',
                r'Response:\s*(.*?)(?=\[Step|\n━|\n=|\n-|$)',
            ]
            
            for pattern in obs_patterns:
                obs_matches = re.findall(pattern, block_content, re.DOTALL)
                for obs in obs_matches:
                    if obs.strip():
                        step_info['observations'].append(obs.strip())
                if obs_matches:
                    break
            
            # If no formal observations, use part of content as observation
            if not step_info['observations'] and block_content.strip():
                # Take meaningful content as observation
                content_lines = [line.strip() for line in block_content.split('\n') if line.strip()]
                if content_lines:
                    step_info['observations'].append(' '.join(content_lines[:3]))
            
            # Extract performance metrics
            duration_patterns = [
                r'\[Step\s+\d+:\s*Duration\s+([\d.]+)\s*seconds[^]]*\]',
                r'Duration[:\s]+([\d.]+)\s*s',
                r'took\s+([\d.]+)\s*seconds',
            ]
            
            for pattern in duration_patterns:
                duration_match = re.search(pattern, block_content)
                if duration_match:
                    step_info['duration'] = float(duration_match.group(1))
                    step_info['status'] = 'completed'
                    break
            
            token_patterns = [
                r'Input tokens:\s*([0-9,]+).*?Output tokens:\s*([0-9,]+)',
                r'(\d+)\s*input.*?(\d+)\s*output',
            ]
            
            for pattern in token_patterns:
                token_match = re.search(pattern, block_content)
                if token_match:
                    step_info['input_tokens'] = token_match.group(1)
                    step_info['output_tokens'] = token_match.group(2)
                    break
            
            steps.append(step_info)
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"🔍 DEBUG: Parsed step {step_number}: {len(step_info['tools'])} tools, {len(step_info['observations'])} observations")
        
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"🔍 DEBUG: Total parsed steps: {len(steps)}")
        return steps
    
    def format_chat_response(self, response_str: str, execution_time: float, steps_count: int, files_count: int) -> str:
        """Format the chat response in an organized way"""
        
        # Extract key information from response
        success_indicators = ['successfully', 'completed', 'created', 'generated', 'trained', 'plotted']
        is_successful = any(indicator in response_str.lower() for indicator in success_indicators)
        
        # Build organized response
        formatted_response = ""
        
        # Status header
        if is_successful:
            formatted_response += "✅ **Task Completed Successfully**\n\n"
        else:
            formatted_response += "🔄 **Task Processed**\n\n"
        
        # Quick summary
        formatted_response += f"📊 **Quick Summary:**\n"
        formatted_response += f"- ⏱️ Execution time: {execution_time:.1f} seconds\n"
        formatted_response += f"- 🔄 Steps executed: {steps_count}\n"
        formatted_response += f"- 📁 Files created: {files_count}\n\n"
        
        # Main response content
        formatted_response += "📝 **Detailed Response:**\n\n"
        
        # Clean and organize the response
        clean_response = self.clean_agent_response(response_str)
        
        # If response is too long, truncate intelligently
        if len(clean_response) > 1500:
            # Try to find a good break point
            sentences = clean_response.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) < 1200:
                    truncated += sentence + ". "
                else:
                    break
            
            if truncated:
                formatted_response += truncated + "\n\n"
                formatted_response += f"*... (Response truncated for readability. Full details available in steps and files.)*\n\n"
            else:
                formatted_response += clean_response[:1200] + "...\n\n"
        else:
            formatted_response += clean_response + "\n\n"
        
        # Action items or next steps if available
        if "next" in response_str.lower() or "recommend" in response_str.lower():
            formatted_response += "💡 **Next Steps / Recommendations:**\n"
            formatted_response += "Check the 'Execution Steps' and 'Created Files' tabs for detailed information.\n\n"
        
        # Footer
        formatted_response += "🔍 **More Details:**\n"
        formatted_response += "- View step-by-step execution in the 'Execution Steps' tab\n"
        formatted_response += "- Download created files from the 'Created Files' tab\n"
        formatted_response += "- Check system status and performance metrics in the 'System Status' tab"
        
        return formatted_response
    
    def clean_agent_response(self, response_str: str) -> str:
        """Clean and organize the raw agent response"""
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', response_str)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # Remove common agent artifacts
        artifacts_to_remove = [
            r'task_completed[^:]*:',
            r'dataset[^:]*:',
            r'features[^:]*:',
            r'targets[^:]*:',
            r'model_architecture[^:]*:',
        ]
        
        for artifact in artifacts_to_remove:
            cleaned = re.sub(artifact, '', cleaned, flags=re.IGNORECASE)
        
        # Format structured data better
        # Convert dictionary-like output to readable format
        if '{' in cleaned and '}' in cleaned:
            # Try to format JSON-like structures
            import json
            json_blocks = re.findall(r'\{[^{}]*\}', cleaned)
            for block in json_blocks:
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict):
                        formatted_dict = "\n".join([f"  - **{k}**: {v}" for k, v in parsed.items()])
                        cleaned = cleaned.replace(block, f"\n{formatted_dict}\n")
                except:
                    pass
        
        # Format lists better
        cleaned = re.sub(r"'([^']*)':\s*'([^']*)'", r"**\1**: \2", cleaned)
        
        # Improve readability of technical terms
        technical_terms = {
            'epochs_completed': 'Training Epochs Completed',
            'early_stopping': 'Early Stopping Applied',
            'final_training_loss': 'Final Training Loss',
            'final_validation_loss': 'Final Validation Loss',
            'final_training_mae': 'Final Training MAE',
            'final_validation_mae': 'Final Validation MAE',
            'trainable_parameters': 'Trainable Parameters',
            'total_parameters': 'Total Parameters',
        }
        
        for tech_term, readable_term in technical_terms.items():
            cleaned = cleaned.replace(tech_term, readable_term)
        
        # Clean up remaining artifacts
        cleaned = re.sub(r'^\s*[,:]', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def format_realtime_steps(self, steps: List[Dict], completed: bool = False, execution_time: float = None) -> str:
        """Format real-time steps for display"""
        if not steps:
            return "**Execution Steps**\n\n🔄 Monitoring agent execution...\n\nWaiting for steps to appear..."
        
        display = "**Execution Steps** - Real-time Monitoring\n\n"
        
        if completed and execution_time:
            display += f"✅ **Task completed in {execution_time:.1f} seconds**\n\n"
        else:
            display += "🔄 **Task in progress...**\n\n"
        
        for step in steps:
            step_num = step.get('step_number', '?')
            timestamp = step.get('timestamp', '')
            duration = step.get('duration')
            status = step.get('status', 'in_progress')
            
            # Step header
            if status == 'completed':
                status_icon = "✅"
            else:
                status_icon = "🔄"
            
            header = f"### {status_icon} Step {step_num}"
            if duration:
                header += f" (Duration: {duration:.2f}s)"
            header += f" [{timestamp}]\n\n"
            display += header
            
            # Show tools called
            if step.get('tools'):
                display += "**🔧 Tool Execution:**\n"
                for tool in step['tools']:
                    display += "```\n"
                    display += "╭─────────────────────────────────────────────────────────────╮\n"
                    display += f"│ Calling tool: '{tool['name']}'\n"
                    
                    if isinstance(tool['arguments'], dict):
                        display += "│ with arguments: {\n"
                        for k, v in tool['arguments'].items():
                            val_str = str(v)
                            if len(val_str) > 60:
                                val_str = val_str[:60] + "..."
                            display += f"│   '{k}': '{val_str}'\n"
                        display += "│ }\n"
                    else:
                        args_str = str(tool['arguments'])
                        if len(args_str) > 60:
                            args_str = args_str[:60] + "..."
                        display += f"│ with arguments: {args_str}\n"
                    
                    display += "╰─────────────────────────────────────────────────────────────╯\n"
                    display += "```\n\n"
            
            # Show observations
            if step.get('observations'):
                display += "**📊 Observations:**\n"
                for obs in step['observations']:
                    if obs.strip():
                        # Truncate very long observations for real-time display
                        obs_text = obs[:500] + "..." if len(obs) > 500 else obs
                        
                        if "## Search Results" in obs or "|" in obs:
                            display += f"```\n{obs_text}\n```\n\n"
                        else:
                            display += f"{obs_text}\n\n"
            
            # Show performance metrics
            if duration or step.get('input_tokens') or step.get('output_tokens'):
                display += "**⏱️ Performance:**\n"
                if duration:
                    display += f"- Duration: {duration:.2f} seconds\n"
                if step.get('input_tokens'):
                    display += f"- Input tokens: {step['input_tokens']}\n"
                if step.get('output_tokens'):
                    display += f"- Output tokens: {step['output_tokens']}\n"
                display += "\n"
            
            display += "---\n\n"
        
        if not completed:
            display += "🔄 **Execution continuing...**\n\n"
            display += "💡 Steps will update in real-time as the agent progresses."
        
        return display
    
    def create_interface(self):
        """Create the main Gradio interface"""
        
        # Custom CSS for bright, professional look
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        }
        
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            padding: 20px !important;
            border-radius: 10px !important;
            margin-bottom: 20px !important;
            text-align: center !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }
        
        .chat-container {
            background: white !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
            border: 1px solid #e1e5e9 !important;
        }
        
        .info-panel {
            background: white !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
            border: 1px solid #e1e5e9 !important;
            padding: 15px !important;
        }
        
        .submit-btn {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 30px !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .clear-btn {
            background: #6c757d !important;
            border: none !important;
            color: white !important;
            font-weight: 500 !important;
            padding: 12px 20px !important;
            border-radius: 8px !important;
        }
        
        .example-btn {
            background: #f8f9fa !important;
            border: 1px solid #dee2e6 !important;
            color: #495057 !important;
            padding: 10px 15px !important;
            border-radius: 6px !important;
            margin: 5px !important;
            transition: all 0.3s ease !important;
        }
        
        .example-btn:hover {
            background: #e9ecef !important;
            border-color: #adb5bd !important;
        }
        """
        
        with gr.Blocks(css=css, title="Stella - Scientific Discovery Agent", theme=gr.themes.Soft()) as interface:
            
            # Header with logo
            with open('./small_logo_b64.txt', 'r') as f:
                logo_data = f.read()
            
            gr.HTML(f"""
            <div class="main-header">
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                    <img src="{logo_data}" alt="Stella Logo" style="height: 80px; width: auto; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <div>
                        <h1 style="margin: 0; font-size: 2.5em; color: white;">Stella</h1>
                        <p style="margin: 5px 0 0 0; font-size: 1.1em; opacity: 0.9; color: white;">Self-Evolving LLM Agent for Scientific Discovery</p>
                    </div>
                </div>
            </div>
            """)
            
            with gr.Row():
                # Left Column - Chat Interface
                with gr.Column(scale=1, elem_classes="chat-container"):
                    gr.Markdown("### 💬 Chat Interface")
                    
                    chatbot = gr.Chatbot(
                        height=400,
                        show_label=False,
                        container=True,
                        bubble_full_width=False,
                        type='messages'
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Enter your request here...",
                            show_label=False,
                            scale=4,
                            container=False
                        )
                        submit_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=1,
                            elem_classes="submit-btn"
                        )
                    
                    with gr.Row():
                        clear_btn = gr.Button(
                            "Clear Chat",
                            variant="secondary",
                            elem_classes="clear-btn"
                        )
                        examples_btn = gr.Button(
                            "Show Examples",
                            variant="secondary",
                            elem_classes="example-btn"
                        )
                
                # Right Column - Information Panels
                with gr.Column(scale=1, elem_classes="info-panel"):
                    with gr.Tabs():
                        # Execution Steps Tab
                        with gr.Tab("🔍 Execution Steps"):
                            steps_display = gr.Markdown(
                                "**Execution Steps**\n\nWaiting for task execution...",
                                height=400
                            )
                        
                        # Created Files Tab
                        with gr.Tab("📁 Created Files"):
                            files_display = gr.Markdown(
                                "**Created Files**\n\nNo files created yet.",
                                height=300
                            )
                            
                            # File download section
                            gr.Markdown("### Download Files")
                            download_files = gr.File(
                                label="Download Created Files",
                                file_count="multiple",
                                visible=True
                            )
                        
                        # System Status Tab
                        with gr.Tab("⚙️ System Status"):
                            status_display = gr.Markdown(
                                "**System Status**\n\nReady for tasks.",
                                height=400
                            )
            
            # Examples section (initially hidden)
            with gr.Row(visible=False) as examples_row:
                gr.Markdown("### 💡 Example Requests")
                with gr.Row():
                    ex1 = gr.Button("CRISPR-Cas9 gene editing mechanisms", elem_classes="example-btn")
                    ex2 = gr.Button("Spatial omics data analysis", elem_classes="example-btn")
                    ex3 = gr.Button("Protein structure prediction with AlphaFold", elem_classes="example-btn")
                with gr.Row():
                    ex4 = gr.Button("Quantum computing algorithms", elem_classes="example-btn")
                    ex5 = gr.Button("Nuclear fusion plasma dynamics", elem_classes="example-btn")
                    ex6 = gr.Button("Single-cell RNA sequencing analysis", elem_classes="example-btn")
            
            # Event handlers
            def submit_message(message, history):
                if not message.strip():
                    yield history, "", "", "", []
                    return
                
                # Add user message
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": "🤖 Processing your request..."})
                yield history, "**Execution Steps**\n\n⏳ Starting task execution...", "**Created Files**\n\nNo files created yet.", "**System Status**\n\n🔄 Task in progress...", []
                
                try:
                    # Real-time execution monitoring
                    import io
                    import contextlib
                    import threading
                    import queue
                    import sys
                    
                    # Create queues for real-time communication
                    output_queue = queue.Queue()
                    accumulated_output = ""
                    
                    # Custom stdout capture that forwards to both terminal and queue
                    class RealTimeCapture:
                        def __init__(self, original_stdout, output_queue):
                            self.original_stdout = original_stdout
                            self.output_queue = output_queue
                            self.buffer = ""
                        
                        def write(self, text):
                            # Write to original stdout (terminal) - keep terminal output visible
                            self.original_stdout.write(text)
                            self.original_stdout.flush()
                            
                            # Also send to queue for web interface
                            self.output_queue.put(('output', text))
                        
                        def flush(self):
                            self.original_stdout.flush()
                    
                    # Set up real-time capture
                    original_stdout = sys.stdout
                    capture = RealTimeCapture(original_stdout, output_queue)
                    
                    # Start agent execution in a separate thread
                    def run_agent():
                        try:
                            sys.stdout = capture
                            start_time = time.time()
                            print(f"🚀 Processing request: {message}")
                            response = stella_core.manager_agent.run(message, reset=False)
                            execution_time = time.time() - start_time
                            output_queue.put(('response', response, execution_time))
                        except Exception as e:
                            output_queue.put(('error', str(e)))
                        finally:
                            sys.stdout = original_stdout
                            output_queue.put(('done', None))
                    
                    # Start the agent thread
                    agent_thread = threading.Thread(target=run_agent)
                    agent_thread.start()
                    
                    # Monitor output and update UI in real-time
                    last_update_time = time.time()
                    update_interval = 1.0  # Update every 1 second
                    
                    while True:
                        try:
                            # Get output with timeout
                            item = output_queue.get(timeout=0.5)
                            
                            if item[0] == 'output':
                                # New output received
                                text = item[1]
                                accumulated_output += text
                                
                                # Update UI periodically to avoid too frequent updates
                                current_time = time.time()
                                if current_time - last_update_time >= update_interval:
                                    # Parse steps from accumulated output
                                    parsed_steps = self.parse_realtime_steps(accumulated_output)
                                    
                                    # Format steps for display
                                    steps_text = self.format_realtime_steps(parsed_steps)
                                    
                                    # Update UI with current progress
                                    yield history, steps_text, "**Created Files**\n\nScanning for created files...", f"**System Status**\n\n🔄 Executing steps... ({len(parsed_steps)} steps detected)", []
                                    
                                    last_update_time = current_time
                            
                            elif item[0] == 'response':
                                # Agent completed
                                response, execution_time = item[1], item[2]
                                response_str = str(response)
                                
                                # Final parsing and display
                                final_steps = self.parse_realtime_steps(accumulated_output)
                                steps_text = self.format_realtime_steps(final_steps, completed=True, execution_time=execution_time)
                                
                                # Extract created files
                                created_files = self.extract_created_files(response_str + "\n" + accumulated_output)
                                self.created_files.extend(created_files)
                                files_text = self.format_files_display(created_files)
                                
                                # Update chat with organized final response
                                final_response = self.format_chat_response(response_str, execution_time, len(final_steps), len(created_files))
                                
                                history[-1]["content"] = final_response
                                
                                # Prepare download files
                                download_file_paths = [f['path'] for f in created_files if os.path.exists(f['path'])]
                                
                                # System status
                                status_text = f"""**System Status**

✅ Task completed in {execution_time:.1f}s

**Summary:**
- 🎯 Task execution: Successful
- ⏱️ Execution time: {execution_time:.1f} seconds
- 📁 Files created: {len(created_files)}
- 📝 Response length: {len(response_str)} characters
- 🔄 Steps executed: {len(final_steps)}

**Real-time Monitoring:**
- ✅ Live step-by-step execution captured
- 🔧 Tool calls and observations monitored
- 📊 Performance metrics tracked in real-time
- 📺 Full details also visible in terminal"""
                                
                                print(f"✅ Task completed in {execution_time:.1f} seconds")
                                
                                yield history, steps_text, files_text, status_text, download_file_paths
                                break
                            
                            elif item[0] == 'error':
                                # Error occurred
                                error_msg = f"❌ Error processing request: {item[1]}"
                                print(error_msg)
                                history[-1]["content"] = error_msg
                                yield history, f"**Execution Steps**\n\n{error_msg}", "**Created Files**\n\nNo files created yet.", f"**System Status**\n\n{error_msg}", []
                                break
                            
                            elif item[0] == 'done':
                                # Thread finished
                                break
                                
                        except queue.Empty:
                            # Timeout - check if we should update UI anyway
                            current_time = time.time()
                            if current_time - last_update_time >= update_interval and accumulated_output:
                                parsed_steps = self.parse_realtime_steps(accumulated_output)
                                steps_text = self.format_realtime_steps(parsed_steps)
                                yield history, steps_text, "**Created Files**\n\nScanning for created files...", f"**System Status**\n\n🔄 Executing steps... ({len(parsed_steps)} steps detected)", []
                                last_update_time = current_time
                            continue
                    
                    # Wait for thread to complete
                    agent_thread.join()
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    history[-1]["content"] = error_msg
                    yield history, f"**Execution Steps**\n\n❌ Error occurred: {str(e)}", "**Created Files**\n\nNo files created.", f"**System Status**\n\n❌ Error: {str(e)}", []
            
            def clear_chat():
                self.conversation_history = []
                self.created_files = []
                return [], "**Execution Steps**\n\nWaiting for task execution...", "**Created Files**\n\nNo files created yet.", "**System Status**\n\nReady for tasks.", []
            
            def toggle_examples():
                return gr.update(visible=True)
            
            # Connect events
            submit_btn.click(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, steps_display, files_display, status_display, download_files]
            )
            
            msg_input.submit(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, steps_display, files_display, status_display, download_files]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, steps_display, files_display, status_display, download_files]
            )
            
            examples_btn.click(
                toggle_examples,
                outputs=[examples_row]
            )
            
            # Example button events
            ex1.click(lambda: "Explain CRISPR-Cas9 gene editing mechanisms and recent advances in precision genome editing", outputs=[msg_input])
            ex2.click(lambda: "Analyze spatial omics data integration methods and computational approaches for tissue mapping", outputs=[msg_input])
            ex3.click(lambda: "Compare AlphaFold3 with other protein structure prediction methods and discuss accuracy improvements", outputs=[msg_input])
            ex4.click(lambda: "Explain quantum computing algorithms for optimization problems and their advantages over classical methods", outputs=[msg_input])
            ex5.click(lambda: "Discuss nuclear fusion plasma dynamics, confinement methods, and recent breakthroughs in fusion energy", outputs=[msg_input])
            ex6.click(lambda: "Analyze single-cell RNA sequencing data processing pipelines and cell type identification methods", outputs=[msg_input])
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        interface.launch(**kwargs)

def main():
    """Launch Stella English UI"""
    print("🚀 Starting Stella AI Assistant (English Interface)")
    print("🌟 Features:")
    print("   - Real-time execution monitoring")
    print("   - Detailed step-by-step analysis")
    print("   - Downloadable file management")
    print("   - Professional bright UI design")
    print()
    
    # Initialize Stella core without launching its Gradio interface
    print("🔧 Initializing Stella core...")
    success = stella_core.initialize_stella(use_template=True, use_mem0=True)
    if not success:
        print("❌ Failed to initialize Stella core!")
        return
    
    print("✅ Stella core initialized successfully!")
    stella = StellaEnglishUI()
    stella.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Enable public sharing for easier access
        debug=True,
        inbrowser=True  # Automatically open browser
    )

if __name__ == "__main__":
    main() 
