#!/usr/bin/env python3
"""
OpenAI O1-Series Chat Agent - Production-Ready Chat Interface

This module provides a complete chat agent implementation for OpenAI's O1 model
using the standard chat completions API endpoint. Features include:

- Support for O1 model with reasoning capabilities
- Persistent conversation history with rolling backups
- Streaming and non-streaming response support
- File inclusion in messages via {filename} syntax with programming file support
- Advanced configuration management
- Comprehensive logging and statistics
- Export capabilities (JSON, TXT, MD, HTML)
- Secure API key management
- Interactive CLI with colored output

Example Usage:
    # Start interactive chat
    python openai_oseries_agent.py --agent-id my-agent

    # List all agents
    python openai_oseries_agent.py --list

    # Export conversation
    python openai_oseries_agent.py --agent-id my-agent --export html

    # Configure agent settings
    python openai_oseries_agent.py --agent-id my-agent --config
"""

import os
import sys
import json
import requests
import argparse
import re
import logging
import yaml
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Optional, Generator, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from requests.exceptions import RequestException, HTTPError, Timeout

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""

    class Style:
        BRIGHT = DIM = RESET_ALL = ""


@dataclass
class AgentConfig:
    """Configuration settings for the OpenAI O1-Series Chat Agent"""
    model: str = "o1"
    temperature: float = 1.0
    reasoning_effort: str = "medium"  # low, medium, high
    reasoning_summary: str = "auto"   # auto, detailed, none
    max_output_tokens: Optional[int] = None
    max_history_size: int = 1000
    stream: bool = True
    system_prompt: Optional[str] = None
    store: bool = True
    text_format: str = "text"  # text
    text_verbosity: str = "medium"  # low, medium, high
    top_p: float = 1.0
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class OpenAIOSeriesChatAgent:
    """Advanced OpenAI O1-Series Chat Agent with persistence and streaming support"""

    # Supported O1 model variant
    SUPPORTED_MODELS = {
        "o1": {
            "name": "O1",
            "description": "Advanced reasoning model with sophisticated problem-solving capabilities",
            "reasoning_timeout": {"low": 180, "medium": 480, "high": 900},  # 3, 8, 15 minutes
            "has_reasoning": True
        }
    }

    # Programming and common file extensions
    SUPPORTED_EXTENSIONS = {
        # Programming languages
        '.py', '.r', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.cxx',
        '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
        '.clj', '.hs', '.ml', '.fs', '.vb', '.pl', '.pm', '.sh', '.bash', '.zsh', '.fish',
        '.ps1', '.bat', '.cmd', '.sql', '.html', '.htm', '.css', '.scss', '.sass', '.less',
        '.xml', '.xsl', '.xslt', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
        '.properties', '.env', '.dockerfile', '.docker', '.makefile', '.cmake', '.gradle',
        '.sbt', '.pom', '.lock', '.mod', '.sum',

        # Data and markup
        '.md', '.markdown', '.rst', '.tex', '.latex', '.csv', '.tsv', '.jsonl', '.ndjson',
        '.xml', '.svg', '.rss', '.atom', '.plist',

        # Configuration and infrastructure
        '.tf', '.tfvars', '.hcl', '.nomad', '.consul', '.vault', '.k8s', '.kubectl',
        '.helm', '.kustomize', '.ansible', '.inventory', '.playbook',

        # Documentation and text
        '.txt', '.log', '.out', '.err', '.trace', '.debug', '.info', '.warn', '.error',
        '.readme', '.license', '.changelog', '.authors', '.contributors', '.todo',

        # Notebooks and scripts
        '.ipynb', '.rmd', '.qmd', '.jl', '.m', '.octave', '.R', '.Rmd',

        # Web and API
        '.graphql', '.gql', '.rest', '.http', '.api', '.postman', '.insomnia',

        # Other useful formats
        '.editorconfig', '.gitignore', '.gitattributes', '.dockerignore', '.eslintrc',
        '.prettierrc', '.babelrc', '.webpack', '.rollup', '.vite', '.parcel'
    }

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.base_dir = Path(f"agents/{agent_id}")
        self.api_url = "https://api.openai.com/v1/chat/completions"

        # Create directory structure
        self._setup_directories()

        # Setup logging
        self._setup_logging()

        # Load or create config
        self.config = self._load_config()

        # Load conversation history
        self.messages = self._load_history()

        # Setup API key
        self.api_key = self._get_api_key()

        self.logger.info(f"Initialized OpenAI O1-Series Chat Agent: {agent_id} with model: {self.config.model}")

    def _setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.base_dir,
            self.base_dir / "backups",
            self.base_dir / "logs",
            self.base_dir / "exports",
            self.base_dir / "uploads"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Configure logging to file and console"""
        log_file = self.base_dir / "logs" / f"{datetime.now().strftime('%Y-%m-%d')}.log"

        # Create logger
        self.logger = logging.getLogger(f"OpenAIOSeriesAgent_{self.agent_id}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self) -> AgentConfig:
        """Load agent configuration from config.yaml"""
        config_file = self.base_dir / "config.yaml"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    return AgentConfig(**config_data)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                return AgentConfig()
        else:
            config = AgentConfig()
            self._save_config(config)
            return config

    def _save_config(self, config: Optional[AgentConfig] = None):
        """Save agent configuration to config.yaml"""
        if config is None:
            config = self.config

        config.updated_at = datetime.now().isoformat()
        config_file = self.base_dir / "config.yaml"

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(config), f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

    def _get_api_key(self) -> str:
        """Get API key from environment or secrets file, prompt if needed"""
        # First try environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.logger.info("Using API key from environment variable")
            return api_key

        # Try secrets file
        secrets_file = self.base_dir / "secrets.json"
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
                    keys = secrets.get('keys', {})
                    api_key = keys.get(self.config.model) or keys.get('default')
                    if api_key:
                        self.logger.info("Using API key from secrets file")
                        return api_key
            except Exception as e:
                self.logger.error(f"Error reading secrets file: {e}")

        # Prompt user for API key
        model_display = self.SUPPORTED_MODELS.get(self.config.model, {}).get('name', self.config.model)
        print(f"{Fore.YELLOW}API key not found for OpenAI {model_display} model.")
        print(f"You can set the OPENAI_API_KEY environment variable or enter it now.{Style.RESET_ALL}")

        api_key = input(f"{Fore.CYAN}Enter API key for OpenAI {model_display}: {Style.RESET_ALL}").strip()

        if not api_key:
            raise ValueError("API key is required")

        # Save to secrets file
        secrets = {
            "provider": "openai",
            "keys": {
                "default": api_key,
                self.config.model: api_key
            }
        }

        try:
            with open(secrets_file, 'w') as f:
                json.dump(secrets, f, indent=2)

            # Add to .gitignore
            gitignore_file = Path('.gitignore')
            gitignore_content = ""
            if gitignore_file.exists():
                gitignore_content = gitignore_file.read_text()

            if 'secrets.json' not in gitignore_content:
                with open(gitignore_file, 'a') as f:
                    f.write('\n# API Keys\n**/secrets.json\nsecrets.json\n')

            masked_key = f"{api_key[:4]}...{api_key[-2:]}" if len(api_key) > 6 else "***"
            print(f"{Fore.GREEN}API key saved ({masked_key}){Style.RESET_ALL}")
            self.logger.info(f"API key saved for user (length: {len(api_key)})")

        except Exception as e:
            self.logger.error(f"Error saving API key: {e}")
            print(f"{Fore.RED}Warning: Could not save API key to file{Style.RESET_ALL}")

        return api_key

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from history.json"""
        history_file = self.base_dir / "history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading history: {e}")
                return []
        return []

    def _save_history(self):
        """Save conversation history to history.json with backup"""
        history_file = self.base_dir / "history.json"

        # Create backup if history exists
        if history_file.exists():
            self._create_backup()

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")

    def _create_backup(self):
        """Create rolling backup of history"""
        history_file = self.base_dir / "history.json"
        backup_dir = self.base_dir / "backups"

        if not history_file.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"history_{timestamp}.json"

        try:
            shutil.copy2(history_file, backup_file)

            # Keep only last 10 backups
            backups = sorted(backup_dir.glob("history_*.json"))
            while len(backups) > 10:
                oldest = backups.pop(0)
                oldest.unlink()

        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.messages.append(message)

        # Truncate history if needed
        if len(self.messages) > self.config.max_history_size:
            removed = self.messages[:-self.config.max_history_size]
            self.messages = self.messages[-self.config.max_history_size:]
            self.logger.info(f"Truncated history: removed {len(removed)} old messages")

        self._save_history()

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file extension is supported for inclusion"""
        if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
            return True

        # Check for files without extensions but with known names
        known_files = {
            'makefile', 'dockerfile', 'rakefile', 'gemfile', 'podfile',
            'readme', 'license', 'changelog', 'authors', 'contributors',
            'todo', 'manifest', 'requirements', 'pipfile', 'poetry'
        }

        return file_path.name.lower() in known_files

    def _process_file_inclusions(self, content: str) -> str:
        """Replace {filename} patterns with file contents"""
        def replace_file(match):
            filename = match.group(1)

            # Search paths
            search_paths = [
                Path('.'),
                Path('src'),
                Path('lib'),
                Path('scripts'),
                Path('data'),
                Path('documents'),
                Path('files'),
                Path('config'),
                Path('configs'),
                self.base_dir / 'uploads'
            ]

            for search_path in search_paths:
                file_path = search_path / filename
                if file_path.exists() and file_path.is_file():

                    # Check if file is supported
                    if not self._is_supported_file(file_path):
                        self.logger.warning(f"Unsupported file type: {filename}")
                        return f"[WARNING: Unsupported file type {filename}]"

                    try:
                        # Check file size (limit to 2MB for programming files)
                        max_size = 2 * 1024 * 1024  # 2MB
                        if file_path.stat().st_size > max_size:
                            self.logger.error(f"File {filename} too large (>2MB)")
                            return f"[ERROR: File {filename} too large (max 2MB)]"

                        # Try UTF-8 first
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                        except UnicodeDecodeError:
                            # Fallback to latin-1
                            with open(file_path, 'r', encoding='latin-1') as f:
                                file_content = f.read()

                        # Add file info header for programming files
                        file_info = f"// File: {filename} ({file_path.suffix})\n"
                        if file_path.suffix.lower() in ['.py', '.r']:
                            file_info = f"# File: {filename} ({file_path.suffix})\n"
                        elif file_path.suffix.lower() in ['.html', '.xml']:
                            file_info = f"<!-- File: {filename} ({file_path.suffix}) -->\n"
                        elif file_path.suffix.lower() in ['.css', '.scss', '.sass']:
                            file_info = f"/* File: {filename} ({file_path.suffix}) */\n"
                        elif file_path.suffix.lower() in ['.sql']:
                            file_info = f"-- File: {filename} ({file_path.suffix})\n"

                        full_content = file_info + file_content

                        self.logger.info(f"Included file: {filename} ({len(file_content)} chars, {file_path.suffix})")
                        return full_content

                    except Exception as e:
                        self.logger.error(f"Error reading file {filename}: {e}")
                        return f"[ERROR: Could not read {filename}: {e}]"

            self.logger.warning(f"File not found: {filename}")
            return f"[ERROR: File {filename} not found]"

        return re.sub(r'\{([^}]+)\}', replace_file, content)

    def _build_api_payload(self, new_message: str, override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the API request payload matching the O1-series CURL structure"""
        # Process file inclusions
        processed_message = self._process_file_inclusions(new_message)

        # Build messages in the exact O1-series CURL structure
        messages = []

        # Add system prompt as developer role if configured
        if self.config.system_prompt:
            messages.append({
                "role": "developer",
                "content": [
                    {"type": "text", "text": self.config.system_prompt}
                ]
            })

        # Add conversation history (convert from storage format to API format)
        for msg in self.messages:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": [
                        {"type": "text", "text": msg["content"]}
                    ]
                })

        # Add new user message
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": processed_message}
            ]
        })

        # Apply config overrides
        config = asdict(self.config)
        if override_config:
            config.update(override_config)

        # Build payload matching the exact O1-series CURL structure
        payload = {
            "model": config["model"],
            "messages": messages,
            "response_format": {"type": config["text_format"]},
            "reasoning_effort": config["reasoning_effort"]
        }

        # Add streaming if enabled
        if config["stream"]:
            payload["stream"] = True

        # Add optional parameters
        if config.get("max_output_tokens"):
            payload["max_completion_tokens"] = config["max_output_tokens"]

        if config.get("temperature") != 1.0:
            payload["temperature"] = config["temperature"]

        if config.get("top_p") != 1.0:
            payload["top_p"] = config["top_p"]

        return payload

    def _get_timeout_for_reasoning(self, model: str = None, reasoning_effort: str = "medium") -> int:
        """Get appropriate timeout based on model variant and reasoning effort level"""
        if model is None:
            model = self.config.model

        model_info = self.SUPPORTED_MODELS.get(model)
        if not model_info:
            # Fallback to default o1 timeouts
            model_info = self.SUPPORTED_MODELS["o1"]

        return model_info["reasoning_timeout"].get(reasoning_effort, 300)

    def _make_api_request(self, payload: Dict[str, Any]) -> requests.Response:
        """Make API request with retries and error handling, extended timeout for reasoning"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Get appropriate timeout based on model and reasoning effort
        model = payload.get("model", self.config.model)
        reasoning_effort = payload.get("reasoning_effort", "medium")
        timeout = self._get_timeout_for_reasoning(model, reasoning_effort)

        model_display = self.SUPPORTED_MODELS.get(model, {}).get('name', model)
        self.logger.info(f"Using timeout of {timeout}s for {model_display} with reasoning effort: {reasoning_effort}")

        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Making API request to {model_display} (attempt {attempt + 1}/{max_retries}) with {timeout}s timeout...")

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    stream=payload.get("stream", True),
                    timeout=timeout
                )

                if response.status_code == 200:
                    self.logger.info("API request successful")
                    return response
                elif response.status_code == 401:
                    raise ValueError("Invalid API key")
                elif response.status_code == 403:
                    raise ValueError("API access forbidden")
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Rate limited, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                elif response.status_code >= 500:
                    # Server error - retry
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Server error {response.status_code}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    response.raise_for_status()

            except Timeout as e:
                self.logger.warning(f"Request timed out after {timeout}s (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise Exception(f"Request timed out after {timeout}s. Try reducing reasoning effort.")
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Retrying in {delay}s...")
                time.sleep(delay)
            except RequestException as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Request failed ({e}), retrying in {delay}s...")
                time.sleep(delay)

        raise Exception(f"Failed to complete API request after {max_retries} attempts")

    def _parse_streaming_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Parse streaming Server-Sent Events response matching the exact format"""
        assistant_message = ""

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or line.strip() == "":
                    continue

                try:
                    # Handle Server-Sent Events format
                    if line.startswith("data: "):
                        data_str = line[5:].strip()

                        if data_str == "[DONE]":
                            break

                        data = json.loads(data_str)

                        # Handle O1-series streaming format
                        choices = data.get("choices", [])
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                assistant_message += content
                                yield content

                            # Check for completion
                            finish_reason = choice.get("finish_reason")
                            if finish_reason == "stop":
                                break

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON in stream: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error processing stream line: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error parsing streaming response: {e}")

        # Add assistant message to history if we got content
        if assistant_message.strip():
            self.add_message("assistant", assistant_message)

    def _parse_non_streaming_response(self, response: requests.Response) -> str:
        """Parse non-streaming response from OpenAI chat completions API"""
        try:
            data = response.json()

            # Extract message content from standard OpenAI response format
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                # Handle structured content format
                content_array = message.get("content", [])
                if isinstance(content_array, list) and content_array:
                    content = content_array[0].get("text", "")
                else:
                    content = message.get("content", "")

                if content:
                    self.add_message("assistant", content)
                    return content

            return "No response content received"

        except Exception as e:
            self.logger.error(f"Error parsing non-streaming response: {e}")
            return f"Error parsing response: {e}"

    def call_api(self, new_message: str, override_config: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """Call OpenAI O1-Series API with the new message"""
        try:
            # Add user message to history
            self.add_message("user", new_message)

            # Build API payload
            payload = self._build_api_payload(new_message, override_config)

            self.logger.info(f"Making API call to {self.api_url}")
            self.logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            # Show model and reasoning info to user
            model = payload.get("model", self.config.model)
            reasoning_effort = payload.get("reasoning_effort", "medium")
            model_display = self.SUPPORTED_MODELS.get(model, {}).get('name', model)

            if reasoning_effort in ["medium", "high"]:
                timeout = self._get_timeout_for_reasoning(model, reasoning_effort)
                print(f"{Fore.YELLOW}🧠 Using {model_display} with {reasoning_effort.upper()} reasoning (timeout: {timeout//60}min {timeout%60}s)...{Style.RESET_ALL}")

            # Make request
            response = self._make_api_request(payload)

            # Handle streaming vs non-streaming
            if payload.get("stream", True):
                yield from self._parse_streaming_response(response)
            else:
                result = self._parse_non_streaming_response(response)
                yield result

        except Exception as e:
            error_msg = f"API call failed: {e}"
            self.logger.error(error_msg)
            yield error_msg

    def clear_history(self):
        """Clear conversation history"""
        self._create_backup()
        self.messages.clear()
        self._save_history()
        self.logger.info("Conversation history cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.messages:
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "total_characters": 0,
                "average_message_length": 0,
                "first_message": None,
                "last_message": None,
                "conversation_duration": None
            }

        user_msgs = [m for m in self.messages if m["role"] == "user"]
        assistant_msgs = [m for m in self.messages if m["role"] == "assistant"]

        total_chars = sum(len(m["content"]) for m in self.messages)
        avg_length = total_chars // len(self.messages) if self.messages else 0

        first_time = datetime.fromisoformat(self.messages[0]["timestamp"])
        last_time = datetime.fromisoformat(self.messages[-1]["timestamp"])
        duration = last_time - first_time

        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "total_characters": total_chars,
            "average_message_length": avg_length,
            "first_message": first_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_message": last_time.strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_duration": str(duration).split('.')[0] if duration.total_seconds() > 0 else "0:00:00"
        }

    def export_conversation(self, format_type: str) -> str:
        """Export conversation to specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.base_dir / "exports"

        if format_type == "json":
            filename = f"conversation_{timestamp}.json"
            filepath = export_dir / filename

            export_data = {
                "agent_id": self.agent_id,
                "exported_at": datetime.now().isoformat(),
                "config": asdict(self.config),
                "messages": self.messages,
                "statistics": self.get_statistics()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        elif format_type == "txt":
            filename = f"conversation_{timestamp}.txt"
            filepath = export_dir / filename

            model_display = self.SUPPORTED_MODELS.get(self.config.model, {}).get('name', self.config.model)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"OpenAI {model_display} Chat Agent Conversation Export\n")
                f.write(f"Agent ID: {self.agent_id}\n")
                f.write(f"Model: {self.config.model}\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                for msg in self.messages:
                    timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {msg['role'].upper()}:\n")
                    f.write(f"{msg['content']}\n\n")

        elif format_type == "md":
            filename = f"conversation_{timestamp}.md"
            filepath = export_dir / filename

            model_display = self.SUPPORTED_MODELS.get(self.config.model, {}).get('name', self.config.model)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# OpenAI {model_display} Chat Agent Conversation\n\n")
                f.write(f"**Agent ID:** {self.agent_id}  \n")
                f.write(f"**Model:** {self.config.model}  \n")
                f.write(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

                for msg in self.messages:
                    timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    role_emoji = "🧑" if msg["role"] == "user" else "🧠"
                    f.write(f"## {role_emoji} {msg['role'].title()} - {timestamp}\n\n")
                    f.write(f"{msg['content']}\n\n")

        elif format_type == "html":
            filename = f"conversation_{timestamp}.html"
            filepath = export_dir / filename

            stats = self.get_statistics()
            model_display = self.SUPPORTED_MODELS.get(self.config.model, {}).get('name', self.config.model)

            # HTML template with modern styling
            html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenAI {model_display} Conversation - {self.agent_id}</title>
    <style>
        :root {{
            --primary-color: #8b5cf6;
            --secondary-color: #f1f5f9;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
            --user-bg: #3b82f6;
            --assistant-bg: #8b5cf6;
            --code-bg: #f8fafc;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 4xl;
            margin: 0 auto;
            background: white;
            border-radius: 1rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            overflow: hidden;
        }}

        .header {{
            background: var(--primary-color);
            color: white;
            padding: 2rem;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .header-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
            font-size: 0.9rem;
        }}

        .stats {{
            background: var(--secondary-color);
            padding: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }}

        .stat-item {{
            text-align: center;
            padding: 1rem;
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--primary-color);
        }}

        .stat-label {{
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}

        .messages {{
            padding: 2rem;
            max-height: 70vh;
            overflow-y: auto;
        }}

        .message {{
            margin-bottom: 2rem;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        }}

        .message.user {{
            flex-direction: row-reverse;
        }}

        .message-avatar {{
            width: 3rem;
            height: 3rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            font-weight: bold;
            color: white;
            flex-shrink: 0;
        }}

        .message.user .message-avatar {{
            background: var(--user-bg);
        }}

        .message.assistant .message-avatar {{
            background: var(--assistant-bg);
        }}

        .message-content {{
            flex: 1;
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            position: relative;
        }}

        .message.user .message-content {{
            background: #eff6ff;
            border-color: var(--user-bg);
        }}

        .message.assistant .message-content {{
            background: #faf5ff;
            border-color: var(--assistant-bg);
        }}

        .message-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .message-role {{
            font-weight: 600;
            text-transform: capitalize;
        }}

        .message-time {{
            font-size: 0.8rem;
            color: #64748b;
        }}

        .message-text {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .code-block {{
            background: var(--code-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
        }}

        .footer {{
            background: var(--secondary-color);
            padding: 1rem 2rem;
            text-align: center;
            font-size: 0.8rem;
            color: #64748b;
            border-top: 1px solid var(--border-color);
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}

            .header {{
                padding: 1.5rem;
            }}

            .header h1 {{
                font-size: 1.5rem;
            }}

            .header-info {{
                grid-template-columns: 1fr;
            }}

            .messages {{
                padding: 1rem;
            }}

            .message-content {{
                padding: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 OpenAI {model_display} Chat Agent</h1>
            <p>Reasoning Model Conversation Export</p>
            <div class="header-info">
                <div><strong>Agent ID:</strong> {self.agent_id}</div>
                <div><strong>Model:</strong> {self.config.model}</div>
                <div><strong>Reasoning Effort:</strong> {self.config.reasoning_effort}</div>
                <div><strong>Exported:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div><strong>Temperature:</strong> {self.config.temperature}</div>
            </div>
        </div>

        <div class="stats">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats['total_messages']}</div>
                    <div class="stat-label">Total Messages</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['user_messages']}</div>
                    <div class="stat-label">User Messages</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['assistant_messages']}</div>
                    <div class="stat-label">Assistant Messages</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['total_characters']:,}</div>
                    <div class="stat-label">Total Characters</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats['average_message_length']:,}</div>
                    <div class="stat-label">Avg Message Length</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('conversation_duration', 'N/A')}</div>
                    <div class="stat-label">Duration</div>
                </div>
            </div>
        </div>

        <div class="messages">""" 

            # Add messages
            for msg in self.messages:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                role = msg["role"]
                content = msg["content"]

                # Escape HTML and preserve formatting
                import html
                content_escaped = html.escape(content)

                # Simple code block detection
                if '```' in content_escaped:
                    parts = content_escaped.split('```')
                    formatted_content = ""
                    for i, part in enumerate(parts):
                        if i % 2 == 1:  # Code block
                            formatted_content += f'<div class="code-block">{part}</div>'
                        else:  # Regular text
                            formatted_content += part
                    content_escaped = formatted_content

                avatar_text = "U" if role == "user" else "🧠"

                html_template += f"""
            <div class="message {role}">
                <div class="message-avatar">{avatar_text}</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-role">{role}</span>
                        <span class="message-time">{timestamp}</span>
                    </div>
                    <div class="message-text">{content_escaped}</div>
                </div>
            </div>"""

            # Close HTML
            html_template += f"""
        </div>

        <div class="footer">
            Generated by OpenAI {model_display} Chat Agent • Agent ID: {self.agent_id} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_template)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        self.logger.info(f"Exported conversation to {filepath}")
        return str(filepath)

    def search_history(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversation history for a term"""
        results = []
        term_lower = term.lower()

        for i, msg in enumerate(self.messages):
            if term_lower in msg["content"].lower():
                results.append({
                    "index": i,
                    "message": msg,
                    "preview": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                })

            if len(results) >= limit:
                break

        return results

    def list_files(self) -> List[str]:
        """List available files for inclusion"""
        files = []
        search_paths = [
            Path('.'),
            Path('src'),
            Path('lib'),
            Path('scripts'),
            Path('data'),
            Path('documents'),
            Path('files'),
            Path('config'),
            Path('configs'),
            self.base_dir / 'uploads'
        ]

        for search_path in search_paths:
            if search_path.exists():
                for file_path in search_path.rglob("*"):
                    if (file_path.is_file() and
                        not file_path.name.startswith('.') and
                        self._is_supported_file(file_path)):

                        size = file_path.stat().st_size
                        size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                        files.append(f"{file_path} ({size_str}) [{file_path.suffix}]")

        return sorted(files)

def list_agents() -> List[Dict[str, Any]]:
    """List all available agents"""
    agents_dir = Path("agents")
    agents = []

    if not agents_dir.exists():
        return agents

    for agent_dir in agents_dir.iterdir():
        if agent_dir.is_dir():
            metadata_file = agent_dir / "metadata.json"
            config_file = agent_dir / "config.yaml"
            history_file = agent_dir / "history.json"

            # Get basic info
            agent_info = {
                "id": agent_dir.name,
                "path": str(agent_dir),
                "exists": True
            }

            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        agent_info.update(metadata)
                except:
                    pass

            # Get config info
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                        agent_info["model"] = config.get("model", "o1")
                        agent_info["created_at"] = config.get("created_at")
                        agent_info["updated_at"] = config.get("updated_at")
                except:
                    pass

            # Get history size
            if history_file.exists():
                try:
                    with open(history_file) as f:
                        history = json.load(f)
                        agent_info["message_count"] = len(history)
                        agent_info["history_size"] = history_file.stat().st_size
                except:
                    agent_info["message_count"] = 0
                    agent_info["history_size"] = 0
            else:
                agent_info["message_count"] = 0
                agent_info["history_size"] = 0

            agents.append(agent_info)

    return sorted(agents, key=lambda x: x.get("updated_at", ""))

def show_agent_info(agent_id: str):
    """Display detailed agent information"""
    agent_dir = Path(f"agents/{agent_id}")

    if not agent_dir.exists():
        print(f"{Fore.RED}Agent '{agent_id}' not found{Style.RESET_ALL}")
        return

    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"Agent Information: {Fore.YELLOW}{agent_id}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")

    # Load and display config
    config_file = agent_dir / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)

            model = config.get('model', 'o1')
            model_display = OpenAIOSeriesChatAgent.SUPPORTED_MODELS.get(model, {}).get('name', model)

            print(f"\n{Fore.GREEN}Configuration:")
            print(f"{Fore.WHITE}  Model: {model} ({model_display})")
            print(f"  Temperature: {config.get('temperature', 1.0)}")
            print(f"  Reasoning Effort: {config.get('reasoning_effort', 'medium')}")
            print(f"  Reasoning Summary: {config.get('reasoning_summary', 'auto')}")
            print(f"  Streaming: {config.get('stream', True)}")
            print(f"  Created: {config.get('created_at', 'Unknown')}")
            print(f"  Updated: {config.get('updated_at', 'Unknown')}")

        except Exception as e:
            print(f"{Fore.RED}Error loading config: {e}")

    # Display history stats
    history_file = agent_dir / "history.json"
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)

            user_msgs = len([m for m in history if m.get("role") == "user"])
            assistant_msgs = len([m for m in history if m.get("role") == "assistant"])
            total_chars = sum(len(m.get("content", "")) for m in history)

            print(f"\n{Fore.GREEN}Conversation History:")
            print(f"{Fore.WHITE}  Total Messages: {len(history)}")
            print(f"  User Messages: {user_msgs}")
            print(f"  Assistant Messages: {assistant_msgs}")
            print(f"  Total Characters: {total_chars:,}")
            print(f"  File Size: {history_file.stat().st_size:,} bytes")

            if history:
                first_msg = datetime.fromisoformat(history[0]["timestamp"])
                last_msg = datetime.fromisoformat(history[-1]["timestamp"])
                print(f"  First Message: {first_msg.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Last Message: {last_msg.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            print(f"{Fore.RED}Error loading history: {e}")
    else:
        print(f"\n{Fore.YELLOW}No conversation history found")

    # Display directory structure
    print(f"\n{Fore.GREEN}Directory Structure:")
    for item in sorted(agent_dir.rglob("*")):
        if item.is_file():
            size = item.stat().st_size
            size_str = f"{size:,}" if size < 1024 else f"{size/1024:.1f}K"
            rel_path = item.relative_to(agent_dir)
            print(f"{Fore.WHITE}  {rel_path} ({size_str} bytes)")


def create_agent_config_interactive() -> AgentConfig:
    """Interactive configuration creation"""
    print(f"\n{Fore.CYAN}Creating Agent Configuration{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Press Enter to use default values{Style.RESET_ALL}\n")

    config = AgentConfig()

    # Since only o1 is supported, skip model selection
    print(f"{Fore.GREEN}Available O1-Series Model:")
    model_info = OpenAIOSeriesChatAgent.SUPPORTED_MODELS["o1"]
    timeouts = model_info["reasoning_timeout"]
    print(f"{Fore.WHITE}  1. {model_info['name']} (o1)")
    print(f"     {model_info['description']}")
    print(f"     Timeouts: Low={timeouts['low']}s, Medium={timeouts['medium']}s, High={timeouts['high']}s")
    print()

    # No model selection needed; model is fixed to o1
    selected_model_info = model_info
    print(f"{Fore.GREEN}Selected: {selected_model_info.get('name', config.model)}{Style.RESET_ALL}\n")

    # Temperature
    temp_input = input(f"Temperature (0.0-2.0) [{config.temperature}]: ").strip()
    if temp_input:
        try:
            temperature = float(temp_input)
            if 0.0 <= temperature <= 2.0:
                config.temperature = temperature
            else:
                print(f"{Fore.RED}Temperature out of range, using default{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid temperature, using default{Style.RESET_ALL}")

    # Reasoning effort
    effort_input = input(f"Reasoning effort (low/medium/high) [{config.reasoning_effort}]: ").strip().lower()
    if effort_input and effort_input in ["low", "medium", "high"]:
        config.reasoning_effort = effort_input

        # Show timeout for selected model and effort
        timeout = selected_model_info.get("reasoning_timeout", {}).get(config.reasoning_effort, 300)
        print(f"{Fore.YELLOW}  → Timeout for o1 with {config.reasoning_effort} effort: {timeout}s ({timeout//60}min {timeout%60}s){Style.RESET_ALL}")

    # Reasoning summary
    summary_input = input(f"Reasoning summary (auto/detailed/none) [{config.reasoning_summary}]: ").strip().lower()
    if summary_input and summary_input in ["auto", "detailed", "none"]:
        config.reasoning_summary = summary_input

    # System prompt
    system_prompt = input(f"System prompt (optional): ").strip()
    if system_prompt:
        config.system_prompt = system_prompt

    # Max output tokens
    tokens_input = input(f"Max output tokens (optional): ").strip()
    if tokens_input:
        try:
            max_tokens = int(tokens_input)
            if max_tokens > 0:
                config.max_output_tokens = max_tokens
            else:
                print(f"{Fore.RED}Invalid token count, leaving unset{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid token count, leaving unset{Style.RESET_ALL}")

    # Streaming
    stream_input = input(f"Enable streaming (y/n) [{'y' if config.stream else 'n'}]: ").strip().lower()
    if stream_input in ['n', 'no', 'false']:
        config.stream = False
    elif stream_input in ['y', 'yes', 'true']:
        config.stream = True

    return config


def interactive_chat(agent: OpenAIOSeriesChatAgent):
    """Interactive chat session"""
    model_display = agent.SUPPORTED_MODELS.get(agent.config.model, {}).get('name', agent.config.model)

    print(f"\n{Fore.GREEN}Starting interactive chat with OpenAI {model_display}")
    print(f"Agent: {Fore.YELLOW}{agent.agent_id}")
    print(f"{Fore.GREEN}Type '/help' for commands, '/quit' to exit{Style.RESET_ALL}\n")

    while True:
        try:
            user_input = input(f"{Fore.CYAN}You: {Style.RESET_ALL}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                command_parts = user_input[1:].split()
                command = command_parts[0].lower()

                if command == 'help':
                    print(f"\n{Fore.YELLOW}Available Commands:")
                    print(f"{Fore.WHITE}/help - Show this help message")
                    print(f"/history [n] - Show last n messages (default 5)")
                    print(f"/search <term> - Search conversation history")
                    print(f"/stats - Show conversation statistics")
                    print(f"/config - Show current configuration")
                    print(f"/export <json|txt|md|html> - Export conversation")
                    print(f"/clear - Clear conversation history")
                    print(f"/files - List available files for inclusion")
                    print(f"/info - Show agent information")
                    print(f"/quit - Exit chat{Style.RESET_ALL}\n")
                    print(f"{Fore.CYAN}File Inclusion: Use {{filename}} in messages to include file contents")
                    print(f"Supported: Programming files (.py, .r, .js, etc.), config files, documentation{Style.RESET_ALL}\n")
                    print(f"{Fore.YELLOW}🧠 O1-Series Models & Reasoning Timeouts:")
                    model_info = agent.SUPPORTED_MODELS["o1"]
                    timeouts = model_info["reasoning_timeout"]
                    print(f"{Fore.WHITE}  {model_info['name']} (o1):")
                    print(f"    {model_info['description']}")
                    print(f"    Timeouts: Low={timeouts['low']}s, Medium={timeouts['medium']}s, High={timeouts['high']}s")
                    print()

                elif command == 'history':
                    limit = 5
                    if len(command_parts) > 1:
                        try:
                            limit = int(command_parts[1])
                        except ValueError:
                            print(f"{Fore.RED}Invalid number{Style.RESET_ALL}")
                            continue

                    recent_messages = agent.messages[-limit:]
                    if not recent_messages:
                        print(f"{Fore.YELLOW}No messages in history{Style.RESET_ALL}")
                    else:
                        print(f"\n{Fore.YELLOW}Last {len(recent_messages)} messages:")
                        for msg in recent_messages:
                            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                            role_color = Fore.CYAN if msg["role"] == "user" else Fore.MAGENTA
                            print(f"{Fore.WHITE}[{timestamp}] {role_color}{msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                    print()

                elif command == 'search':
                    if len(command_parts) < 2:
                        print(f"{Fore.RED}Usage: /search <term>{Style.RESET_ALL}")
                        continue

                    search_term = ' '.join(command_parts[1:])
                    results = agent.search_history(search_term)

                    if not results:
                        print(f"{Fore.YELLOW}No matches found for '{search_term}'{Style.RESET_ALL}")
                    else:
                        print(f"\n{Fore.YELLOW}Found {len(results)} matches for '{search_term}':")
                        for result in results:
                            msg = result["message"]
                            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                            role_color = Fore.CYAN if msg["role"] == "user" else Fore.MAGENTA
                            print(f"{Fore.WHITE}[{timestamp}] {role_color}{msg['role']}: {result['preview']}")

                    print()

                elif command == 'stats':
                    stats = agent.get_statistics()
                    print(f"\n{Fore.YELLOW}Conversation Statistics:")
                    print(f"{Fore.WHITE}Model: {agent.config.model} ({model_display})")
                    print(f"Total Messages: {stats['total_messages']}")
                    print(f"User Messages: {stats['user_messages']}")
                    print(f"Assistant Messages: {stats['assistant_messages']}")
                    print(f"Total Characters: {stats['total_characters']:,}")
                    print(f"Average Message Length: {stats['average_message_length']:,}")
                    if stats['first_message']:
                        print(f"First Message: {stats['first_message']}")
                        print(f"Last Message: {stats['last_message']}")
                        print(f"Duration: {stats['conversation_duration']}")
                    print()

                elif command == 'config':
                    print(f"\n{Fore.YELLOW}Current Configuration:")
                    config_dict = asdict(agent.config)
                    for key, value in config_dict.items():
                        if key not in ['created_at', 'updated_at']:
                            if key == 'model':
                                model_name = agent.SUPPORTED_MODELS.get(str(value), {}).get('name', value)
                                print(f"{Fore.WHITE}{key}: {value} ({model_name})")
                            elif key == 'reasoning_effort':
                                timeout = agent._get_timeout_for_reasoning(agent.config.model, str(value))
                                print(f"{Fore.WHITE}{key}: {value} (timeout: {timeout}s)")
                            else:
                                print(f"{Fore.WHITE}{key}: {value}")
                    print()

                elif command == 'export':
                    if len(command_parts) < 2:
                        print(f"{Fore.RED}Usage: /export <json|txt|md|html>{Style.RESET_ALL}")
                        continue

                    format_type = command_parts[1].lower()
                    if format_type not in ['json', 'txt', 'md', 'html']:
                        print(f"{Fore.RED}Invalid format. Use: json, txt, md, or html{Style.RESET_ALL}")
                        continue

                    try:
                        filepath = agent.export_conversation(format_type)
                        print(f"{Fore.GREEN}Exported to: {filepath}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Export failed: {e}{Style.RESET_ALL}")

                elif command == 'clear':
                    confirm = input(f"{Fore.YELLOW}Clear conversation history? (y/N): {Style.RESET_ALL}").strip().lower()
                    if confirm in ['y', 'yes']:
                        agent.clear_history()
                        print(f"{Fore.GREEN}Conversation history cleared{Style.RESET_ALL}")

                elif command == 'files':
                    files = agent.list_files()
                    if not files:
                        print(f"{Fore.YELLOW}No supported files found for inclusion{Style.RESET_ALL}")
                    else:
                        print(f"\n{Fore.YELLOW}Available files for inclusion:")
                        for file_info in files[:20]:  # Limit to 20 files
                            print(f"{Fore.WHITE}{file_info}")
                        if len(files) > 20:
                            print(f"{Fore.YELLOW}... and {len(files) - 20} more files")
                    print(f"{Fore.CYAN}Use {{filename}} in your message to include file contents{Style.RESET_ALL}\n")

                elif command == 'info':
                    show_agent_info(agent.agent_id)

                elif command in ['quit', 'exit', 'q']:
                    print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")
                    break

                else:
                    print(f"{Fore.RED}Unknown command: {command}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Type '/help' for available commands{Style.RESET_ALL}")

                continue

            # Regular message - send to API
            print(f"\n{Fore.MAGENTA}Assistant: {Style.RESET_ALL}", end="", flush=True)

            response_text = ""
            for chunk in agent.call_api(user_input):
                print(chunk, end="", flush=True)
                response_text += chunk

            print("\n")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Use '/quit' to exit gracefully{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="OpenAI O1-Series Chat Agent - Advanced Reasoning AI Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --agent-id my-agent                    # Start interactive chat
  %(prog)s --list                                 # List all agents
  %(prog)s --agent-id my-agent --export html      # Export conversation as HTML
  %(prog)s --agent-id my-agent --config          # Configure agent interactively
        """
    )

    parser.add_argument("--agent-id", help="Agent ID for the chat session")
    parser.add_argument("--list", action="store_true", help="List all available agents")
    parser.add_argument("--info", metavar="ID", help="Show detailed information for agent")
    parser.add_argument("--config", action="store_true", help="Configure agent interactively")
    parser.add_argument("--temperature", type=float, help="Override temperature (0.0-2.0)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--export", choices=["json", "txt", "md", "html"], help="Export conversation format")

    args = parser.parse_args()

    # Handle list command
    if args.list:
        agents = list_agents()
        if not agents:
            print(f"{Fore.YELLOW}No agents found{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}Available Agents:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'ID':<20} {'Model':<10} {'Messages':<10} {'Last Updated':<20}")
        print("-" * 70)

        for agent in agents:
            updated = agent.get("updated_at", "Unknown")
            if updated != "Unknown":
                try:
                    updated = datetime.fromisoformat(updated).strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            model = agent.get('model', 'o1')
            model_display = OpenAIOSeriesChatAgent.SUPPORTED_MODELS.get(model, {}).get('name', model)
            print(f"{agent['id']:<20} {model_display:<10} {agent.get('message_count', 0):<10} {updated:<20}")

        return

    # Handle info command
    if args.info:
        show_agent_info(args.info)
        return

    # Require agent-id for other operations
    if not args.agent_id:
        parser.print_help()
        print(f"\n{Fore.RED}Error: --agent-id is required{Style.RESET_ALL}")
        return

    try:
        # Initialize agent
        agent = OpenAIOSeriesChatAgent(args.agent_id)

        # Handle config command
        if args.config:
            new_config = create_agent_config_interactive()
            agent.config = new_config
            agent._save_config()
            print(f"{Fore.GREEN}Configuration saved{Style.RESET_ALL}")
            return

        # Handle export command
        if args.export:
            filepath = agent.export_conversation(args.export)
            print(f"{Fore.GREEN}Exported to: {filepath}{Style.RESET_ALL}")
            return

        # Apply command line overrides
        overrides = {}
        if args.temperature is not None:
            if 0.0 <= args.temperature <= 2.0:
                overrides["temperature"] = args.temperature
            else:
                print(f"{Fore.RED}Temperature out of range (0.0-2.0), ignoring...{Style.RESET_ALL}")
        if args.no_stream:
            overrides["stream"] = False

        # Start interactive chat
        if overrides:
            updated_config = AgentConfig(**{**asdict(agent.config), **overrides})
            agent.config = updated_config
            agent._save_config()

        interactive_chat(agent)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()

