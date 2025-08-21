#!/usr/bin/env python3
"""
Anthropic Claude 4 Chat Agent - Production-Ready Chat Interface

Cette application fournit une implémentation complète d'un agent de chat utilisant le modèle Claude Sonnet 4 d'Anthropic
en utilisant l'API standard de completions d'Anthropic. Les fonctionnalités incluent :

- Historique de conversation persistant avec sauvegardes incrémentielles
- Support des réponses en streaming et non-streaming
- Inclusion de fichiers dans les messages via la syntaxe {filename} avec support des fichiers de programmation
- Gestion avancée de la configuration
- Journalisation et statistiques complètes
- Capacité d'exportation (JSON, TXT, MD, HTML)
- Gestion sécurisée des clés API
- Interface CLI interactive avec sortie colorée

Exemples d'utilisation :
    # Démarrer le chat interactif
    python anthropic_claude_agent.py --agent-id mon-agent

    # Lister tous les agents
    python anthropic_claude_agent.py --list

    # Exporter la conversation
    python anthropic_claude_agent.py --agent-id mon-agent --export html

    # Configurer les paramètres de l'agent
    python anthropic_claude_agent.py --agent-id mon-agent --config
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
from typing import Optional, Generator, List, Dict, Any
from pathlib import Path
from datetime import datetime
from requests.exceptions import RequestException, Timeout

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    # Fallback si colorama n'est pas disponible
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""


@dataclass
class AgentConfig:
    """Paramètres de configuration pour l'agent de chat Claude Sonnet 4 d'Anthropic"""
    model: str = "claude-sonnet-4-20250514"  # Modèle par défaut mis à jour
    temperature: float = 1.0
    max_tokens: Optional[int] = 64000  # Mis à jour pour correspondre à la commande curl
    max_history_size: int = 1000
    stream: bool = True
    system_prompt: Optional[str] = "You are an agent "
    response_format: str = "json"  # Output format souvent en JSON pour les API
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now


class AnthropicClaudeChatAgent:
    """Agent de chat Claude Sonnet 4 d'Anthropic avec persistance et support du streaming"""

    SUPPORTED_MODEL = {
        "claude-sonnet-4-20250514": {
            "name": "Claude Sonnet 4",
            "description": "Modèle Claude Sonnet 4 d'Anthropic",
            "timeout": 300,  # 5 minutes pour les très longues réponses
            "max_output_tokens": 64000  # Max tokens supported by this model
        },
        "claude-3-7-sonnet-20250219": {  # Modèle existant en option
            "name": "Claude 3.7 Sonnet",
            "description": "Modèle Claude 3.7 de type Sonnet d'Anthropic",
            "timeout": 300,  # 5 minutes pour les très longues réponses
            "max_output_tokens": 64000  # Max tokens supported by this model
        },
        # Ajouter d'autres versions de Claude si nécessaire
    }

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
        self.api_url = "https://api.anthropic.com/v1/messages"  # URL de l'API Claude

        # Créer la structure de répertoires
        self._setup_directories()

        # Configurer le logging
        self._setup_logging()

        # Charger ou créer la configuration
        self.config = self._load_config()

        # Charger l'historique des conversations
        self.messages = self._load_history()

        # Configurer la clé API
        self.api_key = self._get_api_key()

        self.logger.info(f"Initialized Anthropic Claude Chat Agent: {agent_id}")

    def _setup_directories(self):
        """Créer la structure de répertoires nécessaire"""
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
        """Configurer le logging vers un fichier et la console"""
        log_file = self.base_dir / "logs" / f"{datetime.now().strftime('%Y-%m-%d')}.log"

        self.logger = logging.getLogger(f"AnthropicClaudeAgent_{self.agent_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self) -> AgentConfig:
        """Charger la configuration de l'agent depuis config.yaml"""
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
        """Sauvegarder la configuration de l'agent dans config.yaml"""
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
        """Obtenir la clé API depuis l'environnement ou le fichier secrets, demander si nécessaire"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.logger.info("Using API key from environment variable")
            return api_key

        secrets_file = self.base_dir / "secrets.json"
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
                    api_key = secrets.get('default')
                    if api_key:
                        self.logger.info("Using API key from secrets file")
                        return api_key
            except Exception as e:
                self.logger.error(f"Error reading secrets file: {e}")

        # Demander à l'utilisateur la clé API
        model_display = self.SUPPORTED_MODEL[self.config.model]["name"]
        print(f"{Fore.YELLOW}Clé API non trouvée pour le modèle Anthropic {model_display}.{Style.RESET_ALL}")
        print(f"Vous pouvez définir la variable d'environnement ANTHROPIC_API_KEY ou la saisir maintenant.{Style.RESET_ALL}")

        api_key = input(f"{Fore.CYAN}Entrez la clé API pour Anthropic {model_display}: {Style.RESET_ALL}").strip()

        if not api_key:
            raise ValueError("La clé API est requise")

        # Sauvegarder dans le fichier secrets
        secrets = {
            "provider": "anthropic",
            "keys": {
                "default": api_key
            }
        }

        try:
            with open(secrets_file, 'w') as f:
                json.dump(secrets, f, indent=2)

            # Ajouter au .gitignore
            gitignore_file = Path('.gitignore')
            gitignore_content = ""
            if gitignore_file.exists():
                gitignore_content = gitignore_file.read_text()

            if 'secrets.json' not in gitignore_content:
                with open(gitignore_file, 'a') as f:
                    f.write('\n# API Keys\n**/secrets.json\nsecrets.json\n')

            masked_key = f"{api_key[:4]}...{api_key[-2:]}" if len(api_key) > 6 else "***"
            print(f"{Fore.GREEN}Clé API sauvegardée ({masked_key}){Style.RESET_ALL}")
            self.logger.info(f"Clé API sauvegardée pour l'utilisateur (longueur: {len(api_key)})")

        except Exception as e:
            self.logger.error(f"Error saving API key: {e}")
            print(f"{Fore.RED}Attention : Impossible de sauvegarder la clé API dans le fichier{Style.RESET_ALL}")

        return api_key

    def _load_history(self) -> List[Dict[str, Any]]:
        """Charger l'historique des conversations depuis history.json"""
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
        """Sauvegarder l'historique des conversations dans history.json avec sauvegarde incrémentielle"""
        history_file = self.base_dir / "history.json"

        if history_file.exists():
            self._create_backup()

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")

    def _create_backup(self):
        """Créer une sauvegarde incrémentielle de l'historique"""
        history_file = self.base_dir / "history.json"
        backup_dir = self.base_dir / "backups"

        if not history_file.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"history_{timestamp}.json"

        try:
            shutil.copy2(history_file, backup_file)

            # Garder seulement les 10 dernières sauvegardes
            backups = sorted(backup_dir.glob("history_*.json"))
            while len(backups) > 10:
                oldest = backups.pop(0)
                oldest.unlink()

        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Ajouter un message à l'historique des conversations"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.messages.append(message)

        if len(self.messages) > self.config.max_history_size:
            removed = self.messages[:-self.config.max_history_size]
            self.messages = self.messages[-self.config.max_history_size:]
            self.logger.info(f"Truncated history: removed {len(removed)} old messages")

        self._save_history()

    def _is_supported_file(self, file_path: Path) -> bool:
        """Vérifier si l'extension du fichier est supportée pour l'inclusion"""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _process_file_inclusions(self, content: str) -> str:
        """Remplacer les motifs {filename} par le contenu du fichier"""
        def replace_file(match):
            filename = match.group(1).strip()

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
                    if not self._is_supported_file(file_path):
                        self.logger.warning(f"Unsupported file type: {filename}")
                        return f"[WARNING: Unsupported file type {filename}]"

                    try:
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
        """Construire le payload de la requête API"""
        processed_message = self._process_file_inclusions(new_message)

        messages = []

        # For Anthropic API, system prompt is separate from messages
        # Don't include system message in messages array

        for msg in self.messages:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        messages.append({
            "role": "user",
            "content": processed_message
        })

        config = asdict(self.config)
        if override_config:
            config.update(override_config)

        # Ensure max_tokens doesn't exceed model limit
        max_tokens = config["max_tokens"]
        model_info = self.SUPPORTED_MODEL.get(config["model"], {})
        max_output_tokens = model_info.get("max_output_tokens", 64000)

        if max_tokens > max_output_tokens:
            max_tokens = max_output_tokens
            self.logger.info(f"Adjusted max_tokens from {config['max_tokens']} to {max_tokens} (model limit)")

        payload = {
            "model": config["model"],
            "max_tokens": max_tokens,
            "temperature": config["temperature"],
            "messages": messages
        }

        # Add system prompt if available (separate from messages for Anthropic)
        if self.config.system_prompt and self.config.system_prompt.strip():
            payload["system"] = self.config.system_prompt.strip()

        # Add streaming if enabled
        if config["stream"]:
            payload["stream"] = True

        # Add top_p if different from default
        if config.get("top_p") and config["top_p"] != 1.0:
            payload["top_p"] = config["top_p"]

        return payload

    def _get_timeout_for_model(self) -> int:
        """Obtenir le timeout basé sur le modèle"""
        return self.SUPPORTED_MODEL[self.config.model]["timeout"]

    def _make_api_request(self, payload: Dict[str, Any]) -> requests.Response:
        """Faire la requête API avec des retries et gestion des erreurs"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        timeout = self._get_timeout_for_model()
        self.logger.info(f"Using timeout of {timeout}s for {self.SUPPORTED_MODEL[self.config.model]['name']}")

        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Making API request (attempt {attempt + 1}/{max_retries}) with {timeout}s timeout...")

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
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Rate limited, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                elif response.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Server error {response.status_code}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    # Log the response content for debugging
                    try:
                        error_content = response.text[:500] if response.text else "No response text"
                        self.logger.error(f"API request failed with status {response.status_code}: {error_content}")
                    except:
                        self.logger.error(f"API request failed with status {response.status_code}")
                    response.raise_for_status()

            except Timeout as e:
                self.logger.warning(f"Request timed out after {timeout}s (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise Exception(f"Request timed out after {timeout}s.")
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Retrying in {delay}s...")
                time.sleep(delay)
            except RequestException as e:
                self.logger.warning(f"Request exception: {e}")
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Request failed ({e}), retrying in {delay}s...")
                time.sleep(delay)

        raise Exception(f"Failed to complete API request after {max_retries} attempts")

    def _parse_streaming_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Parser la réponse en streaming d'Anthropic"""
        accumulated_text = ""
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8').strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Parse Server-Sent Events format
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                        
                    try:
                        event = json.loads(data_str)
                        
                        if event.get("type") == "message_start":
                            continue
                        elif event.get("type") == "content_block_start":
                            continue
                        elif event.get("type") == "content_block_delta":
                            delta_text = event.get("delta", {}).get("text", "")
                            if delta_text:
                                accumulated_text += delta_text
                                yield delta_text
                        elif event.get("type") == "content_block_stop":
                            continue
                        elif event.get("type") == "message_delta":
                            continue
                        elif event.get("type") == "message_stop":
                            break
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON in stream: {data_str} - {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error parsing streaming response: {e}")

        # Save the complete assistant message
        if accumulated_text.strip():
            self.add_message("assistant", accumulated_text)

    def _parse_non_streaming_response(self, response: requests.Response) -> str:
        """Parser la réponse non-streaming d'Anthropic"""
        try:
            data = response.json()
            
            # Anthropic returns content in a different format
            content_blocks = data.get("content", [])
            if content_blocks and len(content_blocks) > 0:
                # Get text from the first content block
                text_content = content_blocks[0].get("text", "")
                if text_content:
                    self.add_message("assistant", text_content)
                    return text_content
            
            # Fallback: check if there's a direct message field
            message = data.get("message", "")
            if message:
                self.add_message("assistant", message)
                return message
                
            return "No response content received"
            
        except Exception as e:
            self.logger.error(f"Error parsing non-streaming response: {e}")
            try:
                # Log the response content for debugging
                response_text = response.text[:500] if response.text else "No response text"
                self.logger.error(f"Response content: {response_text}")
            except:
                pass
            return f"Error parsing response: {e}"

    def call_api(self, new_message: str, override_config: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """Appeler l'API Claude Sonnet 4 d'Anthropic avec le nouveau message"""
        try:
            self.add_message("user", new_message)
            payload = self._build_api_payload(new_message, override_config)
            self.logger.info(f"Making API call to {self.api_url}")
            
            # Log payload for debugging (but hide sensitive data)
            debug_payload = payload.copy()
            if 'messages' in debug_payload and len(debug_payload['messages']) > 0:
                # Only log the structure, not the full content
                debug_payload['messages'] = f"[{len(debug_payload['messages'])} messages]"
            self.logger.info(f"Payload structure: {json.dumps(debug_payload, indent=2)}")

            model_display = self.SUPPORTED_MODEL[self.config.model]['name']
            timeout = self._get_timeout_for_model()

            print(f"{Fore.YELLOW}🤖 Using {model_display} (timeout: {timeout//60}m {timeout%60}s)...{Style.RESET_ALL}")

            response = self._make_api_request(payload)

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
        """Effacer l'historique des conversations"""
        self._create_backup()
        self.messages.clear()
        self._save_history()
        self.logger.info("Conversation history cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques de la conversation"""
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
        """Exporter la conversation dans le format spécifié"""
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

            model_display = self.SUPPORTED_MODEL[self.config.model]['name']

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Anthropic {model_display} Chat Agent Conversation Export\n")
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

            model_display = self.SUPPORTED_MODEL[self.config.model]['name']

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Anthropic {model_display} Chat Agent Conversation\n\n")
                f.write(f"**Agent ID:** {self.agent_id}  \n")
                f.write(f"**Model:** {self.config.model}  \n")
                f.write(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

                for msg in self.messages:
                    timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    role_emoji = "🧑" if msg["role"] == "user" else "🤖"
                    f.write(f"## {role_emoji} {msg['role'].title()} - {timestamp}\n\n")
                    f.write(f"{msg['content']}\n\n")

        elif format_type == "html":
            filename = f"conversation_{timestamp}.html"
            filepath = export_dir / filename

            stats = self.get_statistics()
            model_display = self.SUPPORTED_MODEL[self.config.model]['name']
            # HTML template avec style de base - partie statique
            html_template = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anthropic {model_display} Conversation - {self.agent_id}</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --secondary-color: #f1f5f9;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
            --user-bg: #3b82f6;
            --assistant-bg: #10b981;
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
            background: #f0fdf4;
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
            <h1>🤖 Anthropic {model_display} Chat Agent</h1>
            <p>Conversation Export</p>
            <div class="header-info">
                <div><strong>Agent ID:</strong> {self.agent_id}</div>
                <div><strong>Model:</strong> {self.config.model}</div>
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

        <div class="messages">
"""

        # Generate messages dynamically
        for msg in self.messages:
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            role = msg["role"]
            content = msg["content"]

            import html
            content_escaped = html.escape(content)

            if '```' in content_escaped:
                parts = content_escaped.split('```')
                formatted_content = ""
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        formatted_content += f'<div class="code-block">{part}</div>'
                    else:
                        formatted_content += part
                content_escaped = formatted_content

            avatar_text = "U" if role == "user" else "AI"

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

        # Close the HTML structure
        html_template += f"""
        </div>

        <div class="footer">
            Generated by Anthropic {model_display} Chat Agent • Agent ID: {self.agent_id} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

        # Écrire le fichier HTML si demandé
        if format_type == "html":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_template)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        self.logger.info(f"Exported conversation to {filepath}")
        return str(filepath)

    def search_history(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Rechercher dans l'historique des conversations un terme"""
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
        """Lister les fichiers disponibles pour inclusion"""
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

    def _validate_and_update_config(self, override_config: Optional[Dict[str, Any]] = None):
        """Valider et mettre à jour la configuration"""
        if override_config:
            for key, value in override_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    self.logger.warning(f"Unknown configuration key: {key}")
            self.config.updated_at = datetime.now().isoformat()
            self._save_config()

    def interactive_chat(self):
        """Session de chat interactive"""
        model_display = self.SUPPORTED_MODEL.get(self.config.model, {}).get('name', self.config.model)

        print(f"\n{Fore.GREEN}Démarrage du chat interactif avec Claude Sonnet 4 d'Anthropic")
        print(f"Agent: {Fore.YELLOW}{self.agent_id}")
        print(f"{Fore.GREEN}Tapez '/help' pour les commandes, '/quit' pour quitter{Style.RESET_ALL}\n")

        while True:
            try:
                user_input = input(f"{Fore.CYAN}Vous: {Style.RESET_ALL}").strip()

                if not user_input:
                    continue

                if user_input.startswith('/'):
                    command_parts = user_input[1:].split()
                    command = command_parts[0].lower()

                    if command == 'help':
                        print(f"\n{Fore.YELLOW}Commandes Disponibles:")
                        print(f"{Fore.WHITE}/help - Afficher ce message d'aide")
                        print(f"/history [n] - Afficher les derniers n messages (par défaut 5)")
                        print(f"/search <terme> - Rechercher dans l'historique de conversation")
                        print(f"/stats - Afficher les statistiques de la conversation")
                        print(f"/config - Afficher la configuration actuelle")
                        print(f"/export <json|txt|md|html> - Exporter la conversation")
                        print(f"/clear - Effacer l'historique de conversation")
                        print(f"/files - Lister les fichiers disponibles pour inclusion")
                        print(f"/info - Afficher les informations de l'agent")
                        print(f"/quit - Quitter le chat{Style.RESET_ALL}\n")
                        print(f"{Fore.CYAN}Inclusion de Fichiers: Utilisez {{filename}} dans vos messages pour inclure le contenu des fichiers")
                        print(f"Supporté: Fichiers de programmation (.py, .r, .js, etc.), fichiers de configuration, documentation{Style.RESET_ALL}\n")
                        continue

                    elif command == 'history':
                        limit = 5
                        if len(command_parts) > 1:
                            try:
                                limit = int(command_parts[1])
                            except ValueError:
                                print(f"{Fore.RED}Nombre invalide{Style.RESET_ALL}")
                                continue

                        recent_messages = self.messages[-limit:]
                        if not recent_messages:
                            print(f"{Fore.YELLOW}Aucun message dans l'historique{Style.RESET_ALL}")
                        else:
                            print(f"\n{Fore.YELLOW}Derniers {len(recent_messages)} messages:")
                            for msg in recent_messages:
                                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                                role_color = Fore.CYAN if msg["role"] == "user" else Fore.GREEN
                                print(f"{Fore.WHITE}[{timestamp}] {role_color}{msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                        print()

                    elif command == 'search':
                        if len(command_parts) < 2:
                            print(f"{Fore.RED}Utilisation: /search <terme>{Style.RESET_ALL}")
                            continue

                        search_term = ' '.join(command_parts[1:])
                        results = self.search_history(search_term)

                        if not results:
                            print(f"{Fore.YELLOW}Aucune correspondance trouvée pour '{search_term}'{Style.RESET_ALL}")
                        else:
                            print(f"\n{Fore.YELLOW}Trouvé {len(results)} correspondances pour '{search_term}':")
                            for result in results:
                                msg = result["message"]
                                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                                role_color = Fore.CYAN if msg["role"] == "user" else Fore.GREEN
                                print(f"{Fore.WHITE}[{timestamp}] {role_color}{msg['role']}: {result['preview']}")
                        print()

                    elif command == 'stats':
                        stats = self.get_statistics()
                        print(f"\n{Fore.YELLOW}Statistiques de la Conversation:")
                        print(f"{Fore.WHITE}Modèle: {self.config.model} ({model_display})")
                        print(f"Total des Messages: {stats['total_messages']}")
                        print(f"Messages de l'Utilisateur: {stats['user_messages']}")
                        print(f"Messages de l'Assistant: {stats['assistant_messages']}")
                        print(f"Total de Caractères: {stats['total_characters']:,}")
                        print(f"Longueur Moyenne des Messages: {stats['average_message_length']:,}")
                        if stats['first_message']:
                            print(f"Premier Message: {stats['first_message']}")
                            print(f"Dernier Message: {stats['last_message']}")
                            print(f"Durée: {stats['conversation_duration']}")
                        print()

                    elif command == 'config':
                        print(f"\n{Fore.YELLOW}Configuration Actuelle:")
                        config_dict = asdict(self.config)
                        for key, value in config_dict.items():
                            if key not in ['created_at', 'updated_at']:
                                if key == 'model':
                                    model_name = self.SUPPORTED_MODEL.get(str(value), {}).get('name', value)
                                    print(f"{Fore.WHITE}{key}: {value} ({model_name})")
                                else:
                                    print(f"{Fore.WHITE}{key}: {value}")
                        print()

                    elif command == 'export':
                        if len(command_parts) < 2:
                            print(f"{Fore.RED}Utilisation: /export <json|txt|md|html>{Style.RESET_ALL}")
                            continue

                        format_type = command_parts[1].lower()
                        if format_type not in ['json', 'txt', 'md', 'html']:
                            print(f"{Fore.RED}Format invalide. Utilisez: json, txt, md, ou html{Style.RESET_ALL}")
                            continue

                        try:
                            filepath = self.export_conversation(format_type)
                            print(f"{Fore.GREEN}Exporté vers: {filepath}{Style.RESET_ALL}")
                        except Exception as e:
                            print(f"{Fore.RED}Échec de l'exportation: {e}{Style.RESET_ALL}")

                    elif command == 'clear':
                        confirm = input(f"{Fore.YELLOW}Effacer l'historique de la conversation? (y/N): {Style.RESET_ALL}").strip().lower()
                        if confirm in ['y', 'yes']:
                            self.clear_history()
                            print(f"{Fore.GREEN}Historique de conversation effacé{Style.RESET_ALL}")

                    elif command == 'files':
                        files = self.list_files()
                        if not files:
                            print(f"{Fore.YELLOW}Aucun fichier supporté trouvé pour l'inclusion{Style.RESET_ALL}")
                        else:
                            print(f"\n{Fore.YELLOW}Fichiers Disponibles pour Inclusion:")
                            for file_info in files[:20]:
                                print(f"{Fore.WHITE}{file_info}")
                            if len(files) > 20:
                                print(f"{Fore.YELLOW}... et {len(files) - 20} autres fichiers")
                        print(f"{Fore.CYAN}Utilisez {{filename}} dans votre message pour inclure le contenu des fichiers{Style.RESET_ALL}\n")

                    elif command == 'info':
                        AnthropicClaudeChatAgent.show_agent_info(self.agent_id)

                    elif command in ['quit', 'exit', 'q']:
                        print(f"{Fore.GREEN}Au revoir!{Style.RESET_ALL}")
                        break

                    else:
                        print(f"{Fore.RED}Commande inconnue: {command}{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Tapez '/help' pour les commandes disponibles{Style.RESET_ALL}")

                    continue

                # Message régulier - envoyer à l'API
                print(f"\n{Fore.GREEN}Assistant: {Style.RESET_ALL}", end="", flush=True)

                response_text = ""
                for chunk in self.call_api(user_input):
                    print(chunk, end="", flush=True)
                    response_text += chunk

                print("\n")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Utilisez '/quit' pour quitter proprement{Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}Erreur: {e}{Style.RESET_ALL}")

    @staticmethod
    def list_agents() -> List[Dict[str, Any]]:
        """Lister tous les agents disponibles"""
        agents_dir = Path("agents")
        agents = []

        if not agents_dir.exists():
            return agents

        for agent_dir in agents_dir.iterdir():
            if agent_dir.is_dir():
                metadata_file = agent_dir / "metadata.json"
                config_file = agent_dir / "config.yaml"
                history_file = agent_dir / "history.json"

                agent_info = {
                    "id": agent_dir.name,
                    "path": str(agent_dir),
                    "exists": True
                }

                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = yaml.safe_load(f)
                            agent_info["model"] = config.get("model", "claude-sonnet-4-20250514")
                            agent_info["created_at"] = config.get("created_at")
                            agent_info["updated_at"] = config.get("updated_at")
                    except:
                        pass

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

    @staticmethod
    def show_agent_info(agent_id: str):
        """Afficher les informations détaillées de l'agent"""
        agent_dir = Path(f"agents/{agent_id}")

        if not agent_dir.exists():
            print(f"{Fore.RED}Agent '{agent_id}' non trouvé{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"Informations de l'Agent: {Fore.YELLOW}{agent_id}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")

        config_file = agent_dir / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)

                model = config.get('model', 'claude-sonnet-4-20250514')
                model_display = AnthropicClaudeChatAgent.SUPPORTED_MODEL.get(model, {}).get('name', model)

                print(f"\n{Fore.GREEN}Configuration:")
                print(f"{Fore.WHITE}  Modèle: {model} ({model_display})")
                print(f"  Température: {config.get('temperature', 1.0)}")
                print(f"  Max Tokens: {config.get('max_tokens', 64000)}")
                print(f"  Streaming: {config.get('stream', True)}")
                print(f"  Créé: {config.get('created_at', 'Unknown')}")
                print(f"  Mis à jour: {config.get('updated_at', 'Unknown')}")

            except Exception as e:
                print(f"{Fore.RED}Erreur lors du chargement de la config: {e}")

        history_file = agent_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)

                user_msgs = len([m for m in history if m.get("role") == "user"])
                assistant_msgs = len([m for m in history if m.get("role") == "assistant"])
                total_chars = sum(len(m.get("content", "")) for m in history)

                print(f"\n{Fore.GREEN}Historique de Conversation:")
                print(f"{Fore.WHITE}  Total des Messages: {len(history)}")
                print(f"  Messages de l'Utilisateur: {user_msgs}")
                print(f"  Messages de l'Assistant: {assistant_msgs}")
                print(f"  Total de Caractères: {total_chars:,}")
                print(f"  Taille du Fichier: {history_file.stat().st_size:,} bytes")

                if history:
                    first_msg = datetime.fromisoformat(history[0]["timestamp"])
                    last_msg = datetime.fromisoformat(history[-1]["timestamp"])
                    print(f"  Premier Message: {first_msg.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Dernier Message: {last_msg.strftime('%Y-%m-%d %H:%M:%S')}")

            except Exception as e:
                print(f"{Fore.RED}Erreur lors du chargement de l'historique: {e}")
        else:
            print(f"\n{Fore.YELLOW}Aucun historique de conversation trouvé{Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}Structure des Répertoires:")
        for item in sorted(agent_dir.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                size_str = f"{size:,}" if size < 1024 else f"{size/1024:.1f}K"
                rel_path = item.relative_to(agent_dir)
                print(f"{Fore.WHITE}  {rel_path} ({size_str} bytes)")

    @staticmethod
    def create_agent_config_interactive() -> AgentConfig:
        """Création interactive de la configuration de l'agent"""
        print(f"\n{Fore.CYAN}Création de la Configuration de l'Agent{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Appuyez sur Entrée pour utiliser les valeurs par défaut{Style.RESET_ALL}\n")

        config = AgentConfig()

        # Temperature
        temp_input = input(f"Température (0.0-2.0) [{config.temperature}]: ").strip()
        if temp_input:
            try:
                temp = float(temp_input)
                if 0.0 <= temp <= 2.0:
                    config.temperature = temp
                else:
                    print(f"{Fore.RED}Température invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Température invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")

        # System prompt
        system_prompt = input(f"System prompt (optionnel): ").strip()
        if system_prompt:
            config.system_prompt = system_prompt

        # Max completion tokens
        tokens_input = input(f"Max completion tokens [{config.max_tokens}]: ").strip()
        if tokens_input:
            try:
                tokens = int(tokens_input)
                config.max_tokens = tokens
            except ValueError:
                print(f"{Fore.RED}Nombre de tokens invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")

        # Streaming
        stream_input = input(f"Activer le streaming (y/n) [{'y' if config.stream else 'n'}]: ").strip().lower()
        if stream_input in ['n', 'no', 'false']:
            config.stream = False
        elif stream_input in ['y', 'yes', 'true']:
            config.stream = True

        # Top P
        top_p_input = input(f"Top P (0.0-1.0) [{config.top_p}]: ").strip()
        if top_p_input:
            try:
                top_p = float(top_p_input)
                if 0.0 <= top_p <= 1.0:
                    config.top_p = top_p
                else:
                    print(f"{Fore.RED}Top P invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Top P invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")

        # Frequency penalty
        freq_penalty_input = input(f"Frequency penalty (0.0-2.0) [{config.frequency_penalty}]: ").strip()
        if freq_penalty_input:
            try:
                freq_penalty = float(freq_penalty_input)
                if 0.0 <= freq_penalty <= 2.0:
                    config.frequency_penalty = freq_penalty
                else:
                    print(f"{Fore.RED}Frequency penalty invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Frequency penalty invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")

        # Presence penalty
        pres_penalty_input = input(f"Presence penalty (0.0-2.0) [{config.presence_penalty}]: ").strip()
        if pres_penalty_input:
            try:
                pres_penalty = float(pres_penalty_input)
                if 0.0 <= pres_penalty <= 2.0:
                    config.presence_penalty = pres_penalty
                else:
                    print(f"{Fore.RED}Presence penalty invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Presence penalty invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")

        return config

    @staticmethod
    def main():
        """Point d'entrée principal de l'agent"""
        parser = argparse.ArgumentParser(
            description="Anthropic Claude Sonnet 4 Chat Agent - Advanced AI Chat Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --agent-id mon-agent                    # Start interactive chat
  %(prog)s --list                                 # List all agents
  %(prog)s --agent-id mon-agent --export html      # Export conversation as HTML
  %(prog)s --agent-id mon-agent --config          # Configure agent interactively
            """
        )

        parser.add_argument("--agent-id", help="ID de l'agent pour la session de chat")
        parser.add_argument("--list", action="store_true", help="Lister tous les agents disponibles")
        parser.add_argument("--info", metavar="ID", help="Afficher des informations détaillées pour un agent")
        parser.add_argument("--config", action="store_true", help="Configurer l'agent de manière interactive")
        parser.add_argument("--temperature", type=float, help="Surcharger la température (0.0-2.0)")
        parser.add_argument("--no-stream", action="store_true", help="Désactiver le streaming")
        parser.add_argument("--export", choices=["json", "txt", "md", "html"], help="Exporter la conversation dans le format spécifié")

        args = parser.parse_args()

        if args.list:
            agents = AnthropicClaudeChatAgent.list_agents()
            if not agents:
                print(f"{Fore.YELLOW}Aucun agent trouvé{Style.RESET_ALL}")
                return

            print(f"\n{Fore.CYAN}Agents Disponibles:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'ID':<25} {'Modèle':<30} {'Messages':<10} {'Dernière Mise à Jour':<25}")
            print("-" * 90)

            for agent in agents:
                updated = agent.get("updated_at", "Unknown")
                if updated != "Unknown":
                    try:
                        updated = datetime.fromisoformat(updated).strftime("%Y-%m-%d %H:%M")
                    except:
                        pass

                model = agent.get('model', 'claude-sonnet-4-20250514')
                model_display = AnthropicClaudeChatAgent.SUPPORTED_MODEL.get(model, {}).get('name', model)
                print(f"{agent['id']:<25} {model_display:<30} {agent.get('message_count', 0):<10} {updated:<25}")

            return

        if args.info:
            AnthropicClaudeChatAgent.show_agent_info(args.info)
            return

        if not args.agent_id:
            parser.print_help()
            print(f"\n{Fore.RED}Erreur: --agent-id est requis{Style.RESET_ALL}")
            return

        try:
            agent = AnthropicClaudeChatAgent(args.agent_id)

            if args.config:
                new_config = AnthropicClaudeChatAgent.create_agent_config_interactive()
                agent.config = new_config
                agent._save_config()
                print(f"{Fore.GREEN}Configuration sauvegardée{Style.RESET_ALL}")
                return

            if args.export:
                filepath = agent.export_conversation(args.export)
                print(f"{Fore.GREEN}Exporté vers: {filepath}{Style.RESET_ALL}")
                return

            overrides = {}
            if args.temperature is not None:
                overrides["temperature"] = args.temperature
            if args.no_stream:
                overrides["stream"] = False

            if overrides:
                agent._validate_and_update_config(overrides)

            agent.interactive_chat()

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrompu par l'utilisateur{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Erreur: {e}{Style.RESET_ALL}")
            sys.exit(1)


if __name__ == "__main__":
    AnthropicClaudeChatAgent.main()

