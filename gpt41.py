#!/usr/bin/env python3  
"""  
OpenAI GPT-4.1 Chat Agent - Interface de Chat Avancée Prête pour la Production

Ce module fournit une implémentation complète d'un agent de chat pour le modèle GPT-4.1  
d'OpenAI en utilisant l'endpoint standard des complétions de chat. Les fonctionnalités incluent :

- Historique de conversation persistant avec sauvegardes automatiques  
- Support des réponses en streaming et non-streaming  
- Inclusion de fichiers dans les messages via la syntaxe {filename} avec support de fichiers de programmation  
- Gestion avancée de la configuration  
- Journalisation complète et statistiques  
- Capacités d'exportation (JSON, TXT, MD, HTML)  
- Gestion sécurisée des clés d'API  
- Interface CLI interactive avec sortie colorée

Exemples d'utilisation :  
    # Démarrer un chat interactif  
    python openai_gpt41_agent.py --agent-id mon-agent

    # Lister tous les agents  
    python openai_gpt41_agent.py --list

    # Exporter la conversation  
    python openai_gpt41_agent.py --agent-id mon-agent --export html

    # Configurer les paramètres de l'agent  
    python openai_gpt41_agent.py --agent-id mon-agent --config  
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
from requests.exceptions import RequestException, HTTPError, Timeout

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
    """Paramètres de configuration pour l'Agent de Chat OpenAI GPT-4.1"""  
    model: str = "gpt-4.1"  
    temperature: float = 1.0  
    max_tokens: Optional[int] = 32768  # Renommé de max_completion_tokens
    max_history_size: int = 1000  
    stream: bool = True  
    system_prompt: Optional[str] = None  
    # response_format: str = "text"  # Supprimé car non reconnu par l'API standard
    top_p: float = 1.0  
    frequency_penalty: float = 0.0  
    presence_penalty: float = 0.0  
    created_at: str = ""  
    updated_at: str = ""

    def __post_init__(self):  
        if not self.created_at:  
            self.created_at = datetime.now().isoformat()  
        self.updated_at = datetime.now().isoformat()

class OpenAIGPT41ChatAgent:  
    """Agent de Chat OpenAI GPT-4.1 avec persistance et support de streaming"""

    SUPPORTED_EXTENSIONS = {  
        # Langages de programmation  
        '.py', '.r', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.cxx',  
        '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',  
        '.clj', '.hs', '.ml', '.fs', '.vb', '.pl', '.pm', '.sh', '.bash', '.zsh', '.fish',  
        '.ps1', '.bat', '.cmd', '.sql', '.html', '.htm', '.css', '.scss', '.sass', '.less',  
        '.xml', '.xsl', '.xslt', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',  
        '.properties', '.env', '.dockerfile', '.docker', '.makefile', '.cmake', '.gradle',  
        '.sbt', '.pom', '.lock', '.mod', '.sum',

        # Données et balisage  
        '.md', '.markdown', '.rst', '.tex', '.latex', '.csv', '.tsv', '.jsonl', '.ndjson',  
        '.svg', '.rss', '.atom', '.plist',

        # Configuration et infrastructure  
        '.tf', '.tfvars', '.hcl', '.nomad', '.consul', '.vault', '.k8s', '.kubectl',  
        '.helm', '.kustomize', '.ansible', '.inventory', '.playbook',

        # Documentation et texte  
        '.txt', '.log', '.out', '.err', '.trace', '.debug', '.info', '.warn', '.error',  
        '.readme', '.license', '.changelog', '.authors', '.contributors', '.todo',

        # Notebooks et scripts  
        '.ipynb', '.rmd', '.qmd', '.jl', '.m', '.octave', '.R', '.Rmd',

        # Web et API  
        '.graphql', '.gql', '.rest', '.http', '.api', '.postman', '.insomnia',

        # Autres formats utiles  
        '.editorconfig', '.gitignore', '.gitattributes', '.dockerignore', '.eslintrc',  
        '.prettierrc', '.babelrc', '.webpack', '.rollup', '.vite', '.parcel'  
    }

    def __init__(self, agent_id: str):  
        self.agent_id = agent_id  
        self.base_dir = Path(f"agents/{agent_id}")  
        self.api_url = "https://api.openai.com/v1/chat/completions"

        # Créer la structure des répertoires  
        self._setup_directories()

        # Configurer la journalisation  
        self._setup_logging()

        # Charger ou créer la configuration  
        self.config = self._load_config()

        # Charger l'historique de conversation  
        self.messages = self._load_history()

        # Configurer la clé API  
        self.api_key = self._get_api_key()

        self.logger.info(f"Agent de Chat OpenAI GPT-4.1 initialisé : {agent_id}")

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
        """Configurer la journalisation vers un fichier et la console"""  
        log_file = self.base_dir / "logs" / f"{datetime.now().strftime('%Y-%m-%d')}.log"

        # Créer le logger  
        self.logger = logging.getLogger(f"OpenAIGPT41Agent_{self.agent_id}")  
        self.logger.setLevel(logging.INFO)

        # Supprimer les handlers existants  
        self.logger.handlers.clear()

        # Handler de fichier  
        file_handler = logging.FileHandler(log_file, encoding='utf-8')  
        file_formatter = logging.Formatter(  
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  
        )  
        file_handler.setFormatter(file_formatter)

        # Handler de console  
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
                self.logger.error(f"Erreur lors du chargement de la config : {e}")  
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
            self.logger.error(f"Erreur lors de la sauvegarde de la config : {e}")

    def _get_api_key(self) -> str:  
        """Obtenir la clé API depuis les variables d'environnement ou un fichier secrets, demander si nécessaire"""  
        # Premièrement, tenter avec la variable d'environnement  
        api_key = os.getenv('OPENAI_API_KEY')  
        if api_key:  
            self.logger.info("Clé API utilisée depuis la variable d'environnement")  
            return api_key

        # Tenter avec le fichier secrets.json  
        secrets_file = self.base_dir / "secrets.json"  
        if secrets_file.exists():  
            try:  
                with open(secrets_file, 'r') as f:  
                    secrets = json.load(f)  
                    api_key = secrets.get('keys', {}).get('default')  
                    if api_key:  
                        self.logger.info("Clé API utilisée depuis le fichier secrets")  
                        return api_key  
            except Exception as e:  
                self.logger.error(f"Erreur lors de la lecture du fichier secrets : {e}")

        # Demander à l'utilisateur de saisir la clé API  
        print(f"{Fore.YELLOW}Clé API non trouvée pour le modèle OpenAI GPT-4.1.")  
        print(f"Vous pouvez définir la variable d'environnement OPENAI_API_KEY ou la saisir maintenant.{Style.RESET_ALL}")

        api_key = input(f"{Fore.CYAN}Entrez la clé API pour OpenAI GPT-4.1 : {Style.RESET_ALL}").strip()

        if not api_key:  
            raise ValueError("Une clé API est requise")

        # Sauvegarder la clé dans le fichier secrets.json  
        secrets = {  
            "provider": "openai",  
            "keys": {  
                "default": api_key  
            }  
        }

        try:  
            with open(secrets_file, 'w') as f:  
                json.dump(secrets, f, indent=2)

            # Ajouter au .gitignore  
            gitignore_file = Path('.gitignore')  
            if gitignore_file.exists():  
                gitignore_content = gitignore_file.read_text()  
            else:  
                gitignore_content = ""

            if 'secrets.json' not in gitignore_content:  
                with open(gitignore_file, 'a') as f:  
                    f.write('\n# Clés API\n**/secrets.json\nsecrets.json\n')

            masked_key = f"{api_key[:4]}...{api_key[-2:]}" if len(api_key) > 6 else "***"  
            print(f"{Fore.GREEN}Clé API sauvegardée ({masked_key}){Style.RESET_ALL}")  
            self.logger.info(f"Clé API sauvegardée pour l'utilisateur (longueur : {len(api_key)})")

        except Exception as e:  
            self.logger.error(f"Erreur lors de la sauvegarde de la clé API : {e}")  
            print(f"{Fore.RED}Attention : Impossible de sauvegarder la clé API dans le fichier{Style.RESET_ALL}")

        return api_key

    def _load_history(self) -> List[Dict[str, Any]]:  
        """Charger l'historique de conversation depuis history.json"""  
        history_file = self.base_dir / "history.json"

        if history_file.exists():  
            try:  
                with open(history_file, 'r', encoding='utf-8') as f:  
                    return json.load(f)  
            except Exception as e:  
                self.logger.error(f"Erreur lors du chargement de l'historique : {e}")  
                return []  
        return []

    def _save_history(self):  
        """Sauvegarder l'historique de conversation dans history.json avec sauvegarde automatique"""  
        history_file = self.base_dir / "history.json"

        # Créer une sauvegarde si l'historique existe  
        if history_file.exists():  
            self._create_backup()

        try:  
            with open(history_file, 'w', encoding='utf-8') as f:  
                json.dump(self.messages, f, indent=2, ensure_ascii=False)  
        except Exception as e:  
            self.logger.error(f"Erreur lors de la sauvegarde de l'historique : {e}")

    def _create_backup(self):  
        """Créer une sauvegarde roulante de l'historique"""  
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
            self.logger.error(f"Erreur lors de la création de la sauvegarde : {e}")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):  
        """Ajouter un message à l'historique de conversation"""  
        message = {  
            "role": role,  
            "content": content,  
            "timestamp": datetime.now().isoformat(),  
            "metadata": metadata or {}  
        }

        self.messages.append(message)

        # Tronquer l'historique si nécessaire  
        if len(self.messages) > self.config.max_history_size:  
            removed = self.messages[:-self.config.max_history_size]  
            self.messages = self.messages[-self.config.max_history_size:]  
            self.logger.info(f"Historique tronqué : {len(removed)} anciens messages supprimés")

        self._save_history()

    def _is_supported_file(self, file_path: Path) -> bool:  
        """Vérifier si l'extension du fichier est supportée pour inclusion"""  
        if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:  
            return True

        # Vérifier les fichiers sans extension mais avec des noms connus  
        known_files = {  
            'makefile', 'dockerfile', 'readme', 'license', 'changelog'  
        }

        return file_path.name.lower() in known_files

    def _process_file_inclusions(self, content: str) -> str:  
        """Remplacer les motifs {filename} par le contenu des fichiers"""  
        def replace_file(match):  
            filename = match.group(1)

            # Chemins de recherche  
            search_paths = [  
                Path('.'),  
                Path('src'), Path('lib'),  
                Path('scripts'), Path('data'),  
                Path('documents'), Path('files'),  
                Path('config'), Path('configs'),  
                self.base_dir / 'uploads'  
            ]

            for search_path in search_paths:  
                file_path = search_path / filename  
                if file_path.exists() and file_path.is_file():

                    # Vérifier si le fichier est supporté  
                    if not self._is_supported_file(file_path):  
                        self.logger.warning(f"Type de fichier non supporté : {filename}")  
                        return f"[WARNING : Type de fichier non supporté {filename}]"

                    try:  
                        # Vérifier la taille du fichier (limite à 2MB pour les fichiers de programmation)  
                        max_size = 2 * 1024 * 1024  # 2MB  
                        if file_path.stat().st_size > max_size:  
                            self.logger.error(f"Fichier {filename} trop volumineux (>2MB)")  
                            return f"[ERROR : Fichier {filename} trop volumineux (max 2MB)]"

                        # Essayer UTF-8 d'abord  
                        try:  
                            with open(file_path, 'r', encoding='utf-8') as f:  
                                file_content = f.read()  
                        except UnicodeDecodeError:  
                            # Revenir à latin-1  
                            with open(file_path, 'r', encoding='latin-1') as f:  
                                file_content = f.read()

                        # Ajouter un en-tête d'information sur le fichier  
                        file_info = f"// Fichier : {filename} ({file_path.suffix})\n"  
                        if file_path.suffix.lower() in ['.py', '.r']:  
                            file_info = f"# Fichier : {filename} ({file_path.suffix})\n"  
                        elif file_path.suffix.lower() in ['.html', '.xml']:  
                            file_info = f"<!-- Fichier : {filename} ({file_path.suffix}) -->\n"  
                        elif file_path.suffix.lower() in ['.css', '.scss', '.sass']:  
                            file_info = f"/* Fichier : {filename} ({file_path.suffix}) */\n"  
                        elif file_path.suffix.lower() in ['.sql']:  
                            file_info = f"-- Fichier : {filename} ({file_path.suffix})\n"

                        full_content = file_info + file_content

                        self.logger.info(f"Fichier inclus : {filename} ({len(file_content)} caractères, {file_path.suffix})")  
                        return full_content

                    except Exception as e:  
                        self.logger.error(f"Erreur lors de la lecture du fichier {filename} : {e}")  
                        return f"[ERROR : Impossible de lire {filename} : {e}]"

            self.logger.warning(f"Fichier non trouvé : {filename}")  
            return f"[ERROR : Fichier {filename} non trouvé]"

        return re.sub(r'\{([^}]+)\}', replace_file, content)

    def _build_api_payload(self, new_message: str, override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  
        """Construire le payload de la requête API conforme à la structure GPT-4.1"""  
        # Traiter les inclusions de fichiers  
        processed_message = self._process_file_inclusions(new_message)

        # Construire les messages au format API  
        messages = []

        # Ajouter l'invite système si configurée  
        if self.config.system_prompt:  
            messages.append({  
                "role": "system",  
                "content": self.config.system_prompt  
            })

        # Ajouter l'historique de conversation  
        for msg in self.messages:  
            if msg["role"] in ["user", "assistant"]:  
                messages.append({  
                    "role": msg["role"],  
                    "content": msg["content"]  
                })

        # Ajouter le nouveau message de l'utilisateur  
        messages.append({  
            "role": "user",  
            "content": processed_message  
        })

        # Appliquer les éventuelles surcharges de configuration  
        config = asdict(self.config)  
        if override_config:  
            config.update(override_config)

        # Construire le payload  
        payload = {  
            "model": config["model"],  
            "messages": messages,  
            "temperature": config["temperature"],  
            "max_tokens": config["max_tokens"],  
            "top_p": config["top_p"],  
            "frequency_penalty": config["frequency_penalty"],  
            "presence_penalty": config["presence_penalty"],  
            "stream": config["stream"]  
        }

        return payload

    def _make_api_request(self, payload: Dict[str, Any]) -> requests.Response:  
        """Effectuer la requête API avec gestion des erreurs et des tentatives"""  
        headers = {  
            "Content-Type": "application/json",  
            "Authorization": f"Bearer {self.api_key}"  
        }

        timeout = 300  # Timeout fixe pour gpt-4.1

        max_retries = 3  
        base_delay = 1

        for attempt in range(max_retries):  
            try:  
                self.logger.info(f"Envoi de la requête API à OpenAI GPT-4.1 (tentative {attempt + 1}/{max_retries}) avec un timeout de {timeout}s...")

                response = requests.post(  
                    self.api_url,  
                    headers=headers,  
                    json=payload,  
                    stream=payload.get("stream", True),  
                    timeout=timeout  
                )

                if response.status_code == 200:  
                    self.logger.info("Requête API réussie")  
                    return response  
                elif response.status_code == 401:  
                    raise ValueError("Clé API invalide")  
                elif response.status_code == 403:  
                    raise ValueError("Accès API refusé")  
                elif response.status_code == 429:  
                    # Limite de taux atteinte - attendre et réessayer  
                    delay = base_delay * (2 ** attempt)  
                    self.logger.warning(f"Limite de taux atteinte, nouvelle tentative dans {delay}s...")  
                    time.sleep(delay)  
                    continue  
                elif response.status_code >= 500:  
                    # Erreur serveur - réessayer  
                    delay = base_delay * (2 ** attempt)  
                    self.logger.warning(f"Erreur serveur {response.status_code}, nouvelle tentative dans {delay}s...")  
                    time.sleep(delay)  
                    continue  
                else:  
                    response.raise_for_status()

            except Timeout as e:  
                self.logger.warning(f"Requête expirée après {timeout}s (tentative {attempt + 1}/{max_retries})")  
                if attempt == max_retries - 1:  
                    raise Exception(f"Requête expirée après {timeout}s.")  
                delay = base_delay * (2 ** attempt)  
                self.logger.warning(f"Nouvelle tentative dans {delay}s...")  
                time.sleep(delay)  
            except RequestException as e:  
                if attempt == max_retries - 1:  
                    raise  
                delay = base_delay * (2 ** attempt)  
                self.logger.warning(f"Requête échouée ({e}), nouvelle tentative dans {delay}s...")  
                time.sleep(delay)

        raise Exception(f"Échec de la requête API après {max_retries} tentatives")

    def _parse_streaming_response(self, response: requests.Response) -> Generator[str, None, None]:  
        """Analyser la réponse en streaming des événements côté serveur"""  
        assistant_message = ""

        try:  
            for line in response.iter_lines(decode_unicode=True):  
                if not line or line.strip() == "":  
                    continue

                try:  
                    # Gérer le format des événements côté serveur  
                    if line.startswith("data: "):  
                        data_str = line[6:].strip()

                        if data_str == "[DONE]":  
                            break

                        data = json.loads(data_str)

                        # Extraire le contenu  
                        choices = data.get("choices", [])  
                        if choices:  
                            choice = choices[0]  
                            delta = choice.get("delta", {})  
                            content = delta.get("content", "")

                            if content:  
                                assistant_message += content  
                                yield content

                            finish_reason = choice.get("finish_reason")  
                            if finish_reason == "stop":  
                                break

                except json.JSONDecodeError as e:  
                    self.logger.warning(f"JSON invalide dans le stream : {e}")  
                    continue  
                except Exception as e:  
                    self.logger.warning(f"Erreur lors du traitement de la ligne du stream : {e}")  
                    continue

        except Exception as e:  
            self.logger.error(f"Erreur lors de l'analyse de la réponse en streaming : {e}")

        # Ajouter le message de l'assistant à l'historique si du contenu a été reçu  
        if assistant_message.strip():  
            self.add_message("assistant", assistant_message)

    def _parse_non_streaming_response(self, response: requests.Response) -> str:  
        """Analyser la réponse non-streaming de l'API des complétions de chat OpenAI"""  
        try:  
            data = response.json()

            # Extraire le contenu du message selon le format standard de réponse d'OpenAI  
            choices = data.get("choices", [])  
            if choices:  
                message = choices[0].get("message", {})  
                content = message.get("content", "")

                if content:  
                    self.add_message("assistant", content)  
                    return content

            return "Aucun contenu de réponse reçu"

        except Exception as e:  
            self.logger.error(f"Erreur lors de l'analyse de la réponse non-streaming : {e}")  
            return f"Erreur lors de l'analyse de la réponse : {e}"

    def call_api(self, new_message: str, override_config: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:  
        """Appeler l'API OpenAI GPT-4.1 avec le nouveau message"""  
        try:  
            # Ajouter le message de l'utilisateur à l'historique  
            self.add_message("user", new_message)

            # Construire le payload de l'API  
            payload = self._build_api_payload(new_message, override_config)

            self.logger.info(f"Appel de l'API à {self.api_url}")  
            self.logger.debug(f"Payload : {json.dumps(payload, indent=2)}")

            # Afficher les informations du modèle à l'utilisateur  
            print(f"{Fore.YELLOW}🤖 Utilisation de GPT-4.1 (timeout : 5min 0s)...{Style.RESET_ALL}")

            # Effectuer la requête  
            response = self._make_api_request(payload)

            # Gérer le streaming vs non-streaming  
            if payload.get("stream", True):  
                yield from self._parse_streaming_response(response)  
            else:  
                result = self._parse_non_streaming_response(response)  
                yield result

        except Exception as e:  
            error_msg = f"Appel API échoué : {e}"  
            self.logger.error(error_msg)  
            yield error_msg

    def clear_history(self):  
        """Effacer l'historique de conversation"""  
        self._create_backup()  
        self.messages.clear()  
        self._save_history()  
        self.logger.info("Historique de conversation effacé")

    def get_statistics(self) -> Dict[str, Any]:  
        """Obtenir les statistiques de conversation"""  
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

            with open(filepath, 'w', encoding='utf-8') as f:  
                f.write(f"Exportation de Conversation - OpenAI GPT-4.1 Chat Agent\n")  
                f.write(f"Agent ID: {self.agent_id}\n")  
                f.write(f"Modèle: {self.config.model}\n")  
                f.write(f"Exporté le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  
                f.write("=" * 50 + "\n\n")

                for msg in self.messages:  
                    timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")  
                    f.write(f"[{timestamp}] {msg['role'].upper()}:\n")  
                    f.write(f"{msg['content']}\n\n")

        elif format_type == "md":  
            filename = f"conversation_{timestamp}.md"  
            filepath = export_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:  
                f.write(f"# Conversation OpenAI GPT-4.1 Chat Agent\n\n")  
                f.write(f"**Agent ID:** {self.agent_id}  \n")  
                f.write(f"**Modèle:** {self.config.model}  \n")  
                f.write(f"**Exporté le:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

                for msg in self.messages:  
                    timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")  
                    role_emoji = "🧑" if msg["role"] == "user" else "🤖"  
                    f.write(f"## {role_emoji} {msg['role'].title()} - {timestamp}\n\n")  
                    f.write(f"{msg['content']}\n\n")

        elif format_type == "html":  
            filename = f"conversation_{timestamp}.html"  
            filepath = export_dir / filename

            stats = self.get_statistics()

            # Template HTML avec style basique  
            html_template = f"""<!DOCTYPE html>  
<html lang="fr">  
<head>  
    <meta charset="UTF-8">  
    <title>Conversation OpenAI GPT-4.1 - {self.agent_id}</title>  
    <style>  
        body {{  
            font-family: Arial, sans-serif;  
            padding: 2rem;  
            background-color: #f0f0f0;  
        }}  
        .container {{  
            max-width: 800px;  
            margin: 0 auto;  
            background-color: #fff;  
            padding: 2rem;  
            border-radius: 8px;  
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);  
        }}  
        .header {{  
            text-align: center;  
            margin-bottom: 2rem;  
        }}  
        .message {{  
            margin-bottom: 1.5rem;  
        }}  
        .message.user {{  
            text-align: right;  
        }}  
        .message.assistant {{  
            text-align: left;  
        }}  
        .timestamp {{  
            color: #888;  
            font-size: 0.9rem;  
        }}  
        .content {{  
            display: inline-block;  
            padding: 0.5rem 1rem;  
            border-radius: 16px;  
            max-width: 80%;  
        }}  
        .user .content {{  
            background-color: #dcf8c6;  
        }}  
        .assistant .content {{  
            background-color: #f1f0f0;  
        }}  
    </style>  
</head>  
<body>  
    <div class="container">  
        <div class="header">  
            <h1>Conversation OpenAI GPT-4.1 Chat Agent</h1>  
            <p>Agent ID : {self.agent_id}</p>  
            <p>Exporté le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>  
        </div>  
"""

            # Ajouter les messages  
            for msg in self.messages:  
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")  
                role = msg["role"]  
                content = msg["content"].replace("\n", "<br>")  
                html_template += f"""        <div class="message {role}">  
            <div class="timestamp">{timestamp}</div>  
            <div class="content">{content}</div>  
        </div>  
"""

            # Clôturer le HTML  
            html_template += f"""        <div class="footer">  
            <p>Exporté par OpenAI GPT-4.1 Chat Agent</p>  
        </div>  
    </div>  
</body>  
</html>"""

            with open(filepath, 'w', encoding='utf-8') as f:  
                f.write(html_template)

        else:  
            raise ValueError(f"Format d'exportation non supporté : {format_type}")

        self.logger.info(f"Conversation exportée vers {filepath}")  
        return str(filepath)

    def search_history(self, term: str, limit: int = 10) -> List[Dict[str, Any]]:  
        """Rechercher dans l'historique de conversation un terme"""  
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
            Path('.'), Path('src'), Path('lib'),  
            Path('scripts'), Path('data'),  
            Path('documents'), Path('files'),  
            Path('config'), Path('configs'),  
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

            # Infos de base  
            agent_info = {  
                "id": agent_dir.name,  
                "path": str(agent_dir),  
                "exists": True  
            }

            # Charger les métadonnées si disponibles  
            if metadata_file.exists():  
                try:  
                    with open(metadata_file) as f:  
                        metadata = json.load(f)  
                        agent_info.update(metadata)  
                except:  
                    pass

            # Charger les infos de config  
            if config_file.exists():  
                try:  
                    with open(config_file) as f:  
                        config = yaml.safe_load(f)  
                        agent_info["model"] = config.get("model", "gpt-4.1")  
                        agent_info["created_at"] = config.get("created_at")  
                        agent_info["updated_at"] = config.get("updated_at")  
                except:  
                    pass

            # Infos sur l'historique  
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
    """Afficher les informations détaillées d'un agent"""  
    agent_dir = Path(f"agents/{agent_id}")

    if not agent_dir.exists():  
        print(f"{Fore.RED}Agent '{agent_id}' introuvable{Style.RESET_ALL}")  
        return

    print(f"\n{Fore.CYAN}{'='*50}")  
    print(f"Informations de l'Agent : {Fore.YELLOW}{agent_id}")  
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")

    # Charger et afficher la configuration  
    config_file = agent_dir / "config.yaml"  
    if config_file.exists():  
        try:  
            with open(config_file) as f:  
                config = yaml.safe_load(f)

            model = config.get('model', 'gpt-4.1')

            print(f"\n{Fore.GREEN}Configuration :")  
            print(f"{Fore.WHITE}  Modèle : {model}")  
            print(f"  Temperature : {config.get('temperature', 1.0)}")  
            print(f"  Tokens max : {config.get('max_tokens', 32768)}")  
            print(f"  Streaming : {config.get('stream', True)}")  
            print(f"  Créé le : {config.get('created_at', 'Inconnu')}")  
            print(f"  Mis à jour le : {config.get('updated_at', 'Inconnu')}")

        except Exception as e:  
            print(f"{Fore.RED}Erreur lors du chargement de la configuration : {e}")

    # Afficher les stats de l'historique  
    history_file = agent_dir / "history.json"  
    if history_file.exists():  
        try:  
            with open(history_file) as f:  
                history = json.load(f)

            user_msgs = len([m for m in history if m.get("role") == "user"])  
            assistant_msgs = len([m for m in history if m.get("role") == "assistant"])  
            total_chars = sum(len(m.get("content", "")) for m in history)

            print(f"\n{Fore.GREEN}Historique de Conversation :")  
            print(f"{Fore.WHITE}  Total de messages : {len(history)}")  
            print(f"  Messages utilisateur : {user_msgs}")  
            print(f"  Messages assistant : {assistant_msgs}")  
            print(f"  Caractères totaux : {total_chars:,}")  
            print(f"  Taille du fichier : {history_file.stat().st_size:,} bytes")

            if history:  
                first_msg = datetime.fromisoformat(history[0]["timestamp"])  
                last_msg = datetime.fromisoformat(history[-1]["timestamp"])  
                print(f"  Premier message : {first_msg.strftime('%Y-%m-%d %H:%M:%S')}")  
                print(f"  Dernier message : {last_msg.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:  
            print(f"{Fore.RED}Erreur lors du chargement de l'historique : {e}")  
    else:  
        print(f"\n{Fore.YELLOW}Aucun historique de conversation trouvé")

    # Afficher la structure des répertoires  
    print(f"\n{Fore.GREEN}Structure des Répertoires :")  
    for item in sorted(agent_dir.rglob("*")):  
        if item.is_file():  
            size = item.stat().st_size  
            size_str = f"{size:,}" if size < 1024 else f"{size/1024:.1f}K"  
            rel_path = item.relative_to(agent_dir)  
            print(f"{Fore.WHITE}  {rel_path} ({size_str} bytes)")

def create_agent_config_interactive() -> AgentConfig:  
    """Création interactive de la configuration"""  
    print(f"\n{Fore.CYAN}Création de la Configuration de l'Agent{Style.RESET_ALL}")  
    print(f"{Fore.YELLOW}Appuyez sur Entrée pour utiliser les valeurs par défaut{Style.RESET_ALL}\n")

    config = AgentConfig()

    # Afficher le modèle utilisé  
    print(f"{Fore.GREEN}Modèle utilisé : GPT-4.1 ({config.model}){Style.RESET_ALL}\n")

    # Température  
    temp_input = input(f"Temperature (0.0-2.0) [{config.temperature}]: ").strip()  
    if temp_input:  
        try:  
            config.temperature = float(temp_input)  
        except ValueError:  
            print(f"{Fore.RED}Température invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")

    # Invite système  
    system_prompt = input(f"Invite système (optionnel) : ").strip()  
    if system_prompt:  
        config.system_prompt = system_prompt

    # Tokens max de complétion  
    tokens_input = input(f"Tokens max de complétion [{config.max_tokens}]: ").strip()  
    if tokens_input:  
        try:  
            config.max_tokens = int(tokens_input)  
        except ValueError:  
            print(f"{Fore.RED}Nombre de tokens invalide, utilisation de la valeur par défaut{Style.RESET_ALL}")

    # Streaming  
    stream_input = input(f"Activer le streaming (y/n) [{'y' if config.stream else 'n'}] : ").strip().lower()  
    if stream_input in ['n', 'no', 'false']:  
        config.stream = False  
    elif stream_input in ['y', 'yes', 'true']:  
        config.stream = True

    return config

def interactive_chat(agent: OpenAIGPT41ChatAgent):  
    """Session de chat interactive"""  
    print(f"\n{Fore.GREEN}Démarrage du chat interactif avec OpenAI GPT-4.1")  
    print(f"Agent : {Fore.YELLOW}{agent.agent_id}")  
    print(f"{Fore.GREEN}Tapez '/help' pour les commandes, '/quit' pour quitter{Style.RESET_ALL}\n")

    while True:  
        try:  
            user_input = input(f"{Fore.CYAN}Vous : {Style.RESET_ALL}").strip()

            if not user_input:  
                continue

            # Gérer les commandes  
            if user_input.startswith('/'):  
                command_parts = user_input[1:].split()  
                command = command_parts[0].lower()

                if command == 'help':  
                    print(f"\n{Fore.YELLOW}Commandes Disponibles :")  
                    print(f"{Fore.WHITE}/help - Afficher ce message d'aide")  
                    print(f"/history [n] - Afficher les n derniers messages (par défaut 5)")  
                    print(f"/search <terme> - Rechercher dans l'historique de conversation")  
                    print(f"/stats - Afficher les statistiques de conversation")  
                    print(f"/config - Afficher la configuration actuelle")  
                    print(f"/export <json|txt|md|html> - Exporter la conversation")  
                    print(f"/clear - Effacer l'historique de conversation")  
                    print(f"/files - Lister les fichiers disponibles pour inclusion")  
                    print(f"/info - Afficher les informations de l'agent")  
                    print(f"/quit - Quitter le chat{Style.RESET_ALL}\n")  
                    print(f"{Fore.CYAN}Inclusion de Fichiers : Utilisez {{filename}} dans vos messages pour inclure le contenu des fichiers.")  
                    print(f"Supporté : Fichiers de programmation (.py, .r, .js, etc.), fichiers de config, documentation{Style.RESET_ALL}\n")

                elif command == 'history':  
                    limit = 5  
                    if len(command_parts) > 1:  
                        try:  
                            limit = int(command_parts[1])  
                        except ValueError:  
                            print(f"{Fore.RED}Nombre invalide{Style.RESET_ALL}")  
                            continue

                    recent_messages = agent.messages[-limit:]  
                    if not recent_messages:  
                        print(f"{Fore.YELLOW}Aucun message dans l'historique{Style.RESET_ALL}")  
                    else:  
                        print(f"\n{Fore.YELLOW}Les {len(recent_messages)} derniers messages :")  
                        for msg in recent_messages:  
                            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")  
                            role_color = Fore.CYAN if msg["role"] == "user" else Fore.GREEN  
                            print(f"{Fore.WHITE}[{timestamp}] {role_color}{msg['role']} : {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")  
                    print()

                elif command == 'search':  
                    if len(command_parts) < 2:  
                        print(f"{Fore.RED}Usage : /search <terme>{Style.RESET_ALL}")  
                        continue

                    search_term = ' '.join(command_parts[1:])  
                    results = agent.search_history(search_term)

                    if not results:  
                        print(f"{Fore.YELLOW}Aucun résultat pour '{search_term}'{Style.RESET_ALL}")  
                    else:  
                        print(f"\n{Fore.YELLOW}Trouvé {len(results)} résultats pour '{search_term}' :")  
                        for result in results:  
                            msg = result["message"]  
                            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")  
                            role_color = Fore.CYAN if msg["role"] == "user" else Fore.GREEN  
                            print(f"{Fore.WHITE}[{timestamp}] {role_color}{msg['role']} : {result['preview']}")  
                    print()

                elif command == 'stats':  
                    stats = agent.get_statistics()  
                    print(f"\n{Fore.YELLOW}Statistiques de la Conversation :")  
                    print(f"{Fore.WHITE}Modèle : {agent.config.model}")  
                    print(f"Total des messages : {stats['total_messages']}")  
                    print(f"Messages utilisateur : {stats['user_messages']}")  
                    print(f"Messages assistant : {stats['assistant_messages']}")  
                    print(f"Total de caractères : {stats['total_characters']:,}")  
                    print(f"Longueur moyenne des messages : {stats['average_message_length']:,}")  
                    if stats['first_message']:  
                        print(f"Premier message : {stats['first_message']}")  
                        print(f"Dernier message : {stats['last_message']}")  
                        print(f"Durée : {stats['conversation_duration']}")  
                    print()

                elif command == 'config':  
                    print(f"\n{Fore.YELLOW}Configuration Actuelle :")  
                    config_dict = asdict(agent.config)  
                    for key, value in config_dict.items():  
                        if key not in ['created_at', 'updated_at']:  
                            print(f"{Fore.WHITE}{key} : {value}")  
                    print()

                elif command == 'export':  
                    if len(command_parts) < 2:  
                        print(f"{Fore.RED}Usage : /export <json|txt|md|html>{Style.RESET_ALL}")  
                        continue

                    format_type = command_parts[1].lower()  
                    if format_type not in ['json', 'txt', 'md', 'html']:  
                        print(f"{Fore.RED}Format invalide. Utilisez : json, txt, md, ou html{Style.RESET_ALL}")  
                        continue

                    try:  
                        filepath = agent.export_conversation(format_type)  
                        print(f"{Fore.GREEN}Exporté vers : {filepath}{Style.RESET_ALL}")  
                    except Exception as e:  
                        print(f"{Fore.RED}Échec de l'exportation : {e}{Style.RESET_ALL}")

                elif command == 'clear':  
                    confirm = input(f"{Fore.YELLOW}Effacer l'historique de conversation ? (y/N) : {Style.RESET_ALL}").strip().lower()  
                    if confirm in ['y', 'yes']:  
                        agent.clear_history()  
                        print(f"{Fore.GREEN}Historique de conversation effacé{Style.RESET_ALL}")

                elif command == 'files':  
                    files = agent.list_files()  
                    if not files:  
                        print(f"{Fore.YELLOW}Aucun fichier supporté trouvé pour inclusion{Style.RESET_ALL}")  
                    else:  
                        print(f"\n{Fore.YELLOW}Fichiers disponibles pour inclusion :")  
                        for file_info in files[:20]:  # Limiter à 20 fichiers  
                            print(f"{Fore.WHITE}{file_info}")  
                        if len(files) > 20:  
                            print(f"{Fore.YELLOW}... et {len(files) - 20} autres fichiers")  
                        print(f"{Fore.CYAN}Utilisez {{filename}} dans votre message pour inclure le contenu du fichier{Style.RESET_ALL}\n")

                elif command == 'info':  
                    show_agent_info(agent.agent_id)

                elif command in ['quit', 'exit', 'q']:  
                    print(f"{Fore.GREEN}Au revoir !{Style.RESET_ALL}")  
                    break

                else:  
                    print(f"{Fore.RED}Commande inconnue : {command}{Style.RESET_ALL}")  
                    print(f"{Fore.YELLOW}Tapez '/help' pour les commandes disponibles{Style.RESET_ALL}")

                continue

            # Message régulier - envoyer à l'API  
            print(f"\n{Fore.GREEN}Assistant : {Style.RESET_ALL}", end="", flush=True)

            response_text = ""  
            for chunk in agent.call_api(user_input):  
                print(chunk, end="", flush=True)  
                response_text += chunk

            print("\n")

        except KeyboardInterrupt:  
            print(f"\n{Fore.YELLOW}Utilisez '/quit' pour quitter en douceur{Style.RESET_ALL}")  
        except Exception as e:  
            print(f"\n{Fore.RED}Erreur : {e}{Style.RESET_ALL}")

def main():  
    """Interface principale de la CLI"""  
    parser = argparse.ArgumentParser(  
        description="OpenAI GPT-4.1 Chat Agent - Interface Avancée de Chat IA",  
        formatter_class=argparse.RawDescriptionHelpFormatter,  
        epilog="""  
Exemples :  
  %(prog)s --agent-id mon-agent                    # Démarrer un chat interactif  
  %(prog)s --list                                 # Lister tous les agents  
  %(prog)s --agent-id mon-agent --export html      # Exporter la conversation en HTML  
  %(prog)s --agent-id mon-agent --config          # Configurer l'agent de manière interactive  
        """  
    )

    parser.add_argument("--agent-id", help="ID de l'agent pour la session de chat")  
    parser.add_argument("--list", action="store_true", help="Lister tous les agents disponibles")  
    parser.add_argument("--info", metavar="ID", help="Afficher les informations détaillées pour un agent")  
    parser.add_argument("--config", action="store_true", help="Configurer l'agent de manière interactive")  
    parser.add_argument("--temperature", type=float, help="Surcharger la température (0.0-2.0)")  
    parser.add_argument("--no-stream", action="store_true", help="Désactiver le streaming")  
    parser.add_argument("--export", choices=["json", "txt", "md", "html"], help="Format d'exportation de la conversation")

    args = parser.parse_args()

    # Gérer la commande list  
    if args.list:  
        agents = list_agents()  
        if not agents:  
            print(f"{Fore.YELLOW}Aucun agent trouvé{Style.RESET_ALL}")  
            return

        print(f"\n{Fore.CYAN}Agents Disponibles :{Style.RESET_ALL}")  
        print(f"{Fore.WHITE}{'ID':<20} {'Modèle':<15} {'Messages':<10} {'Dernière Mise à Jour':<20}")  
        print("-" * 75)

        for agent in agents:  
            updated = agent.get("updated_at", "Inconnu")  
            if updated != "Inconnu":  
                try:  
                    updated = datetime.fromisoformat(updated).strftime("%Y-%m-%d %H:%M")  
                except:  
                    pass

            model = agent.get('model', 'gpt-4.1')  
            print(f"{agent['id']:<20} {model:<15} {agent.get('message_count', 0):<10} {updated:<20}")

        return

    # Gérer la commande info  
    if args.info:  
        show_agent_info(args.info)  
        return

    # Requérir l'ID de l'agent pour les autres opérations  
    if not args.agent_id:  
        parser.print_help()  
        print(f"\n{Fore.RED}Erreur : --agent-id est requis{Style.RESET_ALL}")  
        return

    try:  
        # Initialiser l'agent  
        agent = OpenAIGPT41ChatAgent(args.agent_id)

        # Gérer la commande config  
        if args.config:  
            new_config = create_agent_config_interactive()  
            agent.config = new_config  
            agent._save_config()  
            print(f"{Fore.GREEN}Configuration sauvegardée{Style.RESET_ALL}")  
            return

        # Gérer la commande export  
        if args.export:  
            filepath = agent.export_conversation(args.export)  
            print(f"{Fore.GREEN}Exporté vers : {filepath}{Style.RESET_ALL}")  
            return

        # Appliquer les surcharges de la ligne de commande  
        overrides = {}  
        if args.temperature is not None:  
            overrides["temperature"] = args.temperature  
        if args.no_stream:  
            overrides["stream"] = False

        if overrides:  
            agent.config = AgentConfig(**{**asdict(agent.config), **overrides})  
            agent._save_config()

        # Démarrer le chat interactif  
        interactive_chat(agent)

    except KeyboardInterrupt:  
        print(f"\n{Fore.YELLOW}Interrompu par l'utilisateur{Style.RESET_ALL}")  
    except Exception as e:  
        print(f"{Fore.RED}Erreur : {e}{Style.RESET_ALL}")  
        sys.exit(1)

if __name__ == "__main__":  
    main()
