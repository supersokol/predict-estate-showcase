from typing import Dict, List, Optional
import json
import os
from datetime import datetime
from openai import OpenAI
from src.core.logger import logger
from src.registry import data_source_registry
import asyncio

class AssistantManager:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.assistant_id = None
        self.thread_id = None
        self.current_run = None
        
    def create_assistant(self, name: str, instructions: str, tools: List[Dict] = None) -> str:
        """Create a new assistant with specified configuration."""
        try:
            if not tools:
                tools = [
                    {"type": "code_interpreter"},
                    {"type": "retrieval"}
                ]
            
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model="gpt-4-turbo-preview"
            )
            self.assistant_id = assistant.id
            logger.info(f"Created assistant: {assistant.id}")
            return assistant.id
        except Exception as e:
            logger.error(f"Error creating assistant: {e}")
            raise

    def create_thread(self) -> str:
        """Create a new conversation thread."""
        try:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
            logger.info(f"Created thread: {thread.id}")
            return thread.id
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            raise

    async def add_context_files(self, context_sources: List[Dict]) -> List[str]:
        """Add context files to the thread."""
        file_ids = []
        
        try:
            for source in context_sources:
                if source["type"] == "csv":
                    # Get file from DataSourceRegistry and save temporarily
                    df = data_source_registry.get_table(source["file"])
                    temp_path = f"data/temp/{source['file']}"
                    df.to_csv(temp_path, index=False)
                    
                    with open(temp_path, "rb") as file:
                        response = self.client.files.create(
                            file=file,
                            purpose="assistants"
                        )
                        file_ids.append(response.id)
                    
                    os.remove(temp_path)
                
                elif source["type"] == "pdf":
                    file_path = data_source_registry.get_file_path(source["file"])
                    with open(file_path, "rb") as file:
                        response = self.client.files.create(
                            file=file,
                            purpose="assistants"
                        )
                        file_ids.append(response.id)
                
                elif source["type"] in ["wikipedia", "text"]:
                    # Save text content as file
                    content = source.get("text", "") or get_context("wiki", term=source["query"])
                    temp_path = f"data/temp/content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    
                    with open(temp_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    with open(temp_path, "rb") as file:
                        response = self.client.files.create(
                            file=file,
                            purpose="assistants"
                        )
                        file_ids.append(response.id)
                    
                    os.remove(temp_path)
            
            # Attach files to assistant
            self.client.beta.assistants.update(
                assistant_id=self.assistant_id,
                file_ids=file_ids
            )
            
            logger.info(f"Added {len(file_ids)} files to assistant")
            return file_ids
            
        except Exception as e:
            logger.error(f"Error adding context files: {e}")
            raise

    async def process_message(self, message: str) -> Dict:
        """Process a message in the thread."""
        try:
            # Add message to thread
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=message
            )

            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            )
            self.current_run = run

            # Wait for completion
            run = await self._wait_for_run(run.id)
            
            # Get messages
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread_id
            )
            
            # Get last assistant message
            last_message = next(
                (msg for msg in messages if msg.role == "assistant"),
                None
            )

            if last_message:
                return {
                    "response": last_message.content[0].text.value,
                    "run_id": run.id,
                    "status": run.status
                }
            else:
                raise Exception("No assistant response found")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise

    async def _wait_for_run(self, run_id: str) -> Dict:
        """Wait for a run to complete and handle any required actions."""
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                return run
            elif run.status == "requires_action":
                # Handle tool calls if needed
                await self._handle_tool_calls(run)
            elif run.status in ["failed", "expired"]:
                raise Exception(f"Run failed with status: {run.status}")
            
            await asyncio.sleep(1)

    async def _handle_tool_calls(self, run: Dict):
        """Handle required tool calls."""
        tool_outputs = []
        
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            # Handle different tool types here
            # For now, we'll just log the calls
            logger.info(f"Tool call requested: {tool_call.type} - {tool_call.function.name}")
            
        if tool_outputs:
            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

    def save_conversation(self, file_path: Optional[str] = None) -> str:
        """Save the current conversation history."""
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread_id
            )
            
            conversation = {
                "thread_id": self.thread_id,
                "assistant_id": self.assistant_id,
                "timestamp": datetime.now().isoformat(),
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content[0].text.value,
                        "created_at": msg.created_at
                    }
                    for msg in messages
                ]
            }
            
            if not file_path:
                file_path = f"data/conversations/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation, f, indent=2)
            
            logger.info(f"Saved conversation to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            raise

    def load_conversation(self, file_path: str) -> Dict:
        """Load a saved conversation."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                conversation = json.load(f)
            
            self.thread_id = conversation["thread_id"]
            self.assistant_id = conversation["assistant_id"]
            
            logger.info(f"Loaded conversation from {file_path}")
            return conversation
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            raise