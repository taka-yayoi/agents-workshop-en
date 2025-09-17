from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

import os
from typing import List, Dict, Any
import json

############################################
# Get settings from environment variables
############################################
def get_env_config():
    """Get the minimum required settings from environment variables"""
    
    # Get required environment variables
    llm_endpoint = os.environ.get("LLM_ENDPOINT_NAME")
    uc_tool_names = os.environ.get("UC_TOOL_NAMES", "")
    vs_name = os.environ.get("VS_NAME", "")
    
    # Validate required fields
    if not llm_endpoint:
        raise ValueError("LLM_ENDPOINT_NAME environment variable is required")
    
    if not uc_tool_names:
        raise ValueError("UC_TOOL_NAMES environment variable is required")

    if not vs_name:
        raise ValueError("VS_NAME environment variable is required")

    # Split tool names (remove empty strings and spaces)
    tool_names = [name.strip() for name in uc_tool_names.split(",") if name.strip()]

    config = {
        "llm_endpoint": llm_endpoint,
        "uc_tool_names": tool_names,
        "vs_name": vs_name
    }
    
    return config

# Get settings
config = get_env_config()

LLM_ENDPOINT_NAME = config['llm_endpoint']
UC_TOOL_NAMES = config['uc_tool_names']
VS_NAME = config['vs_name']

# Output for checking settings
print("Agent settings:")
print(f"LLM_ENDPOINT_NAME: {LLM_ENDPOINT_NAME}")
print(f"UC_TOOL_NAMES: {UC_TOOL_NAMES}")
print(f"VS_NAME: {VS_NAME}")

# Enable LangChain/MLflow autologging
mlflow.langchain.autolog()

# Initialize Databricks Function Client and set as UC function client
client = DatabricksFunctionClient(disable_notice=True, suppress_warnings=True)
set_uc_function_client(client)

############################################
# Create LLM instance
############################################
llm = ChatDatabricks(
    endpoint=config["llm_endpoint"]
)

# System prompt (controls agent behavior)
system_prompt = "You are a customer success specialist at Databricks Lab. For user questions about products, use tools to obtain necessary information and support users so they can fully understand the products. Always strive to provide value in every interaction by including as much information as possible that may interest the customer."

#system_prompt = "You are a customer success specialist at Databricks Lab. For user questions about products, use tools to obtain necessary information and answer only the question concisely, without adding fictional features, colors, or general comments. Marketing expressions and unnecessary background explanations are not needed."


###############################################################################
## Define tools for the agent. This enables data retrieval and actions beyond text generation.
## For more examples of creating and using tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################

############################################
# Create tools
############################################
def create_tools() -> List[BaseTool]:
    """Create tools based on environment variable settings"""
    tools = []
    
    # Add Vector Search tool
    if VS_NAME:
        try:            
            vs_tool = VectorSearchRetrieverTool(
                index_name=VS_NAME,
                tool_name="search_product_docs",
                num_results=3, # Get 3 documents from VS
                #num_results=1, # Get 1 document from VS
                tool_description="Use this tool to search product documents.",
                disable_notice=True
            )
            tools.append(vs_tool)
            print(f"Added Vector Search tool: {VS_NAME}")
        except Exception as e:
            print(f"Warning: Could not load Vector Search tool {VS_NAME}: {e}")
    
    # Add UC function tools
    if UC_TOOL_NAMES:
        try:
            uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
            tools.extend(uc_toolkit.tools)
            print(f"Added UC function tools: {UC_TOOL_NAMES}")
        except Exception as e:
            print(f"Warning: Could not add UC tools {UC_TOOL_NAMES}: {e}")
    
    return tools

# Create tools
# Also used for MLflow logging
tools = create_tools()

#####################
## Define agent logic
#####################
def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    # Bind tools to model
    model = model.bind_tools(tools)

    # Define function to decide which node to proceed to next
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # Continue if there is a function call, otherwise end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    # Preprocessing to add system prompt at the beginning
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    # Function to call the model
    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    # Custom tool execution function
    def execute_tools(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # Get tool_calls
        tool_calls = last_message.get("tool_calls", [])
        if not tool_calls:
            return {"messages": []}
        
        # Execute tools
        tool_outputs = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name") if isinstance(tool_call, dict) else tool_call.function.name
            tool_args = tool_call.get("function", {}).get("arguments") if isinstance(tool_call, dict) else tool_call.function.arguments
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id
            
            # Find and execute tool
            tool_result = None
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        # Parse arguments
                        import json
                        if isinstance(tool_args, str):
                            args = json.loads(tool_args)
                        else:
                            args = tool_args
                        
                        # Execute tool
                        result = tool.invoke(args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    break
            
            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"
            
            # Create message for tool execution result
            tool_message = {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id,
                "name": tool_name
            }
            tool_outputs.append(tool_message)
        
        return {"messages": tool_outputs}

    # Build LangGraph workflow
    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# LangGraphChatAgent class (wrapper for MLflow inference)
class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """Convert ChatAgentMessage to dict (revised version)"""
        converted = []
        
        # Handle case where messages is None or empty
        if not messages:
            return converted
            
        for msg in messages:
            try:
                if msg is None:
                    print("Warning: None message encountered")
                    continue
                    
                # Convert ChatAgentMessage object to dict
                if hasattr(msg, 'dict'):
                    msg_dict = msg.dict()
                elif isinstance(msg, dict):
                    msg_dict = msg
                else:
                    print(f"Warning: Unexpected message type: {type(msg)}")
                    continue
                
                # For tool role messages, handle empty content
                if msg_dict.get("role") == "tool":
                    # Set default value if content is empty or None
                    if not msg_dict.get("content"):
                        msg_dict["content"] = "Tool execution completed"
                    
                    # Set tool_call_id if needed
                    if "tool_call_id" not in msg_dict and msg_dict.get("id"):
                        msg_dict["tool_call_id"] = msg_dict["id"]
                
                converted.append(msg_dict)
            except Exception as e:
                print(f"Error converting message: {e}, Message: {msg}")
                continue
        
        return converted

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # Convert input messages to dict
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        # Collect messages from LangGraph stream (close to original code)
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue
                                    
                                # Convert message object to dict
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    print(f"Warning: Unexpected message type: {type(msg)}")
                                    continue
                                
                                # Check content of tool message
                                if msg_dict.get("role") == "tool":
                                    # Set default value if content is missing
                                    if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                        msg_dict["content"] = "Tool executed successfully"
                                    
                                    # Set tool_call_id if needed
                                    if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                        msg_dict["tool_call_id"] = msg_dict["id"]
                                
                                try:
                                    messages.append(ChatAgentMessage(**msg_dict))
                                except Exception as e:
                                    print(f"Warning: Failed to create ChatAgentMessage: {e}")
                                    print(f"Message data: {msg_dict}")
                                    
        except Exception as e:
            print(f"Error in predict method: {e}")
            import traceback
            traceback.print_exc()
            
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        # Convert input messages to dict
        request = {"messages": self._convert_messages_to_dict(messages)}
        
        # Generate responses sequentially in stream
        try:
            for event in self.agent.stream(request, stream_mode="updates"):
                if event and isinstance(event, dict):
                    for node_data in event.values():
                        if node_data and isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data.get("messages", []):
                                if msg is None:
                                    continue
                                    
                                # Convert message object to dict
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    print(f"Warning: Unexpected message type in stream: {type(msg)}")
                                    continue
                                
                                # Check content of tool message
                                if msg_dict.get("role") == "tool":
                                    # Set default value if content is missing
                                    if not msg_dict.get("content") and not msg_dict.get("tool_calls"):
                                        msg_dict["content"] = "Tool executed successfully"
                                    
                                    # Set tool_call_id if needed
                                    if "tool_call_id" not in msg_dict and "id" in msg_dict:
                                        msg_dict["tool_call_id"] = msg_dict["id"]
                                
                                try:
                                    yield ChatAgentChunk(**{"delta": msg_dict})
                                except Exception as e:
                                    print(f"Warning: Failed to create ChatAgentChunk: {e}")
                                    continue
                                    
        except Exception as e:
            print(f"Error in predict_stream method: {e}")
            import traceback
            traceback.print_exc()
            return

# Create agent object and specify as the agent to use for inference with mlflow.models.set_model()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
