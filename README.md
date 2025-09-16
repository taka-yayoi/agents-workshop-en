### Hands-on Lab: Build an Agent System with Databricks

This lab is divided into two parts. **Part 1** focuses on building and testing a Databricks agent with various tool calls for a customer service scenario. **Part 2** involves creating a simpler agent to answer product-related questions and focuses on evaluating its performance.

---

**Please run the following notebook on serverless compute.**

### Part 1: Design Your First Agent
##### Notebook: 01_create_tools\01_create_tools

#### 1.1 Building Tools
- **SQL Functions**  
  - Create queries to access important data for handling customer service return workflows.  
  - These SQL functions can be easily called from notebooks or agents.
- **Simple Python Functions**  
  - Create Python functions to address common limitations of language models.  
  - Register these as "tools" so the agent can call them as needed.

#### 1.2 Integration with LLM [AI Playground]
- **Combining Tools and LLM**  
  - Combine SQL/Python tools with a large language model (LLM) in Databricks AI Playground.  
  - Model: Claude 3.7 Sonnet

#### 1.3 Testing the Agent [AI Playground]
- **System Prompt**: Call tools until all company policies are satisfied.
- **Sample Question**: According to our policy, should we accept the latest return in the queue?
  - Observe the agent's step-by-step reasoning and final answer.
- **Check MLflow Traces**  
  - Review the agent's execution in MLflow to understand how each tool is called.  
  
---

### Part 2: Agent Evaluation
##### Notebook: 02_agent_eval\agent

#### 2.1 Defining a New Agent and Retriever Tool
- **Vector Search**  
  - A vector search index is pre-built to retrieve relevant product documents.  
  - This VS index is at agents_lab.product.product_docs_index.
- **Creating a Retriever Function**  
  - Wrap this vector search index in a function so the LLM can search for product information.  
  - Use the same LLM for the final answer.

##### Notebook: 02_agent_eval/driver

#### 2.2 Defining the Evaluation Dataset
- **Using the Provided Dataset**  
  - Use the sample evaluation dataset to test the agent's ability to answer product questions.  
  - (Optional) Try [synthetic data generation](https://www.databricks.com/jp/blog/streamline-ai-agent-evaluation-with-new-synthetic-data-capabilities).

#### 2.3 Evaluating the Agent
- **Run `MLflow.evaluate()`** 
  - MLflow compares the agent's answers with the ground truth dataset.  
  - An LLM-based judge scores each answer and collects feedback.

#### 2.4 Improvement and Re-evaluation
- **Improve the Retriever**  
  - Adjust retriever settings (k=5→k=1) based on evaluation feedback.  
- **Re-run Evaluation**  
  - Start a new MLflow run
  - Run `MLflow.evaluate()` again and compare results.  
  - Check for performance improvements in the MLflow evaluation UI

---

### Next Steps
- **Leverage More Tools**: Extend your agent with APIs, advanced Python functions, and additional SQL endpoints.  
- **Production Deployment**: Implement continuous integration/delivery (CI/CD), monitor performance with MLflow, and manage model versions.

---

Congratulations on building and evaluating an agent system with Databricks!