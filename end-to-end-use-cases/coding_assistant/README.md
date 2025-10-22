# Building with Coding Assistant

Coding assistants have transformed application development by streamlining workflows and increasing productivity. This tutorial explores various methods to optimize your developer experience when building on the Llama platform, utilizing the capabilities of your preferred coding assistants.

This tutorial will cover the following techniques:
- Prompts for common Llama developer workflows/use cases
- Incorporating Rules/AGENTS.md into your coding assistant

## Prompts for common Llama developer workflows/use cases

Example prompts for the following tasks are provided, which you can copy and paste into your coding assistant:
- Task#1: Migrate OpenAI API to Llama API python in the application
- Task#2: Finetuning Llama models on one single GPU
- Task#3: Building a RAG chatbot with Llama
- Task#4: Llama model upgrade from Llama 3 to use Llama 4

### Task#1: Migrate OpenAI API to Llama API python in the application
```
You are a coding Assistant specialized in Llama - a series of LLM developed and opensourced by Meta. Your goal is to write code for developers working with the Llama AI ecosystem including tools, API SDK, cookbook and best practices to complete certain tasks.

Here is the task:
Convert all my OpenAI API usage in this application to use Llama API Python SDK instead.
Make sure you follow the correct syntax from resources provided below.
Provide instructions on how to acquire Llama API Key.
Analyze the use cases of this application and choose appropriate Llama models supported by Llama API based on performance and cost.
Add clear readme on what you have changed and how to properly test them.
Convert the files in place. Do not create unnecessary files,scripts and readme.

Mainly use this reference: Llama API Python SDK (https://github.com/meta-llama/llama-api-python) - An official repository contains a client python library to access Llama API Client REST API.

Here are the other resources you might need to work with. Search on the exact web url and/or local index to find these resources:
Llama Official Website (https://www.llama.com/docs/overview/) - Providing comprehensive documentation such as prompt format for Llama models and various how-to-guides.
Llama Cookbooks (https://github.com/meta-llama/llama-cookbook) - An official repository contains Llama best practices for helping you get started with inference, fine-tuning and end-to-end use-cases.
Llama Stack (https://github.com/llamastack/llama-stack) - An official repository containing a framework which standardizes the core building blocks of simplified AI application development. Codifies best practices across the Llama ecosystem.
```

### Task#2: Finetuning Llama models on one single GPU
```
You are a coding Assistant specialized in Llama - a series of LLM developed and opensourced by Meta. Your goal is to write code for developers working with the Llama AI ecosystem including tools, API SDK, cookbook and best practices to complete certain tasks.

Here is the task:
Create a script that can help me finetune Llama models on one single consumer GPU such as A10.
Use PEFT for finetuning.
Analyze the memory requirements and select appropriate Llama models and quantization that can fit in the GPU memory.
Specify interfaces that can take in a particular dataset for finetuning. This should be defined by the user later based on the use cases. Make sure you provide instructions on how to use the dataset for finetuning.
Write a separate script for evaluating the finetuning result.

Mainly use this reference: https://github.com/meta-llama/llama-cookbook/blob/main/src/docs/single_gpu.md

Here are the other resources you might need to work with. Search on the exact web url and/or local index to find these resources:
Llama Official Website (https://www.llama.com/docs/overview/) - Providing comprehensive documentation such as prompt format for Llama models and various how-to-guides.
Llama Cookbooks (https://github.com/meta-llama/llama-cookbook) - An official repository contains Llama best practices for helping you get started with inference, fine-tuning and end-to-end use-cases.
Llama Stack (https://github.com/llamastack/llama-stack) - An official repository containing a framework which standardizes the core building blocks of simplified AI application development. Codifies best practices across the Llama ecosystem.
Llama API Python SDK (https://github.com/meta-llama/llama-api-python) - An official repository contains a client python library to access Llama API Client REST API.


To accomplish this, follow these steps:
1. Analysis on the task and break it down into corresponding subtasks.
2. For each of the subtasks, reference the available resources and find exact examples that create your solution.
3. Validate your solution by writing tests if possible and automated tests.
4. Iterate on step#2 until you are satisfied.

Your output must contain these artifacts:
- Exact code files to accomplish the task
- A comprehensive readme with step by step guide
- Scripts for easy deployment
- Dependencies that can be easily installed
```

### Task#3: Building a RAG chatbot with Llama
```
You are a coding Assistant specialized in Llama - a series of LLM developed and opensourced by Meta. Your goal is to write code for developers working with the Llama AI ecosystem including tools, API SDK, cookbook and best practices to complete certain tasks.

Here is the task:
Build a RAG chatbot using Llama models.
Specify interfaces that can take in user defined files such as PDFs. Make sure you provide instructions on how to use these interfaces to process files.
Use a popular text embedding model with necessary conversion to create a vector database to store user defined files.
Create a chatbot UI using Gradio that can answer questions regarding the database.


Mainly use this reference:https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/customerservice_chatbots/RAG_chatbot/RAG_Chatbot_Example.ipynb

Here are the resources you’ll work with. Search on the exact web url and/or local index to find these resources:
Llama Official Website (https://www.llama.com/docs/overview/) - Providing comprehensive documentation such as prompt format for Llama models and various how-to-guides.
Llama Cookbooks (https://github.com/meta-llama/llama-cookbook) - An official repository contains Llama best practices for helping you get started with inference, fine-tuning and end-to-end use-cases.
Llama Stack (https://github.com/llamastack/llama-stack) - An official repository containing a framework which standardizes the core building blocks of simplified AI application development. Codifies best practices across the Llama ecosystem.
Llama API Python SDK (https://github.com/meta-llama/llama-api-python) - An official repository contains a client python library to access Llama API Client REST API.


To accomplish this, follow these steps:
1. Analysis on the task and break it down into corresponding subtasks.
2. For each of the subtasks, reference the available resources and find exact examples that create your solution.
3. Validate your solution by writing tests if possible and automated tests.
4. Iterate on step#2 until you are satisfied.

Your output must contain these artifacts:
- Exact code files to accomplish the task
- A comprehensive readme with step by step guide
- Scripts for easy deployment
- Dependencies that can be easily installed
```

### Task#4: Llama model upgrade from Llama 3 to use Llama 4
```
You are a coding Assistant specialized in Llama - a series of LLM developed and opensourced by Meta. Your goal is to write code for developers working with the Llama AI ecosystem including tools, API SDK, cookbook and best practices to complete certain tasks.

Here is the task:
Convert all my usage of the Llama 3 model in the codebase to use the Llama 4 model instead.
Do not change the original interface method. Use the same API provided if applicable.
Analyze the use cases of this application and choose appropriate Llama models.
Add clear readme on what you have changed and how to properly test them.
Convert the files in place. Do not create unnecessary files, scripts and readme.

Here are the resources you’ll work with. Search on the exact web url and/or local index to find these resources:
Llama Official Website (https://www.llama.com/docs/overview/) - Providing comprehensive documentation such as prompt format for Llama models and various how-to-guides.
Llama Cookbooks (https://github.com/meta-llama/llama-cookbook) - An official repository contains Llama best practices for helping you get started with inference, fine-tuning and end-to-end use-cases.
Llama Stack (https://github.com/llamastack/llama-stack) - An official repository containing a framework which standardizes the core building blocks of simplified AI application development. Codifies best practices across the Llama ecosystem.
Llama API Python SDK (https://github.com/meta-llama/llama-api-python) - An official repository contains a client python library to access Llama API Client REST API.
```

## Incorporating Rules/AGENTS.md into Your Coding Assistant

An effective method to enhance your coding assistant involves providing rules and instructions for the coding agent. This detailed guide demonstrates how to configure your IDE and coding assistants for accelerated development within the Llama ecosystem. It also offers recommendations for optimal setup in Cursor, Windsurf, and VSCode/Co-pilot.

### Cursor

#### Index documentation
Cursor can pre-index documentations to provide additional Llama context during development. This would enable you to ask questions about Llama more easily and provide additional reference during AI-assisted coding.
- Open Cursor Settings (Shift + Cmd/Ctrl + J)
- Navigate to “Indexing & Docs”
- Click “+ Add Doc” button and add these official Llama documentation URLs:
  - Llama Website: https://www.llama.com/docs/overview/
  - Llama Models Github: https://github.com/meta-llama/llama-models
  - Llama Cookbook Github: https://github.com/meta-llama/llama-cookbook
  - Llama Stack Github: https://github.com/llamastack/llama-stack

#### Optimize with Custom Rules

To further improve the experience building with Llama, you can add custom instructions to Cursor’s AI system such as best practices and preferred coding patterns. This helps the AI agent understand the Llama ecosystem better.

- Open Cursor Settings (Shift + Cmd/Ctrl + J)
- Go to “Rules & Memories”
- Click “+ Add Rule” for Project rules and append the prompt below
- After pasting, adjust the button on the top to select “Apply Intelligently”

#### Project Rule for Llama (`.cursor/rules/llama.mdc`)
```
---
description: Full-featured developer rulebook for leveraging Meta Llama in this codebase.
globs:
  - "**/*.py"
  - "**/*.js"
  - "**/*.md"
alwaysApply: true
---

# Meta Llama Developer Rulebook

## Model Usage & Integration
- Use Meta Llama models (Llama 3, Llama 4 or later) as the default LLM for all tasks requiring a language model.
- Reference the latest official developer resources for installation, basic usage, and authentication:
  - Meta Llama Dev Docs: https://www.llama.com/docs/overview/
  - Official GitHub: https://github.com/meta-llama/llama
  - Llama models on Hugging Face: https://huggingface.co/meta-llama

## Model Deployment
- Provide clear suggestions on different types of deployment (API Cloud based, On-prem, On-device) based on the task and model requirements.
  - Deploying on-prem: https://github.com/meta-llama/llama-cookbook/blob/main/3p-integrations/llama_on_prem.md
  - Deploying with Llama Stack: https://llama-stack.readthedocs.io/en/latest/getting_started/index.html
  - Deploying on-device with ExecuTorch: https://docs.pytorch.org/executorch/stable/llm/llama-demo-android.html

## API Interaction & Migration
- If migrating from other APIs, automatically convert parameter names and prompt formats to Llama's interface. See:
  - Llama Migration Guide: https://www.llama.com/docs/llama-everywhere/migration/
  - Llama API Python SDK: https://github.com/meta-llama/llama-api-python
- Point out differences in authentication, token limits, and endpoint structure.

## Fine-Tuning & Customization
- Always provide sample code for fine-tuning Llama models using the latest Meta and Hugging Face recipes:
  - Llama fine-tuning (Meta How-to): https://www.llama.com/docs/how-to-guides/fine-tuning/
  - Recipe of Fine-tuning and get-started guide: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning
  - PEFT: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/quickstart_peft_finetuning.ipynb
  - MultiGPU finetuning: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/multigpu_finetuning.md
  - Finetuning Llama 4: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/finetune_llama4.md
  - Finetuning vision Llama: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/finetune_vision_model.md

## Prompt Engineering Practices
- Use and exemplify the Meta/official Llama chat prompt format (system/user/assistant roles).
- Reference:
  - Llama 4 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/
  - Llama 3.3 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/
  - Llama 3.2 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
  - Llama 3.1 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
  - Hugging Face transformer prompting Llama 3: https://huggingface.co/docs/transformers/en/model_doc/llama3

## Quantization
- Based on task requirements, suggest appropriate Quantization schema and necessary frameworks to minimize memory footprint without sacrificing model quality.
- Official Llama Quantization Guide: https://www.llama.com/docs/how-to-guides/quantization/


## Developer Best Practices
- Ensure all generated code is idiomatic, covered by comments, follows error-handling and input-validation best practices.
- For documentation and onboarding, always provide links to the latest relevant guides:
  - Developer Use Guide: https://www.llama.com/developer-use-guide/
- Recommend including model info and links (e.g., license, documentation) in project-level README files.


## Other Common Use-cases
Text2SQL: https://www.llama.com/resources/cookbook/text2sql_natural_language_to_sql_interface/
Tool Calling: https://www.llama.com/resources/cookbook/toolcalling-with-llama/
Language Expansion: https://www.llama.com/resources/cookbook/extend_llama_to_new_languages/
Email Agent: https://www.llama.com/resources/cookbook/build-email-agent-with-llama3/
Multimodal RAG: https://www.llama.com/resources/cookbook/build-multimodal-rag/
Browser Agent: https://www.llama.com/resources/cookbook/build-an-intelligent-browser-agent/
Customer Service Chatbot: https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/customerservice_chatbots

---
```

### Windsurf

Similar to Cursor, you can index documents and add custom rules in Windsurf with a slightly different approach.

#### Index Documentation
Windsurf doesn’t local index a resource by just taking your url. It automatically indexes your codebase when opening in the workspace. That means you will need to clone the github repo mentioned above to your local machine.
Remote indexing is only available to Enterprise plan

#### Optimize with Custom Rules
- Create a .windsurfrules file and add it to your project's root directory to set project-specific rules for the AI, such as enforcing coding standards or focusing suggestions on particular frameworks.
- For workspace-wide or global rules, use a global_rules.md file. These can be edited via the Windsurf “Customizations” or “Cascade Memories” settings panel, available in the app.
- Each rule should be concise (under 6,000 characters) and in Markdown format. Rules can be activated manually, always-on, by model decision, or automatically by file path/glob pattern

```
# Windsurf Llama Model Global Rules

## Model Usage & Integration
- Use Meta Llama models (Llama 3, Llama 4 or later) as the default LLM for all tasks requiring a language model.
- Reference the latest official developer resources for installation, basic usage, and authentication:
  - Meta Llama Dev Docs: https://www.llama.com/docs/overview/
  - Official GitHub: https://github.com/meta-llama/llama
  - Llama models on Hugging Face: https://huggingface.co/meta-llama

## Model Deployment
- Provide clear suggestions on different types of deployment (API Cloud based, On-prem, On-device) based on the task and model requirements.
  - Deploying on-prem: https://github.com/meta-llama/llama-cookbook/blob/main/3p-integrations/llama_on_prem.md
  - Deploying with Llama Stack: https://llama-stack.readthedocs.io/en/latest/getting_started/index.html
  - Deploying on-device with ExecuTorch: https://docs.pytorch.org/executorch/stable/llm/llama-demo-android.html

## API Interaction & Migration
- If migrating from other APIs, automatically convert parameter names and prompt formats to Llama's interface. See:
  - Llama Migration Guide: https://www.llama.com/docs/llama-everywhere/migration/ni
  - Llama API Python SDK: https://github.com/meta-llama/llama-api-python
- Point out differences in authentication, token limits, and endpoint structure.

## Fine-Tuning & Customization
- Always provide sample code for fine-tuning Llama models using the latest Meta and Hugging Face recipes:
  - Llama fine-tuning (Meta How-to): https://www.llama.com/docs/how-to-guides/fine-tuning/
  - Recipe of Fine-tuning and get-started guide: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning
  - PEFT: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/quickstart_peft_finetuning.ipynb
  - MultiGPU finetuning: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/multigpu_finetuning.md
  - Finetuning Llama 4: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/finetune_llama4.md
  - Finetuning vision Llama: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/finetune_vision_model.md

## Prompt Engineering Practices
- Use and exemplify the Meta/official Llama chat prompt format (system/user/assistant roles).
- Reference:
  - Llama 4 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/
  - Llama 3.3 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/
  - Llama 3.2 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
  - Llama 3.1 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
  - Hugging Face transformer prompting Llama 3: https://huggingface.co/docs/transformers/en/model_doc/llama3

## Quantization
- Based on task requirements, suggest appropriate Quantization schema and necessary frameworks to minimize memory footprint without sacrificing model quality.
- Official Llama Quantization Guide: https://www.llama.com/docs/how-to-guides/quantization/


## Developer Best Practices
- Ensure all generated code is idiomatic, covered by comments, follows error-handling and input-validation best practices.
- For documentation and onboarding, always provide links to the latest relevant guides:
  - Developer Use Guide: https://www.llama.com/developer-use-guide/
- Recommend including model info and links (e.g., license, documentation) in project-level README files.


## Other Common Use-cases
Text2SQL: https://www.llama.com/resources/cookbook/text2sql_natural_language_to_sql_interface/
Tool Calling: https://www.llama.com/resources/cookbook/toolcalling-with-llama/
Language Expansion: https://www.llama.com/resources/cookbook/extend_llama_to_new_languages/
Email Agent: https://www.llama.com/resources/cookbook/build-email-agent-with-llama3/
Multimodal RAG: https://www.llama.com/resources/cookbook/build-multimodal-rag/
Browser Agent: https://www.llama.com/resources/cookbook/build-an-intelligent-browser-agent/
Customer Service Chatbot: https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/customerservice_chatbots
```

### VSCode/Copilot

#### Index Documentation
GitHub Copilot leverages context from open files, the workspace, or specific instructions to provide relevant code suggestions

#### Optimize with Custom Rules
- Create a .github/copilot-instructions.md file in your workspace to define general coding standards and guidelines for all Copilot chat requests.
- For file- or task-specific rules, use .instructions.md files with the applyTo frontmatter to specify file targeting.
- You can also add settings in settings.json for review, commit message generation, and pull request description instructions, either directly or by referencing instruction files

For more information, refer to https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions

```
Llama Model Global Rules

## Model Usage & Integration
- Use Meta Llama models (Llama 3, Llama 4 or later) as the default LLM for all tasks requiring a language model.
- Reference the latest official developer resources for installation, basic usage, and authentication:
  - Meta Llama Dev Docs: https://www.llama.com/docs/overview/
  - Official GitHub: https://github.com/meta-llama/llama
  - Llama models on Hugging Face: https://huggingface.co/meta-llama

## Model Deployment
- Provide clear suggestions on different types of deployment (API Cloud based, On-prem, On-device) based on the task and model requirements.
  - Deploying on-prem: https://github.com/meta-llama/llama-cookbook/blob/main/3p-integrations/llama_on_prem.md
  - Deploying with Llama Stack: https://llama-stack.readthedocs.io/en/latest/getting_started/index.html
  - Deploying on-device with ExecuTorch: https://docs.pytorch.org/executorch/stable/llm/llama-demo-android.html

## API Interaction & Migration
- If migrating from other APIs, automatically convert parameter names and prompt formats to Llama's interface. See:
  - Llama Migration Guide: https://www.llama.com/docs/llama-everywhere/migration/ni
  - Llama API Python SDK: https://github.com/meta-llama/llama-api-python
- Point out differences in authentication, token limits, and endpoint structure.

## Fine-Tuning & Customization
- Always provide sample code for fine-tuning Llama models using the latest Meta and Hugging Face recipes:
  - Llama fine-tuning (Meta How-to): https://www.llama.com/docs/how-to-guides/fine-tuning/
  - Recipe of Fine-tuning and get-started guide: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning
  - PEFT: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/quickstart_peft_finetuning.ipynb
  - MultiGPU finetuning: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/multigpu_finetuning.md
  - Finetuning Llama 4: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/finetune_llama4.md
  - Finetuning vision Llama: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/finetune_vision_model.md

## Prompt Engineering Practices
- Use and exemplify the Meta/official Llama chat prompt format (system/user/assistant roles).
- Reference:
  - Llama 4 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/
  - Llama 3.3 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/
  - Llama 3.2 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
  - Llama 3.1 Prompt Template and Guide: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
  - Hugging Face transformer prompting Llama 3: https://huggingface.co/docs/transformers/en/model_doc/llama3

## Quantization
- Based on task requirements, suggest appropriate Quantization schema and necessary frameworks to minimize memory footprint without sacrificing model quality.
- Official Llama Quantization Guide: https://www.llama.com/docs/how-to-guides/quantization/


## Developer Best Practices
- Ensure all generated code is idiomatic, covered by comments, follows error-handling and input-validation best practices.
- For documentation and onboarding, always provide links to the latest relevant guides:
  - Developer Use Guide: https://www.llama.com/developer-use-guide/
- Recommend including model info and links (e.g., license, documentation) in project-level README files.


## Other Common Use-cases
Text2SQL: https://www.llama.com/resources/cookbook/text2sql_natural_language_to_sql_interface/
Tool Calling: https://www.llama.com/resources/cookbook/toolcalling-with-llama/
Language Expansion: https://www.llama.com/resources/cookbook/extend_llama_to_new_languages/
Email Agent: https://www.llama.com/resources/cookbook/build-email-agent-with-llama3/
Multimodal RAG: https://www.llama.com/resources/cookbook/build-multimodal-rag/
Browser Agent: https://www.llama.com/resources/cookbook/build-an-intelligent-browser-agent/
Customer Service Chatbot: https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/customerservice_chatbots
```
