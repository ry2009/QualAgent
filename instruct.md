challenge
Multi-Agent QA System Based on Agent-S Architecture
Goal: Build a multi-agent LLM-powered system that functions as a full-stack mobile QA team on
top of Agent-S and android_world.

Architecture Overview
You’ll extend the modular architecture of Agent-S, where agents use LLMs + learned policies
to collaboratively complete tasks in a mobile UI environment.
Required Agents:
1. Planner Agent - Parses high-level QA goals and decomposes them into subgoals.
2. Executor Agent - Executes subgoals in the Android UI environment with grounded
mobile gestures.
3. Verifier Agent - Determines whether the app behaves as expected after each step.
4. Supervisor Agent - Reviews entire test episodes and proposes improvements.

Setup + Planner + Executor (Grounded in Android World)
Tasks:
1. Fork or clone Agent-S
○ Use its modular messaging structure and agent execution framework.

Python

Python
○ Customize agent classes for your QA task.
2. Integrate android_world
○ Choose a task (e.g., "settings_wifi", "clock_alarm", "email_search")
○ Use AndroidEnv to simulate interactions:

env = AndroidEnv(task_name="settings_wifi")
obs = env.reset()

3. Implement Planner Agent
○ Input: "Test turning Wi-Fi on and off"
○ Output: Sequence of actionable, app-specific subgoals.
○ Agent should reason about modal states and update plan dynamically.
4. Implement Executor Agent
○ Receives a subgoal, inspects current UI hierarchy (ui_tree)
○ Selects grounded actions (touch, type, scroll) using:

env.step({
"action_type": "touch",
"element_id": "<ui_element_id>"
})

Deliverables:
● Working multi-agent pipeline using Agent-S components

● Successful test execution of 1 full QA task

Verifier Agent + Error Handling
Tasks:
1. Implement Verifier Agent
○ Receives Planner Goal, Executor Result, and UI State
○ Determines:
■ If the current state matches expectation (pass/fail)
■ If there’s a functional bug (e.g., missing screen, wrong toggle state)
○ Leverage heuristics + LLM reasoning over the UI hierarchy
2. Implement dynamic replanning
○ If Verifier signals a problem:
■ Planner is prompted to adapt steps mid-execution
■ Example: Pop-up block → skip or dismiss

3. Log all interactions
○ Per-agent decisions
○ Final verdict: passed/failed/bug detected

Deliverables:
● verifier_agent.py
● Recovery and replanning logic in Planner
● QA logs in JSON format

Supervisor Agent + Evaluation
Tasks:
1. Simulate or record visual traces
○ Use env.render(mode="rgb_array") to create frame-by-frame UI images
2. Implement Supervisor Agent
○ Processes the full test trace (images + logs)
○ Uses Gemini 2.5 (or mock LLM) to:
■ Suggest prompt improvements
■ Identify poor plans or failures
■ Recommend test coverage expansion

3. Create an evaluation report
○ Bug detection accuracy
○ Agent recovery ability
○ Supervisor feedback effectiveness

Bonus
Consider incorporating the android_in_the_wild dataset into the multi-agent QA
architecture to enhance training, evaluation, and robustness of the system:
This dataset contains:
● Screen recordings and UI traces from real user sessions across thousands of Android
apps.

● Semantic and visual diversity, covering notifications, modals, errors, dialogs, dark
mode, and inconsistent layouts.
● Useful for training and evaluating agents on real-world complexity and UI distribution
shifts.
Task: Use 3–5 videos from android_in_the_wild. For each:
1. Generate the task prompt the user was likely trying to complete.
2. Have the multi-agent system reproduce that flow inside android_world or a
mocked AndroidEnv.
3. Compare video trace of the agent vs. ground truth.
4. Score accuracy, robustness, and generalization.

Further explorations: How the dataset can be used to improve
the agents
Planner Agent
● Use cases: Pretraining or fine-tuning on real user session traces to learn how humans
sequence app tasks.
● Extract action plans from session metadata (if available), or generate pseudo-labels
using GPT/Gemini from session captions.

Executor Agent
● Train visual grounding and gesture control on:
○ Touchpoint locations
○ Motion paths
○ Tap/scroll semantics across varied layouts
● Helps generalize execution across device types, screen sizes, and layout randomness.

Verifier Agent

● Evaluate agent predictions against ground truth recordings to:
○ Detect false positives (agent thinks it's a bug but it's not)
○ Detect false negatives (agent misses layout/flow bugs)
● Can train a contrastive or discriminative model to separate "expected" vs. "anomalous"
flows using this diverse data.

Supervisor Agent
● Use the dataset’s recorded videos as input to Gemini 2.5 or GPT-4V to:
○ Generate original test prompts
○ Identify flaws or improvements in the agent’s approach
○ Suggest new ways to handle non-deterministic flows (e.g., unskippable modals)


for context about android world:
README
Apache-2.0 license
AndroidWorld
Unittests

Website • Paper • Tasks • Leaderboard

Overview

AndroidWorld is an environment for building and benchmarking autonomous computer control agents.

It runs on a live Android emulator and contains a highly reproducible benchmark of 116 hand-crafted tasks across 20 apps, which are dynamically instantiated with randomly-generated parameters to create millions of unique task variations.

In addition to the built-in tasks, AndroidWorld also supports the popular web benchmark, MiniWoB++ from Liu et al..

Key features of AndroidWorld include:

📝 116 diverse tasks across 20 real-world apps
🎲 Dynamic task instantiation for millions of unique variations
🏆 Durable reward signals for reliable evaluation
🐳 Experimental Docker Support for simplified setup and consistent environments (as of 06/02/2025)
🌐 Open environment with access to millions of Android apps and websites
💾 Lightweight footprint (2 GB memory, 8 GB disk)
🔧 Extensible design to easily add new tasks and benchmarks
🖥️ Integration with MiniWoB++ web-based tasks
See demo videos on our website. o

Installation
Set up the Android Emulator

Download Android Studio here
Create an Android Virtual Device (AVD) by following these instructions. For hardware select Pixel 6, for System Image select Tiramisu, API Level 33, and choose AVD name as AndroidWorldAvd. Watch the setup video.
Launch the Android Emulator from the command line

Launch the emulator from the command line, not using the Android Studio UI, with the -grpc 8554 flag which is needed communication with accessibility forwarding app.

# Typically it's located in ~/Android/Sdk/emulator/emulator or
# ~/Library/Android/sdk/emulator/emulator
EMULATOR_NAME=AndroidWorldAvd # From previous step
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
[Optional] It's recommended to use conda, which you can download here.

conda create -n android_world python=3.11.8
conda activate android_world
Install AndroidWorld. Note: Python 3.11 or above is required.

git clone https://github.com/google-research/android_world.git
cd ./android_world
pip install -r requirements.txt
python setup.py install
Add model provider APIs as environment variables.

# Add to .bashrc.
export OPENAI_API_KEY=your-key
export GCP_API_KEY=your-key
Install ffmpeg, if not already installed.

# Linux (Ubuntu/Debian)
# sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
Quickstart
Run the minimal_task_runner.py script to see the basic mechanics of AndroidWorld components. It initializes the environment, sets up a task, and runs the default agent, M3A, on it.

python minimal_task_runner.py --task=ContactsAddContact
If you don't specify a task, a random task will be selected. NOTE: If you want to try open-source apps, i.e. not included with Android OS, please run --perform_emulator_setup in the script below.

Docker Support (Experimental)
AndroidWorld now offers Docker support. This allows you to run the Android environment and server within a Docker container, which can simplify setup and ensure a consistent environment.

Note: This feature is experimental and has not been extensively tested.

Build the Docker image:

Navigate to the root directory of the android_world repository and run:

docker build -t android_world:latest .
Run the Docker container:

docker run --privileged -p 5000:5000 -it android_world:latest
This will start the Android emulator and the FastAPI server inside the container. The server will be accessible on http://localhost:5000.

Interact with the environment: You can see the scripts/run_suite_on_docker.py script as an example client to interact with the Android environment server running in Docker.

Note for Apple Silicon users
There are known issues with installing the required package emulator on ARM chips (Apple Silicon). To get around this, if building images locally, you should build images for the AMD64/x86_64 instruction set, by running:

docker buildx build --platform linux/amd64 -t android-emulator:latest .
Note, running in a Docker container like this, on an Apple Silicon device will run quite slowly compared to running the Android Device and Emulator natively (because you end up running an Android Emulator inside a Linux Emulator...).

Run the benchmark
Note: Task Step Limits Update As of 11/18/2024, the max_steps/step_budget for each task in AndroidWorld have been updated to approximately 2x the human average completion time. This adjustment ensures agents have sufficient time to complete tasks, while also reducing overhead of running thebenchmark. Here are the per-task updates.

python run.py \
  --suite_family=android_world \
  --agent_name=t3a_gpt4 \
  --perform_emulator_setup \
  --tasks=ContactsAddContact,ClockStopWatchRunning \  # Optional: Just run on a subset.
The first time you run this script, you must install the necessary apps and set permissions by specifying --perform_emulator_setup. This is a one-time setup. It may take several minutes depending on the connection speed.

Above we specify the optional --tasks flag to run on a subset of tasks. Leave it empty to run on the entire AndroidWorld suite.

The n_task_combinations argument specifies how many parameter permutations to use for each task. For example, for an SMS task, it would correspond to different phone number/message combinations for each run.

If a run fails part-way through, you can resume it by re-running the script with the --checkpoint_dir flag pointing to the output directory from the original run.

Running MiniWoB++ tasks
To run the MiniWoB++ web-based tasks in AndroidWorld, simply set --suite_family=miniwob and --perform_emulator_setup in the command above.

A key advantage of running MiniWoB++ tasks is that common input elements are rendered as native, commonly used Android UI widgets, rather than as HTML. Thus agents must learn to use universal widgets such as time- and date-pickers:



Create your own agent
In addition to the agents we provide here, you can also easily create your own agent and run the benchmark with it as follows.

Create an agent class that inherits from EnvironmentInteractingAgent and implement the step method. In the current workflow, the agent tries to complete a task in a for loop. In each round, the step method will be called and this is where you implement your agent's logic. A typical approach involves first gathering information like the current screenshot, the UI elements (like buttons, icons) through the AndroidEnv instance within the agent, selecting one of the supported actions, executing it through the AndroidEnv and returning an AgentInteractionResult. The done property on AgentInteractionResult should be set to true to indicate that the task is finished.

Import your agent in run.py and also add it into the _get_agent method which takes in your agent's name and return an instance of it.

Now you can run the benchmark with your new agent using the command above with the agent_name flag changed to your agent's name.

Adding new tasks
Please see the guide on adding new tasks to AndroidWorld.

Citation
If you use our environment or data, please cite our paper:

@misc{rawles2024androidworlddynamicbenchmarkingenvironment,
      title={AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents},
      author={Christopher Rawles and Sarah Clinckemaillie and Yifan Chang and Jonathan Waltz and Gabrielle Lau and Marybeth Fair and Alice Li and William Bishop and Wei Li and Folawiyo Campbell-Ajala and Daniel Toyama and Robert Berry and Divya Tyamagundlu and Timothy Lillicrap and Oriana Riva},
      year={2024},
      eprint={2405.14573},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.14573},
}
This is not an officially supported Google product.

About
AndroidWorld is an environment and benchmark for autonomous agents

Resources
 Readme
License
 Apache-2.0 license
 Activity
 Custom properties
Stars
 361 stars
Watchers
 7 watching
Forks
 62 forks
Report repository
Releases
No releases published
Packages
No packages published
Contributors
11
@clink42
@crawles
@NingLi670
@wichersn
@A-Mahla
@rossamurphy
@alice9210
@lukegb
@hoisie
@gabrielle-lau
Deployments
37
 github-pages 7 months ago
+ 36 deployments
Languages
Python
39.6%
 
HTML
29.9%
 
JavaScript
26.1%
 
Kotlin
2.6%
 
CSS
1.3%
 
Starlark
0.2%
 
Other
0.3%
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Conta


for context abotu agent-s:
README
Apache-2.0 license
Logo Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
  🌐 [S2 blog]  📄 [S2 Paper (COLM 2025)]  🎥 [S2 Video]

  🌐 [S1 blog]  📄 [S1 Paper (ICLR 2025)]  🎥 [S1 Video]

  simular-ai%2FAgent-S | Trendshift

Discord    PyPI Downloads

Deutsch | Español | français | 日本語 | 한국어 | Português | Русский | 中文
🥳 Updates
 2025/07/07: The Agent S2 paper is accepted to COLM 2025! See you in Montreal!
 2025/04/01: Released the Agent S2 paper with new SOTA results on OSWorld, WindowsAgentArena, and AndroidWorld!
 2025/03/12: Released Agent S2 along with v0.2.0 of gui-agents, the new state-of-the-art for computer use agents (CUA), outperforming OpenAI's CUA/Operator and Anthropic's Claude 3.7 Sonnet Computer-Use!
 2025/01/22: The Agent S paper is accepted to ICLR 2025!
 2025/01/21: Released v0.1.2 of gui-agents library, with support for Linux and Windows!
 2024/12/05: Released v0.1.0 of gui-agents library, allowing you to use Agent-S for Mac, OSWorld, and WindowsAgentArena with ease!
 2024/10/10: Released the Agent S paper and codebase!
Table of Contents
💡 Introduction
🎯 Current Results
🛠️ Installation & Setup
🚀 Usage
🤝 Acknowledgements
💬 Citation
💡 Introduction


Welcome to Agent S, an open-source framework designed to enable autonomous interaction with computers through Agent-Computer Interface. Our mission is to build intelligent GUI agents that can learn from past experiences and perform complex tasks autonomously on your computer.

Whether you're interested in AI, automation, or contributing to cutting-edge agent-based systems, we're excited to have you here!

🎯 Current Results

Results of Agent S2's Successful Rate (%) on the OSWorld full test set using Screenshot input only.

Benchmark	Agent S2	Previous SOTA	Δ improve
OSWorld (15 step)	27.0%	22.7% (UI-TARS)	+4.3%
OSWorld (50 step)	34.5%	32.6% (OpenAI CUA)	+1.9%
WindowsAgentArena	29.8%	19.5% (NAVI)	+10.3%
AndroidWorld	54.3%	46.8% (UI-TARS)	+7.5%
🛠️ Installation & Setup
Note: Our agent returns pyautogui code and is intended for a single monitor screen.

❗Warning❗: If you are on a Linux machine, creating a conda environment will interfere with pyatspi. As of now, there's no clean solution for this issue. Proceed through the installation without using conda or any virtual environment.

⚠️Disclaimer⚠️: To leverage the full potential of Agent S2, we utilize UI-TARS as a grounding model (7B-DPO or 72B-DPO for better performance). They can be hosted locally, or on Hugging Face Inference Endpoints. Our code supports Hugging Face Inference Endpoints. Check out Hugging Face Inference Endpoints for more information on how to set up and query this endpoint. However, running Agent S2 does not require this model, and you can use alternative API based models for visual grounding, such as Claude.

Install the package:

pip install gui-agents
Set your LLM API Keys and other environment variables. You can do this by adding the following line to your .bashrc (Linux), or .zshrc (MacOS) file.

export OPENAI_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
Alternatively, you can set the environment variable in your Python script:

import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
We also support Azure OpenAI, Anthropic, Gemini, Open Router, and vLLM inference. For more information refer to models.md.

Setup Retrieval from Web using Perplexica
Agent S works best with web-knowledge retrieval. To enable this feature, you need to setup Perplexica:

Ensure Docker Desktop is installed and running on your system.

Navigate to the directory containing the project files.

 cd Perplexica
 git submodule update --init
Rename the sample.config.toml file to config.toml. For Docker setups, you need only fill in the following fields:

OPENAI: Your OpenAI API key. You only need to fill this if you wish to use OpenAI's models.

OLLAMA: Your Ollama API URL. You should enter it as http://host.docker.internal:PORT_NUMBER. If you installed Ollama on port 11434, use http://host.docker.internal:11434. For other ports, adjust accordingly. You need to fill this if you wish to use Ollama's models instead of OpenAI's.

GROQ: Your Groq API key. You only need to fill this if you wish to use Groq's hosted models.

ANTHROPIC: Your Anthropic API key. You only need to fill this if you wish to use Anthropic models.

Note: You can change these after starting Perplexica from the settings dialog.

SIMILARITY_MEASURE: The similarity measure to use (This is filled by default; you can leave it as is if you are unsure about it.)

Ensure you are in the directory containing the docker-compose.yaml file and execute:

docker compose up -d
Export your Perplexica URL using the port found in the docker-compose.yaml file Under app/ports, you'll see 3000:3000. The port is the left-hand number (in this case, 3000).

export PERPLEXICA_URL=http://localhost:{port}/api/search
Our implementation of Agent S incorporates the Perplexica API to integrate a search engine capability, which allows for a more convenient and responsive user experience. If you want to tailor the API to your settings and specific requirements, you may modify the URL and the message of request parameters in agent_s/query_perplexica.py. For a comprehensive guide on configuring the Perplexica API, please refer to Perplexica Search API Documentation. For a more detailed setup and usage guide, please refer to the Perplexica Repository.

❗Warning❗: The agent will directly run python code to control your computer. Please use with care.

🚀 Usage
Note: Our best configuration uses Claude 3.7 with extended thinking and UI-TARS-72B-DPO. If you are unable to run UI-TARS-72B-DPO due to resource constraints, UI-TARS-7B-DPO can be used as a lighter alternative with minimal performance degradation.

CLI
Run Agent S2 with a specific model (default is gpt-4o):

agent_s2 \
  --provider "anthropic" \
  --model "claude-3-7-sonnet-20250219" \
  --grounding_model_provider "anthropic" \
  --grounding_model "claude-3-7-sonnet-20250219" \
Or use a custom endpoint:

agent_s2 \
  --provider "anthropic" \
  --model "claude-3-7-sonnet-20250219" \
  --endpoint_provider "huggingface" \
  --endpoint_url "<endpoint_url>/v1/"
Main Model Settings
--provider, --model
Purpose: Specifies the main generation model
Supports: all model providers in models.md
Default: --provider "anthropic" --model "claude-3-7-sonnet-20250219"
--model_url, --model_api_key
Purpose: Specifies the custom endpoint for the main generation model and your API key
Note: These are optional. If not specified, gui-agents will default to your environment variables for the URL and API key.
Supports: all model providers in models.md
Default: None
Grounding Configuration Options
You can use either Configuration 1 or Configuration 2:

(Default) Configuration 1: API-Based Models
--grounding_model_provider, --grounding_model
Purpose: Specifies the model for visual grounding (coordinate prediction)
Supports: all model providers in models.md
Default: --grounding_model_provider "anthropic" --grounding_model "claude-3-7-sonnet-20250219"
❗Important❗ --grounding_model_resize_width
Purpose: Some API providers automatically rescale images. Therefore, the generated (x, y) will be relative to the rescaled image dimensions, instead of the original image dimensions.
Supports: Anthropic rescaling
Tips: If your grounding is inaccurate even for very simple queries, double check your rescaling width is correct for your machine's resolution.
Default: --grounding_model_resize_width 1366 (Anthropic)
Configuration 2: Custom Endpoint
--endpoint_provider

Purpose: Specifies the endpoint provider
Supports: HuggingFace TGI, vLLM, Open Router
Default: None
--endpoint_url

Purpose: The URL for your custom endpoint
Default: None
--endpoint_api_key

Purpose: Your API key for your custom endpoint
Note: This is optional. If not specified, gui-agents will default to your environment variables for the API key.
Default: None
Note: Configuration 2 takes precedence over Configuration 1.

This will show a user query prompt where you can enter your query and interact with Agent S2. You can use any model from the list of supported models in models.md.

gui_agents SDK
First, we import the necessary modules. AgentS2 is the main agent class for Agent S2. OSWorldACI is our grounding agent that translates agent actions into executable python code.

import pyautogui
import io
from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI

# Load in your API keys.
from dotenv import load_dotenv
load_dotenv()

current_platform = "linux"  # "darwin", "windows"
Next, we define our engine parameters. engine_params is used for the main agent, and engine_params_for_grounding is for grounding. For engine_params_for_grounding, we support the Claude, GPT series, and Hugging Face Inference Endpoints.

engine_params = {
  "engine_type": provider,
  "model": model,
  "base_url": model_url,     # Optional
  "api_key": model_api_key,  # Optional
}

# Grounding Configuration 1: Load the grounding engine from an API based model
grounding_model_provider = "<your_grounding_model_provider>"
grounding_model = "<your_grounding_model>"
grounding_model_resize_width = 1366
screen_width, screen_height = pyautogui.size()

engine_params_for_grounding = {
  "engine_type": grounding_model_provider,
  "model": grounding_model,
  "grounding_width": grounding_model_resize_width,
  "grounding_height": screen_height
  * grounding_model_resize_width
  / screen_width,
}

# Grounding Configuration 2: Load the grounding engine from a HuggingFace TGI endpoint
endpoint_provider = "<your_endpoint_provider>"
endpoint_url = "<your_endpoint_url>"
endpoint_api_key = "<your_api_key>"

engine_params_for_grounding = {
  "engine_type": endpoint_provider,
  "base_url": endpoint_url,
  "api_key": endpoint_api_key,  # Optional
}
Then, we define our grounding agent and Agent S2.

grounding_agent = OSWorldACI(
    platform=current_platform,
    engine_params_for_generation=engine_params,
    engine_params_for_grounding=engine_params_for_grounding
)

agent = AgentS2(
  engine_params,
  grounding_agent,
  platform=current_platform,
  action_space="pyautogui",
  observation_type="screenshot",
  search_engine="Perplexica",  # Assuming you have set up Perplexica.
  embedding_engine_type="openai"  # Supports "gemini", "openai"
)
Finally, let's query the agent!

# Get screenshot.
screenshot = pyautogui.screenshot()
buffered = io.BytesIO() 
screenshot.save(buffered, format="PNG")
screenshot_bytes = buffered.getvalue()

obs = {
  "screenshot": screenshot_bytes,
}

instruction = "Close VS Code"
info, action = agent.predict(instruction=instruction, observation=obs)

exec(action[0])
Refer to gui_agents/s2/cli_app.py for more details on how the inference loop works.

Downloading the Knowledge Base
Agent S2 uses a knowledge base that continually updates with new knowledge during inference. The knowledge base is initially downloaded when initializing AgentS2. The knowledge base is stored as assets under our GitHub Releases. The AgentS2 initialization will only download the knowledge base for your specified platform and agent version (e.g s1, s2). If you'd like to download the knowledge base programmatically, you can use the following code:

download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform="linux"  # "darwin", "windows"
)
This will download Agent S2's knowledge base for Linux from release tag v0.2.2 to the kb_data directory. Refer to our GitHub Releases or release tags that include the knowledge bases.

OSWorld
To deploy Agent S2 in OSWorld, follow the OSWorld Deployment instructions.

WindowsAgentArena
To deploy Agent S2 in WindowsAgentArena, follow the WindowsAgentArena Deployment Instructions.

💬 Citations
If you find this codebase useful, please cite

@misc{Agent-S2,
      title={Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents}, 
      author={Saaket Agashe and Kyle Wong and Vincent Tu and Jiachen Yang and Ang Li and Xin Eric Wang},
      year={2025},
      eprint={2504.00906},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.00906}, 
}

@inproceedings{Agent-S,
    title={{Agent S: An Open Agentic Framework that Uses Computers Like a Human}},
    author={Saaket Agashe and Jiuzhou Han and Shuyu Gan and Jiachen Yang and Ang Li and Xin Eric Wang},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://arxiv.org/abs/2410.08164}
}
