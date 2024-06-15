import re
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from openai import OpenAI
from rich import print
import os

load_dotenv()

# check if the environment variable is set
required_env_vars = ["OPENAI_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"]
for ev in required_env_vars:
    if os.getenv(ev) is None:
        print(f"Please set environment variable: {ev}")
        exit(1)

client = wrap_openai(OpenAI())

# chat_completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": "Hello world"}
#     ]
# )

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    @traceable
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute() or "Sorry, there was a system error."
        self.messages.append({"role": "system", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.8,
            messages=self.messages
        )
        return completion.choices[0].message.content

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()

@traceable
def calculate(what):
    return eval(what)

@traceable
def average_dog_weight(name):
    if "Scottish Terrier" in name:
        return "Scottish Terrier average 20 lbs"
    elif "Collie" in name:
        return "a Border Collie average 37 lbs"
    elif "Toy Poodle" in name:
        return "a Toy Poodle average 7 lbs"
    else:
        return "an average dog average 50 lbs"


known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

# abot = Agent(prompt)
# result = abot("How much does a toy poodle weigh?")
# print(result)

# result = average_dog_weight("Toy Poodle")
# print(result)

# next_prompt = f"Observation: {result}"
# abot(next_prompt)
# abot.messages

action_re = re.compile('^Action: (\w+): (.*)$')

@traceable
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
                for a in result.split('\n')
                if action_re.match(a) is not None
        ]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
        else:
            print(" -- no actions found, stopping")
            return

question = """I have 3 dogs, a border collie, a scottish terrier and a corgi. \
What is their combined weight"""

query(question)
