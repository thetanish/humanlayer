from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel

from channels import dm_with_ceo
from humanlayer.core.approval import HumanLayer
from llamaindex import SimpleChatEngine, ToolSpec

load_dotenv()

hl = HumanLayer()

task_prompt = """

You are the linkedin inbox assistant. You check on
the CEO's linkedin inbox and decide if there are any messages
that seem interesting, then contact the human in slack with a summary.

don't provide detail on spam-looking messages, or messages
that appear to be selling a service or software

You can offer to perform actions like schedule time.

Example slack dm to send:

Your inbox for today includes 4 spam messages,
and 1 message from Devin who seems interested in your
product - [here's the link](https://linkedin.com/in/devin).

Terri has still not responded to your question about scheduling an onboarding call.

Would you like me to respond to Devin with your availability?

"""


class LinkedInMessage(BaseModel):
    from_name: str
    date: str
    message: str


class LinkedInThread(BaseModel):
    thread_id: str
    thread_url: str
    with_name: str
    messages: list[LinkedInMessage]


def get_time() -> str:
    """get the current time"""
    return datetime.now().isoformat()


def get_linkedin_threads() -> list[LinkedInThread]:
    """get the linkedin threads in the inbox"""
    return [
        LinkedInThread(
            thread_id="123",
            thread_url="https://linkedin.com/in/msg/123",
            with_name="Danny",
            messages=[
                LinkedInMessage(
                    message="Hello, I am wondering if you are interested to try our excellent offshore "
                    "developer service",
                    from_name="Danny",
                    date="2024-08-17",
                )
            ],
        ),
        LinkedInThread(
            thread_id="124",
            with_name="Sarah",
            thread_url="https://linkedin.com/in/msg/124",
            messages=[
                LinkedInMessage(
                    message="Hello, I am interested in your product, what's the best way to get started",
                    from_name="Sarah",
                    date="2024-08-16",
                )
            ],
        ),
        LinkedInThread(
            thread_id="125",
            with_name="Terri",
            thread_url="https://linkedin.com/in/msg/125",
            messages=[
                LinkedInMessage(
                    message="Hello, I am interested in your product, what's the best way to get started",
                    from_name="Terri",
                    date="2024-08-12",
                ),
                LinkedInMessage(
                    message="I would be happy to give you a demo - please let me know when you're "
                    "available, or you can book time at http://calendly.com/im-the-ceo",
                    from_name="you",
                    date="2024-08-12",
                ),
            ],
        ),
    ]


@hl.require_approval(contact_channel=dm_with_ceo)
def send_linkedin_message(thread_id: str, to_name: str, msg: str) -> str:
    """send a message in a thread in LinkedIn"""
    return f"message successfully sent to {to_name}"


tools = [
    ToolSpec.from_function(get_linkedin_threads, description="Fetch LinkedIn threads from the inbox."),
    ToolSpec.from_function(send_linkedin_message, description="Send a message in a LinkedIn thread."),
    ToolSpec.from_function(
        hl.human_as_tool(contact_channel=dm_with_ceo),
        description="Allow the agent to contact the CEO for approval."
    ),
]

llm = SimpleChatEngine(model="gpt-4", temperature=0, tools=tools)

if __name__ == "__main__":
    result = llm.run(task_prompt)
    print("\n\n----------Result----------\n\n")
    print(result)
