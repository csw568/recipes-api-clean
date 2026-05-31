import os
import asyncio
from typing import Any

import dotenv
from github import Github
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate

dotenv.load_dotenv()

repository_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

git = None
repo = None


def get_repo():
    global git, repo
    if repo is None:
        git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else Github()
        if not repository_name:
            raise ValueError("REPOSITORY is not set.")
        repo = git.get_repo(repository_name)
    return repo


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Use this tool to get pull request details from a PR number."""
    pull_request = get_repo().get_pull(pr_number)
    commit_shas = [c.sha for c in pull_request.get_commits()]
    return {
        "user": pull_request.user.login,
        "title": pull_request.title or "",
        "body": pull_request.body or "No PR description provided.",
        "diff_url": pull_request.diff_url or "",
        "state": pull_request.state or "",
        "head_sha": pull_request.head.sha,
        "commit_shas": commit_shas,
    }


def get_file_contents(file_path: str) -> str:
    """Use this tool to get the contents of a file in the repository from its path."""
    file_content = get_repo().get_contents(file_path)
    return file_content.decoded_content.decode("utf-8")


def get_pr_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Use this tool to get changed-file details for a commit SHA."""
    commit = get_repo().get_commit(commit_sha)
    changed_files = []
    for f in commit.files:
        changed_files.append(
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": getattr(f, "patch", None),
            }
        )
    return changed_files


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Useful for saving the gathered repository context to workflow state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for saving the draft pull request review comment to workflow state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for saving the final reviewed pull request comment to workflow state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review."


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Use this tool to post the final review comment to a pull request on GitHub."""
    pull_request = get_repo().get_pull(pr_number)
    pull_request.create_review(body=comment, event="COMMENT")
    return f"Review posted successfully to PR #{pr_number}."


def create_workflow():
    llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
    file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
    pr_commit_details_tool = FunctionTool.from_defaults(fn=get_pr_commit_details)
    add_context_tool = FunctionTool.from_defaults(fn=add_context_to_state)
    add_comment_tool = FunctionTool.from_defaults(fn=add_comment_to_state)
    add_final_review_tool = FunctionTool.from_defaults(fn=add_final_review_to_state)
    post_review_tool = FunctionTool.from_defaults(fn=post_review_to_github)

    context_agent = FunctionAgent(
        llm=llm,
        name="ContextAgent",
        description="Gathers all the needed pull request and repository context for review.",
        tools=[
            pr_details_tool,
            file_contents_tool,
            pr_commit_details_tool,
            add_context_tool,
        ],
        system_prompt="""You are the context gathering agent. When gathering context, you MUST gather:
- The details: user, title, body, diff_url, state, and head_sha;
- Changed files;
- Any requested repository files;
Once you gather the requested info, you MUST hand control back to the CommentorAgent.""",
        can_handoff_to=["CommentorAgent"],
    )

    commentor_agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        description="Uses the context gathered by the context agent to draft a pull review comment.",
        tools=[add_comment_tool],
        system_prompt="""You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
- Request the PR details, changed files, and any other repository files you may need from the ContextAgent.
- Once you have all the needed information, write a good ~200-300 word review in markdown format detailing:
  - What is good about the PR?
  - Did the author follow ALL contribution rules? What is missing?
  - Are there tests for new functionality? If there are new models, are there migrations for them? Use the diff to determine this.
  - Are new endpoints documented? Use the diff to determine this.
  - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
- If you need any additional details, hand off to the ContextAgent.
- You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
- Directly address the author.""",
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    )

    review_and_posting_agent = FunctionAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        description="Reviews the drafted PR comment, requests rewrites if needed, saves the final review, and posts it to GitHub.",
        tools=[add_final_review_tool, post_review_tool],
        system_prompt="""You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
- The review must:
- Be a ~200-300 word review in markdown format.
- Specify what is good about the PR.
- Say whether the author followed ALL contribution rules and what is missing.
- Include notes on test availability for new functionality. If there are new models, mention whether migrations exist.
- Include notes on whether new endpoints were documented.
- Include suggestions on which lines could be improved upon, and these lines should be quoted.
If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
When you are satisfied, save the final review and post it to GitHub.""",
        can_handoff_to=["CommentorAgent"],
    )

    return AgentWorkflow(
        agents=[context_agent, commentor_agent, review_and_posting_agent],
        root_agent=review_and_posting_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "draft_comment": "",
            "final_review": "",
        },
    )


async def main():
    if not pr_number:
        raise ValueError("PR_NUMBER is not set.")

    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    workflow_agent = create_workflow()
    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print(f"\n\nFinal response: {event.response.content}")
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    if git is not None:
        git.close()
