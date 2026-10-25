"""``forget_me``: the right to be forgotten, as a tool.

The tool takes no arguments: the scope is the session's own ``app_name`` and
``user_id``, so a model (or a prompt injection) can't ask it to erase someone else.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools.tool_context import ToolContext

from travel_concierge.callbacks import FORGET_KEY
from travel_concierge.memory.base import MemoryAdmin


async def forget_me(tool_context: ToolContext) -> dict[str, Any]:
    """Delete everything remembered about the current user, in every past session.

    Call this only when the user explicitly asks you to forget them or delete their data.
    """
    ctx = tool_context.get_invocation_context()
    service = ctx.memory_service
    app_name, user_id = ctx.session.app_name, ctx.session.user_id
    tool_context.state[FORGET_KEY] = True
    if service is None:
        return {"status": "error", "error": "no memory service is configured"}
    if isinstance(service, MemoryAdmin):
        count = await service.forget_user(app_name=app_name, user_id=user_id)
        return {"status": "forgotten", "deleted": count}
    return {
        "status": "error",
        "error": f"{type(service).__name__} cannot forget a whole user; use the local or vertex "
        "backend",
    }
