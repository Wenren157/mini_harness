from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class AgentScope:
    """
    Agent资源隔离上下文

    每个Agent拥有独立:
    - workspace
    - metadata

    后续扩展:
    - memory namespace
    - context namespace
    - session id
    """

    agent_id: str
    workspace: str

    metadata: Dict[str, Any] | None = None


    def __post_init__(self):

        self.workspace = str(Path(self.workspace).absolute())

        Path(
            self.workspace
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        if self.metadata is None:
            self.metadata = {}