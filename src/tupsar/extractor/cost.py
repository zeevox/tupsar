"""Token cost tracker for LLM models."""

import dataclasses
from decimal import Decimal


@dataclasses.dataclass
class CostTracker:
    """LLM cost tracker."""

    input_cost_per_token: Decimal
    output_cost_per_token: Decimal

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Track token usage adding it to the total."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def input_cost(self) -> Decimal:
        """Return the total cost of input tokens."""
        return self.input_cost_per_token * self.input_tokens

    @property
    def output_cost(self) -> Decimal:
        """Return the total cost of output tokens."""
        return self.output_cost_per_token * self.output_tokens

    @property
    def total_cost(self) -> Decimal:
        """Return the total cost of input and output tokens."""
        return self.input_cost + self.output_cost

    def __str__(self) -> str:
        """Get a string representation of the costs."""
        return (
            f"Input: ${self.input_cost} "
            f"Output: ${self.output_cost} "
            f"Total: ${self.total_cost}"
        )
